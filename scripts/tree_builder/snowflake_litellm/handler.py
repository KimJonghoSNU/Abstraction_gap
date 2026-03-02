#!/usr/bin/env python3
"""
Snowflake Cortex AI Custom Handler for LiteLLM
SQL-based approach using snowflake.connector and SNOWFLAKE.CORTEX.COMPLETE
"""

import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.errors import DatabaseError
from snowflake.connector.errors import Error as SnowflakeError
from snowflake.connector.errors import ProgrammingError

from litellm import CustomLLM
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    InvalidRequestError,
    RateLimitError,
    Timeout,
)
from litellm.types.utils import (
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class SnowflakeConnectionManager:
    """Snowflake 연결 관리 클래스"""

    def __init__(
        self,
        account: str,
        user: str,
        password: str,
        warehouse: str = "COMPUTE_WH",
        database: str = "SNOWFLAKE_SAMPLE_DATA",
        schema: str = "PUBLIC",
        role: str = None,
    ):
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role
        self._connection = None
        self._lock = threading.Lock()

    def get_connection(self):
        """Snowflake 연결을 가져오거나 새로 생성합니다"""
        with self._lock:
            if self._connection is None or self._connection.is_closed():
                logger.info(f"🔌 Snowflake 연결 생성 중... ({self.account})")
                try:
                    self._connection = snowflake.connector.connect(
                        account=self.account,
                        user=self.user,
                        password=self.password,
                        warehouse=self.warehouse,
                        database=self.database,
                        schema=self.schema,
                        role=self.role,
                    )
                    logger.info(f"✅ Snowflake 연결 성공 ({self.account})")
                except Exception as e:
                    logger.error(f"❌ Snowflake 연결 실패 ({self.account}): {e}")
                    raise

            return self._connection

    def close(self):
        """연결 종료"""
        with self._lock:
            if self._connection and not self._connection.is_closed():
                self._connection.close()
                self._connection = None
                logger.info(f"🔌 Snowflake 연결 종료 ({self.account})")

    def _is_auth_expired_error(self, error: Exception) -> bool:
        """인증 토큰 만료 오류인지 확인합니다."""
        error_code = getattr(error, "errno", None)
        if error_code == 390114:
            return True
        return "Authentication token has expired" in str(error)

    def execute_query(self, query: str, params: Optional[tuple] = None):
        """SQL 쿼리 실행"""

        def _run_query():
            conn = self.get_connection()
            cursor = conn.cursor(DictCursor)
            try:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
            finally:
                cursor.close()

        try:
            return _run_query()
        except ProgrammingError as e:
            if self._is_auth_expired_error(e):
                logger.warning("🔁 Snowflake 인증 토큰 만료 감지 - 재연결 후 재시도")
                self.close()
                return _run_query()
            raise

    def get_current_role(self) -> Optional[str]:
        """현재 세션 역할을 반환합니다."""
        rows = self.execute_query("SELECT CURRENT_ROLE() AS ROLE")
        if rows and isinstance(rows[0], dict):
            return rows[0].get("ROLE")
        return None

    def use_role(self, role: str) -> None:
        """세션 역할을 변경합니다."""
        self.execute_query(f"USE ROLE {role}")


class SnowflakeCortexLLM(CustomLLM):
    """Snowflake Cortex AI Custom LLM Handler - SQL 기반"""

    # 클래스 변수: 싱글톤을 위한 공유 상태
    _shared_connection_managers: Dict[str, SnowflakeConnectionManager] = {}
    _shared_account_configs: Dict[str, Dict[str, str]] = {}
    _shared_model_configs: Dict[str, Dict[str, str]] = {}
    _shared_request_timeout: float = 600.0  # SQL 쿼리 타임아웃
    _shared_lock = threading.Lock()
    _shared_initialized = False
    _shared_suspended_accounts: set = set()
    _shared_cross_region_setting: Optional[str] = None
    _shared_cross_region_applied: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        self._connection_managers = SnowflakeCortexLLM._shared_connection_managers
        self._account_configs = SnowflakeCortexLLM._shared_account_configs
        self._model_configs = SnowflakeCortexLLM._shared_model_configs
        self._lock = SnowflakeCortexLLM._shared_lock
        self._suspended_accounts = SnowflakeCortexLLM._shared_suspended_accounts
        self._cross_region_setting = SnowflakeCortexLLM._shared_cross_region_setting
        self._cross_region_applied = SnowflakeCortexLLM._shared_cross_region_applied

        # 초기화 (한 번만 실행)
        if not SnowflakeCortexLLM._shared_initialized:
            self._initialize_from_env()

    def _initialize_from_env(self):
        """환경변수에서 설정 로드"""
        with self._lock:
            if SnowflakeCortexLLM._shared_initialized:
                return

            logger.info("🔧 Snowflake Cortex Provider 초기화 시작 (SQL 방식)")

            # 환경변수에서 계정 정보 파싱
            accounts_json = os.getenv("SNOWFLAKE_ACCOUNTS")
            if not accounts_json:
                logger.warning("⚠️ SNOWFLAKE_ACCOUNTS 환경변수가 설정되지 않았습니다")
                SnowflakeCortexLLM._shared_initialized = True
                return

            try:
                accounts_data = json.loads(accounts_json)
            except json.JSONDecodeError as e:
                logger.error(f"❌ SNOWFLAKE_ACCOUNTS JSON 파싱 실패: {e}")
                SnowflakeCortexLLM._shared_initialized = True
                return

            # 계정별 연결 관리자 생성
            for account_info in accounts_data:
                account_id = account_info.get("account")
                username = account_info.get("username")
                password = account_info.get("password")
                warehouse = account_info.get("warehouse", "COMPUTE_WH")
                database = account_info.get("database", "SNOWFLAKE_SAMPLE_DATA")
                schema = account_info.get("schema", "PUBLIC")
                role = account_info.get("role")

                if not account_id or not username or not password:
                    logger.warning(f"⚠️ 계정 정보가 불완전합니다: {account_info}")
                    continue

                logger.info(f"   📋 계정 등록: {account_id} (사용자: {username})")

                # 연결 관리자 생성
                conn_manager = SnowflakeConnectionManager(
                    account=account_id,
                    user=username,
                    password=password,
                    warehouse=warehouse,
                    database=database,
                    schema=schema,
                    role=role,
                )

                self._connection_managers[account_id] = conn_manager
                self._account_configs[account_id] = {
                    "account": account_id,
                    "username": username,
                    "warehouse": warehouse,
                    "database": database,
                    "schema": schema,
                    "role": role,
                }

            # 모델 설정 로드
            models_json = os.getenv("SNOWFLAKE_MODELS")
            if models_json:
                try:
                    models_data = json.loads(models_json)
                    for model_info in models_data:
                        model_name = model_info.get("model_name")
                        if model_name:
                            self._model_configs[model_name] = model_info
                except json.JSONDecodeError:
                    logger.warning("⚠️ SNOWFLAKE_MODELS JSON 파싱 실패")

            cross_region_setting = os.getenv("SNOWFLAKE_CROSS_REGION")
            if cross_region_setting:
                self._configure_cross_region_inference(cross_region_setting)

            logger.info(
                f"✅ Snowflake Provider 초기화 완료 (계정: {len(self._connection_managers)}, 모델: {len(self._model_configs)})"
            )
            SnowflakeCortexLLM._shared_initialized = True

    def _normalize_cross_region_setting(self, value: str) -> str:
        """Cross-region 설정 문자열을 표준화합니다."""
        return value.strip().upper()

    def _is_valid_cross_region_setting(self, value: str) -> bool:
        """Cross-region 설정 값 유효성 검사."""
        return bool(re.match(r"^[A-Z_]+(,[A-Z_]+)*$", value))

    def _normalize_role(self, role: str) -> str:
        """역할 문자열을 표준화합니다."""
        return role.strip().upper()

    def _is_valid_role(self, role: str) -> bool:
        """역할 이름 유효성 검사."""
        return bool(re.match(r"^[A-Z0-9_]+$", role))

    def _configure_cross_region_inference(self, raw_setting: str) -> None:
        """계정별 cross-region inference 설정을 적용합니다."""
        normalized = self._normalize_cross_region_setting(raw_setting)
        if not self._is_valid_cross_region_setting(normalized):
            logger.warning(
                "⚠️ 유효하지 않은 cross-region 설정 값: %s (예: ANY_REGION, AWS_US, AWS_US,AWS_EU, DISABLED)",
                raw_setting,
            )
            return

        for account_id, conn_manager in self._connection_managers.items():
            if self._cross_region_applied.get(account_id) == normalized:
                continue

            target_role = "ACCOUNTADMIN"
            configured_role = self._account_configs.get(account_id, {}).get("role")
            if configured_role:
                target_role = self._normalize_role(configured_role)

            if not self._is_valid_role(target_role):
                logger.warning(
                    "⚠️ %s role=%s (유효하지 않은 역할) - cross-region 설정 생략",
                    account_id,
                    target_role,
                )
                continue

            if target_role != "ACCOUNTADMIN":
                logger.warning(
                    "⚠️ %s role=%s (ACCOUNTADMIN 아님) - cross-region 설정 실패 가능",
                    account_id,
                    target_role,
                )

            try:
                original_role = conn_manager.get_current_role()
                if original_role and original_role.upper() != target_role:
                    conn_manager.use_role(target_role)

                query = (
                    "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = '%s'" % normalized
                )
                conn_manager.execute_query(query)
                self._cross_region_applied[account_id] = normalized
                logger.info(
                    "✅ Cross-region inference 설정 적용 (%s): %s",
                    account_id,
                    normalized,
                )

                if original_role and original_role.upper() != target_role:
                    conn_manager.use_role(original_role)
            except ProgrammingError as exc:
                logger.warning(
                    "⚠️ Cross-region 설정 실패 (%s): %s",
                    account_id,
                    exc,
                )
            except DatabaseError as exc:
                logger.warning(
                    "⚠️ Cross-region 설정 DB 오류 (%s): %s",
                    account_id,
                    exc,
                )
            except SnowflakeError as exc:
                logger.warning(
                    "⚠️ Cross-region 설정 오류 (%s): %s",
                    account_id,
                    exc,
                )

    def _get_connection_manager(
        self, account_id: str
    ) -> Optional[SnowflakeConnectionManager]:
        """계정별 연결 관리자 가져오기"""
        return self._connection_managers.get(account_id)

    def _resolve_account_id(self, account_id: str) -> Optional[str]:
        """대소문자 구분 없이 등록된 계정 ID를 찾습니다."""
        if not account_id:
            return None

        account_lookup = account_id.strip().lower()
        for registered_account in self._account_configs.keys():
            if registered_account.lower() == account_lookup:
                return registered_account

        return None

    def get_account_from_model(self, model: str, request_kwargs: dict) -> str:
        """모델명이나 api_base에서 계정 식별"""
        metadata = request_kwargs.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        target_account = str(metadata.get("target_account", "") or "").strip()
        enforce_target_account = bool(metadata.get("enforce_target_account", False))

        # 0. 헬스체크/운영 점검용 강제 계정 지정
        if target_account:
            resolved_target = self._resolve_account_id(target_account)
            if resolved_target:
                if resolved_target.lower() in self._suspended_accounts:
                    if enforce_target_account:
                        raise APIConnectionError(
                            message=(
                                f"Account {resolved_target} is suspended: "
                                "probe requested strict target account"
                            ),
                            model=model,
                            llm_provider="snowflake-cortex",
                        )
                    logger.warning(
                        "⚠️ target_account=%s 는 정지되어 fallback 시도",
                        resolved_target,
                    )
                else:
                    logger.info("🎯 target_account 사용: %s", resolved_target)
                    return resolved_target
            elif enforce_target_account:
                raise InvalidRequestError(
                    message=f"Target account not found: {target_account}",
                    model=model,
                    llm_provider="snowflake-cortex",
                )

        # 1. api_base에서 추출 (가장 명확한 방법)
        api_base = request_kwargs.get("api_base", "")
        if api_base and ".snowflakecomputing.com" in api_base:
            # https://account-name.snowflakecomputing.com → account-name
            import re

            match = re.search(r"https?://([^.]+)\.snowflakecomputing\.com", api_base)
            if match:
                account_candidate = self._resolve_account_id(match.group(1))
                if account_candidate:
                    if account_candidate.lower() in self._suspended_accounts:
                        logger.warning(
                            "⚠️ 계정 %s는 정지되어 제외됩니다 (fallback 시도)",
                            account_candidate,
                        )
                    else:
                        return account_candidate

        # 2. 기존 model@account 형식 (backward compatibility)
        if "@" in model:
            parts = model.split("@", 1)
            if len(parts) == 2:
                account_candidate = self._resolve_account_id(parts[1])
                # 계정 존재 확인
                if account_candidate:
                    if account_candidate.lower() in self._suspended_accounts:
                        logger.warning(
                            "⚠️ 계정 %s는 정지되어 제외됩니다 (fallback 시도)",
                            account_candidate,
                        )
                    else:
                        return account_candidate

        # 3. 기본 계정 사용 (정지되지 않은 계정 중에서)
        available_accounts = [
            acc
            for acc in self._account_configs.keys()
            if acc.lower() not in self._suspended_accounts
        ]

        if not available_accounts:
            raise InvalidRequestError(
                message="All Snowflake accounts are suspended",
                model=model,
                llm_provider="snowflake-cortex",
            )

        default_account = available_accounts[0]
        logger.info(f"📍 기본 계정 사용: {default_account}")
        return default_account

    def _build_cortex_complete_sql(
        self, model: str, messages: List[Dict], **kwargs
    ) -> tuple[str, tuple]:
        """SNOWFLAKE.CORTEX.COMPLETE SQL 쿼리 및 바인딩 파라미터 생성"""
        # 모델명에서 @ 제거
        actual_model = model.split("@", 1)[0] if "@" in model else model

        # 메시지를 JSON 배열로 변환
        messages_json = json.dumps(messages)

        # 옵션 파라미터 처리
        options = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]

        options_json = json.dumps(options) if options else "{}"

        # SQL 쿼리 생성 (바인딩으로 JSON escaping 문제 방지)
        sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            %s,
            PARSE_JSON(%s),
            PARSE_JSON(%s)
        ) as response
        """

        return sql, (actual_model, messages_json, options_json)

    def _parse_cortex_response(self, result: dict, model: str) -> ModelResponse:
        """SNOWFLAKE.CORTEX.COMPLETE 응답 파싱"""
        try:
            # result는 {'RESPONSE': {...}} 형태 (Snowflake는 대문자 사용)
            response_data = result.get("RESPONSE") or result.get("response")

            if isinstance(response_data, str):
                response_data = json.loads(response_data)

            # Snowflake Cortex 응답 구조
            # {"choices": [{"messages": "..."}], "usage": {...}}
            choices_data = response_data.get("choices", [])
            usage_data = response_data.get("usage", {})

            choices = []
            for i, choice in enumerate(choices_data):
                # Snowflake는 "messages" 키 사용
                content = choice.get("messages", choice.get("message", ""))
                if isinstance(content, dict):
                    content = content.get("content", str(content))

                choices.append(
                    Choices(
                        index=i,
                        message=Message(role="assistant", content=content),
                        finish_reason=choice.get("finish_reason", "stop"),
                    )
                )

            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return ModelResponse(
                id=f"cortex-{datetime.now(timezone.utc).timestamp()}",
                object="chat.completion",
                created=int(datetime.now(timezone.utc).timestamp()),
                model=model,
                choices=choices,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"❌ 응답 파싱 실패: {e}")
            logger.error(f"   원본 응답: {result}")
            raise ValueError(f"Failed to parse Cortex response: {e}")

    def _is_account_suspended_error(self, error_msg: str, account_id: str) -> bool:
        """계정 정지 여부를 판별합니다 (일시적 오류와 구분)."""
        msg = (error_msg or "").lower()
        if not msg:
            return False

        account_lower = account_id.lower() if account_id else ""
        account_in_msg = bool(account_lower) and account_lower in msg
        has_account_keyword = "account" in msg

        if (
            "trial" in msg
            and "ended" in msg
            and (account_in_msg or has_account_keyword)
        ):
            return True

        if (
            "billing" in msg
            and ("required" in msg or "past due" in msg or "suspended" in msg)
            and (account_in_msg or has_account_keyword)
        ):
            return True

        if (account_in_msg or has_account_keyword) and any(
            term in msg for term in ("suspended", "suspension", "disabled", "locked")
        ):
            return True

        return False

    def _extract_suspended_account_id(self, error_msg: str) -> Optional[str]:
        """에러 메시지에서 suspended 계정 ID를 추출합니다."""
        if not error_msg:
            return None

        match = re.search(
            r"(?i)\baccount\s+([a-z0-9_-]+)\s+(?:is\s+)?suspended\b", error_msg
        )
        if match:
            return match.group(1)

        return None

    def _mark_account_suspended(self, account_id: str, reason: str) -> None:
        """정지 계정을 라우팅 대상에서 제외합니다."""
        account_key = account_id.lower()
        with self._lock:
            if account_key in self._suspended_accounts:
                return
            self._suspended_accounts.add(account_key)

        logger.error(f"❌ 계정 {account_id} 정지 감지: {reason}")

    def _summarize_suspension_reason(self, error_msg: str) -> str:
        """정지 사유를 간단히 요약합니다."""
        msg = (error_msg or "").strip()
        if len(msg) > 200:
            return f"{msg[:200]}..."
        return msg

    def _handle_snowflake_error(self, e: Exception, model: str, account_id: str):
        """Snowflake 에러를 LiteLLM 예외로 변환"""
        error_msg = str(e)

        # 계정 정지 감지 (일시적 오류와 구분)
        if self._is_account_suspended_error(error_msg, account_id):
            suspended_account = (
                self._extract_suspended_account_id(error_msg) or account_id
            )
            reason = self._summarize_suspension_reason(error_msg)
            self._mark_account_suspended(suspended_account, reason)
            logger.warning(
                "⚠️ 정지 계정 감지: %s (요청 재시도 통해 fallback 처리)",
                suspended_account,
            )
            raise APIConnectionError(
                message=("Account {account} is suspended: {reason}").format(
                    account=suspended_account, reason=reason
                ),
                model=model,
                llm_provider="snowflake-cortex",
            )

        # ProgrammingError (SQL 오류)
        if isinstance(e, ProgrammingError):
            logger.error(f"⚠️ SQL 실행 오류: {error_msg}")
            raise InvalidRequestError(
                message=f"Snowflake SQL error: {error_msg}",
                model=model,
                llm_provider="snowflake-cortex",
            )

        # DatabaseError
        if isinstance(e, DatabaseError):
            logger.error(f"🔴 Snowflake Database Error: {error_msg}")
            raise InternalServerError(
                message=f"Snowflake database error: {error_msg}",
                model=model,
                llm_provider="snowflake-cortex",
            )

        # 일반 Snowflake Error
        if isinstance(e, SnowflakeError):
            logger.error(f"❌ Snowflake Error: {error_msg}")
            raise APIConnectionError(
                message=f"Snowflake error: {error_msg}",
                model=model,
                llm_provider="snowflake-cortex",
            )

        # 기타 예외
        logger.error(f"❌ 예상치 못한 오류: {type(e).__name__} - {error_msg}")
        raise

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """비동기 completion 요청"""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        # 계정 정보 추출
        account_id = self.get_account_from_model(model, kwargs)

        # 연결 관리자 가져오기
        conn_manager = self._get_connection_manager(account_id)
        if not conn_manager:
            raise ValueError(f"연결 관리자를 찾을 수 없습니다: {account_id}")

        # SQL 쿼리 생성 (model과 messages는 이미 추출했으므로 **kwargs에서 제거)
        sql_kwargs = {k: v for k, v in kwargs.items() if k not in ("model", "messages")}
        sql, params = self._build_cortex_complete_sql(model, messages, **sql_kwargs)
        logger.debug(f"🔍 SQL: {sql}")

        try:
            # 비동기 실행을 위해 thread pool 사용
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: conn_manager.execute_query(sql, params)
            )

            # 응답 파싱
            if not result or len(result) == 0:
                raise ValueError("Snowflake returned empty result")

            return self._parse_cortex_response(result[0], model)

        except Exception as e:
            self._handle_snowflake_error(e, model, account_id)

    def completion(self, *args, **kwargs) -> ModelResponse:
        """동기 completion 요청"""
        return asyncio.run(self.acompletion(*args, **kwargs))

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """비동기 스트리밍"""
        # Snowflake Cortex의 SQL stream 지원을 위해서는
        # snowflake.cortex Python 패키지 사용이 필요
        # 현재는 일반 completion 결과를 스트리밍처럼 반환
        response = await self.acompletion(*args, **kwargs)

        content = response.choices[0].message.content if response.choices else ""

        chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": content,
            "tool_use": None,
            "usage": {
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }

        yield chunk

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        """동기 스트리밍"""

        async def async_gen():
            async for chunk in self.astreaming(*args, **kwargs):
                yield chunk

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_iterator = async_gen()
            while True:
                try:
                    chunk = loop.run_until_complete(async_iterator.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


# LiteLLM에서 사용할 인스턴스 생성 함수
def create_snowflake_handler() -> SnowflakeCortexLLM:
    """Snowflake Custom Provider 인스턴스를 생성합니다."""
    return SnowflakeCortexLLM()


# 기본 인스턴스
snowflake_cortex_llm = create_snowflake_handler()
