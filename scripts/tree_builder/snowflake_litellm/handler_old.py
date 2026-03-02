#!/usr/bin/env python3
"""
Snowflake Cortex AI Custom Handler for LiteLLM
SQL-based approach using snowflake.connector and SNOWFLAKE.CORTEX.COMPLETE
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import snowflake.connector
from snowflake.connector import DictCursor

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
# LiteLLM 환경에서 logger가 작동하도록 명시적으로 레벨 설정
logger.setLevel(logging.INFO)
# 핸들러가 없으면 콘솔 핸들러 추가
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class SnowflakeConnectionManager:
    """Snowflake JWT 토큰 생성 및 관리 클래스 (완전 자동화)"""

    def __init__(
        self, account_identifier: str, username: str, private_key_path: str, parent=None
    ):
        self.account_identifier = account_identifier
        self.username = username
        self.private_key_path = os.path.join(
            "/app/keys", os.path.basename(private_key_path)
        )
        self.parent = parent
        self._current_token = None
        self._token_expiry = None
        self._lock = threading.Lock()
        self._key_registered = False  # 공개키 등록 여부 추적

        # 토큰 캐시 파일 경로
        self._token_cache_file = f"/tmp/snowflake_jwt_{self.account_identifier}.cache"

        # 공개키 등록 상태 캐시 파일 경로
        self._key_status_cache_file = (
            f"/tmp/snowflake_key_status_{self.account_identifier}.cache"
        )

        # 시작 시 캐시된 토큰 및 키 상태 로드 시도
        self._load_cached_token()
        self._load_key_status()

    def _load_cached_token(self):
        """캐시된 JWT 토큰을 로드합니다"""
        try:
            if os.path.exists(self._token_cache_file):
                with open(self._token_cache_file, "r") as f:
                    cache_data = json.load(f)

                # 캐시된 토큰의 만료 시간 확인
                expiry_str = cache_data.get("expiry")
                if expiry_str:
                    cached_expiry = datetime.fromisoformat(
                        expiry_str.replace("Z", "+00:00")
                    )
                    now = datetime.now(timezone.utc)

                    # 만료 5분 전까지만 사용
                    if now < (cached_expiry - timedelta(minutes=5)):
                        self._current_token = cache_data.get("token")
                        self._token_expiry = cached_expiry
                        logger.info(
                            f"♻️ 캐시된 JWT 토큰 로드 완료 (계정: {self.account_identifier}, 만료까지: {cached_expiry - now})"
                        )
                        return

                # 만료된 캐시 파일 삭제
                os.remove(self._token_cache_file)

        except Exception as e:
            logger.warning(
                f"JWT 토큰 캐시 로드 실패 (계정: {self.account_identifier}): {e}"
            )
            # 캐시 파일이 손상된 경우 제거
            if os.path.exists(self._token_cache_file):
                try:
                    os.remove(self._token_cache_file)
                except:
                    pass

    def _save_token_to_cache(self):
        """JWT 토큰을 캐시 파일에 저장합니다"""
        try:
            if self._current_token and self._token_expiry:
                cache_data = {
                    "token": self._current_token,
                    "expiry": self._token_expiry.isoformat(),
                    "account": self.account_identifier,
                }

                # 임시 파일에 먼저 쓰고 atomic move
                temp_file = self._token_cache_file + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(cache_data, f)

                os.rename(temp_file, self._token_cache_file)
                logger.debug(
                    f"💾 JWT 토큰 캐시 저장 완료 (계정: {self.account_identifier})"
                )

        except Exception as e:
            logger.warning(
                f"JWT 토큰 캐시 저장 실패 (계정: {self.account_identifier}): {e}"
            )

    def _load_key_status(self):
        """캐시된 공개키 등록 상태를 로드합니다"""
        try:
            if os.path.exists(self._key_status_cache_file):
                with open(self._key_status_cache_file, "r") as f:
                    status_data = json.load(f)

                # 24시간 이내 등록 기록만 유효로 간주
                last_registered = status_data.get("last_registered")
                if last_registered:
                    last_registered_time = datetime.fromisoformat(
                        last_registered.replace("Z", "+00:00")
                    )
                    now = datetime.now(timezone.utc)

                    # 24시간 이내면 이미 등록된 것으로 간주
                    if now - last_registered_time < timedelta(hours=24):
                        self._key_registered = status_data.get("registered", False)
                        logger.info(
                            f"♻️ 공개키 등록 상태 캐시 로드: {self.account_identifier} = {self._key_registered}"
                        )
                        return

                # 24시간이 지난 캐시는 제거
                os.remove(self._key_status_cache_file)

        except Exception as e:
            logger.warning(
                f"공개키 상태 캐시 로드 실패 (계정: {self.account_identifier}): {e}"
            )
            # 손상된 캐시 파일 제거
            if os.path.exists(self._key_status_cache_file):
                try:
                    os.remove(self._key_status_cache_file)
                except:
                    pass

    def _save_key_status(self):
        """공개키 등록 상태를 캐시 파일에 저장합니다"""
        try:
            status_data = {
                "registered": self._key_registered,
                "last_registered": datetime.now(timezone.utc).isoformat(),
                "account": self.account_identifier,
            }

            # 임시 파일에 먼저 쓰고 atomic move
            temp_file = self._key_status_cache_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(status_data, f)

            os.rename(temp_file, self._key_status_cache_file)
            logger.debug(
                f"💾 공개키 상태 캐시 저장 완료 (계정: {self.account_identifier})"
            )

        except Exception as e:
            logger.warning(
                f"공개키 상태 캐시 저장 실패 (계정: {self.account_identifier}): {e}"
            )

    def _extract_public_key_one_line(self, private_key_pem: bytes) -> str:
        """Private key에서 공개키를 추출하여 한 줄 형태로 반환 (Snowflake 등록용)"""
        private_key_obj = serialization.load_pem_private_key(
            private_key_pem, password=None
        )
        public_key = private_key_obj.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return (
            public_pem.decode("utf-8")
            .replace("-----BEGIN PUBLIC KEY-----\n", "")
            .replace("\n-----END PUBLIC KEY-----\n", "")
            .replace("\n", "")
        )

    def _register_public_key_to_snowflake(
        self, account: str, user: str, password: str, public_key: str
    ) -> bool:
        """Snowflake 계정에 public key 자동 등록 (snowflake-connector-python 사용)"""
        try:
            logger.info(f"🔧 Snowflake 계정에 public key 등록 중... ({account})")
            logger.debug(f"   사용자: {user}")
            logger.debug(f"   Public Key: {public_key[:50]}...")

            # Snowflake 연결 설정
            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse="COMPUTE_WH",
                database="SNOWFLAKE",
                schema="ACCOUNT_USAGE",
            )

            try:
                cursor = conn.cursor()
                sql_command = f"ALTER USER {user} SET RSA_PUBLIC_KEY='{public_key}'"
                logger.debug(f"📝 실행 SQL: {sql_command[:100]}...")
                cursor.execute(sql_command)
                result = cursor.fetchone()

                logger.info(f"✅ Public key 등록 성공! ({account})")
                logger.debug(f"   결과: {result}")
                logger.info(f"⏳ Key propagation 대기 중 (10초)...")
                time.sleep(10)  # Optimized for parallel processing
                return True

            except Exception as sql_error:
                logger.error(f"❌ SQL 실행 실패: {str(sql_error)}")
                return False
            finally:
                cursor.close()
                conn.close()

        except snowflake.connector.errors.DatabaseError as db_error:
            logger.error(f"❌ Snowflake 연결 실패: {str(db_error)}")
            logger.info(
                f"🔧 수동 등록 SQL: ALTER USER {user} SET RSA_PUBLIC_KEY='{public_key}';"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Public key 등록 중 오류: {str(e)}")
            logger.info(
                f"🔧 수동 등록 SQL: ALTER USER {user} SET RSA_PUBLIC_KEY='{public_key}';"
            )
            return False

    def _enable_cross_region_inference(
        self, account: str, user: str, password: str
    ) -> bool:
        """Snowflake 계정에 cross-region inference 자동 활성화 (ACCOUNT 레벨 설정)"""
        try:
            logger.info(f"🌍 Cross-region inference 활성화 중... ({account})")

            # Snowflake 연결 설정
            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse="COMPUTE_WH",
                database="SNOWFLAKE",
                schema="ACCOUNT_USAGE",
            )

            try:
                cursor = conn.cursor()
                # ACCOUNT 레벨에서 cross-region inference 활성화 (모든 리전 허용)
                # 중요: 이 설정은 USER 레벨이 아닌 ACCOUNT 레벨에서만 가능
                # ANY_REGION을 사용하여 모든 리전에서 inference 허용
                sql_command = (
                    "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION='ANY_REGION'"
                )
                logger.debug(f"📝 실행 SQL: {sql_command}")
                cursor.execute(sql_command)
                result = cursor.fetchone()

                logger.info(f"✅ Cross-region inference 활성화 성공! ({account})")
                logger.debug(f"   결과: {result}")
                logger.info(f"⏳ 설정 반영 대기 중 (2초)...")
                time.sleep(2)  # Optimized for parallel processing
                return True

            except Exception as sql_error:
                logger.error(f"❌ SQL 실행 실패: {str(sql_error)}")
                return False
            finally:
                cursor.close()
                conn.close()

        except snowflake.connector.errors.DatabaseError as db_error:
            logger.error(f"❌ Snowflake 연결 실패: {str(db_error)}")
            logger.info(
                f"🔧 수동 설정 SQL (ACCOUNTADMIN 필요): ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION='ANY_REGION';"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Cross-region inference 활성화 중 오류: {str(e)}")
            logger.info(
                f"🔧 수동 설정 SQL (ACCOUNTADMIN 필요): ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION='ANY_REGION';"
            )
            return False

    def _ensure_cortex_user_role(self, account: str, user: str, password: str) -> bool:
        """CORTEX REST API 권한(기본 역할) 자동 설정을 시도합니다."""
        try:
            logger.info(f"🔐 CORTEX_USER role 설정 확인 중... ({account})")

            conn = snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse="COMPUTE_WH",
                database="SNOWFLAKE",
                schema="ACCOUNT_USAGE",
            )

            try:
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_ROLE()")
                role_row = cursor.fetchone()
                current_role = role_row[0] if role_row else None

                if not current_role:
                    logger.warning(f"⚠️ 현재 역할을 확인할 수 없습니다: {account}")
                    return False

                grant_sql = (
                    "GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE "
                    f"{current_role}"
                )
                cursor.execute(grant_sql)

                default_role_sql = f"ALTER USER {user} SET DEFAULT_ROLE={current_role}"
                cursor.execute(default_role_sql)

                logger.info(
                    f"✅ CORTEX_USER role 설정 완료 ({account}, role={current_role})"
                )
                return True

            except snowflake.connector.errors.ProgrammingError as sql_error:
                logger.error(f"❌ CORTEX_USER role 설정 실패: {str(sql_error)}")
                logger.info(
                    "🔧 수동 설정 SQL: GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE <role>;"
                )
                logger.info(
                    "🔧 수동 설정 SQL: ALTER USER <user> SET DEFAULT_ROLE=<role>;"
                )
                return False
            finally:
                cursor.close()
                conn.close()

        except snowflake.connector.errors.DatabaseError as db_error:
            logger.error(f"❌ Snowflake 연결 실패: {str(db_error)}")
            logger.info(
                "🔧 수동 설정 SQL: GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE <role>;"
            )
            logger.info("🔧 수동 설정 SQL: ALTER USER <user> SET DEFAULT_ROLE=<role>;")
            return False
        except Exception as e:
            logger.error(f"❌ CORTEX_USER role 설정 중 오류: {str(e)}")
            logger.info(
                "🔧 수동 설정 SQL: GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE <role>;"
            )
            logger.info("🔧 수동 설정 SQL: ALTER USER <user> SET DEFAULT_ROLE=<role>;")
            return False

    def _get_password_for_account(self) -> Optional[str]:
        """현재 JWT Manager 계정의 패스워드 가져오기"""
        if not self.parent or not hasattr(self.parent, "_model_configs"):
            return None

        for model_config in self.parent._model_configs.values():
            if model_config.get("account") == self.account_identifier:
                return model_config.get("password")
        return None

    def _load_private_key(self) -> bytes:
        """RSA private key 로드 (키가 없으면 자동 생성 및 등록)"""
        # 1. 기존 키 파일이 있는지 확인
        key_exists = os.path.exists(self.private_key_path)

        if key_exists:
            # 기존 키 로드
            try:
                with open(self.private_key_path, "rb") as key_file:
                    private_key = serialization.load_pem_private_key(
                        key_file.read(), password=None
                    )
                    return private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
            except Exception as e:
                logger.warning(f"키 로드 실패, 새로 생성: {e}")
                key_exists = False

        # 2. 키가 없으면 자동 생성
        if not key_exists:
            print(f"🔑 RSA 키 쌍 자동 생성 중: {self.private_key_path}")

            # 키 쌍 생성
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

            # Private key를 PEM 형식으로 직렬화
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Public key를 PEM 형식으로 직렬화
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            # 키 디렉토리 생성
            os.makedirs(os.path.dirname(self.private_key_path), exist_ok=True)

            # Private key 파일 저장
            with open(self.private_key_path, "wb") as f:
                f.write(private_pem)
            os.chmod(self.private_key_path, 0o600)  # 소유자만 읽기 가능

            # Public key 파일 저장
            public_key_path = self.private_key_path.replace(".pem", ".pub")
            with open(public_key_path, "wb") as f:
                f.write(public_pem)

            logger.info(f"✅ RSA 키 쌍 생성 완료: {self.private_key_path}")

            # 공개키를 한 줄 형태로 변환하여 로깅 (디버깅용)
            public_key_one_line = self._extract_public_key_one_line(private_pem)
            logger.debug(f"� Public Key: {public_key_one_line[:50]}...")
            logger.info(
                f"⚠️ 키가 생성되었습니다. Snowflake 등록은 _try_register_public_key()를 통해 수행됩니다."
            )

            return private_pem

    def _generate_jwt_token(self) -> Tuple[str, datetime]:
        """JWT 토큰을 생성합니다 (올바른 DER 기반 지문 사용)"""
        private_key = self._load_private_key()

        # 현재 시간 (UTC)
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=1)  # 1시간 후 만료

        try:
            # 개인키에서 공개키 추출
            private_key_obj = serialization.load_pem_private_key(
                private_key, password=None
            )
            public_key = private_key_obj.public_key()

            # DER 형식으로 공개키 직렬화 (Snowflake는 DER을 사용)
            public_der = public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            # DER에 대한 SHA256 해시를 Base64로 인코딩 (Snowflake 방식)
            fingerprint_b64 = base64.b64encode(
                hashlib.sha256(public_der).digest()
            ).decode()

            # JWT payload 구성 (계정명과 사용자명은 대문자)
            account_upper = self.account_identifier.upper()
            username_upper = self.username.upper()

            payload = {
                "iss": f"{account_upper}.{username_upper}.SHA256:{fingerprint_b64}",
                "sub": f"{account_upper}.{username_upper}",
                "aud": f"https://{self.account_identifier.lower()}.snowflakecomputing.com",
                "iat": int(now.timestamp()),
                "exp": int(expiry.timestamp()),
            }

            # JWT 토큰 생성
            token = jwt.encode(payload, private_key_obj, algorithm="RS256")
            return token, expiry

        except Exception as e:
            logger.error(f"JWT 토큰 생성 실패 ({self.account_identifier}): {e}")
            raise

    def get_valid_token(self, force_register=False) -> str:
        """유효한 JWT 토큰을 반환합니다 (필요시 갱신, 401 에러 시 자동 재등록)"""
        with self._lock:
            now = datetime.now(timezone.utc)

            # 강제 재등록 요청 시 공개키 재등록
            if force_register and not self._key_registered:
                self._try_register_public_key()

            # 토큰이 없거나 만료 5분 전이면 갱신
            token_was_regenerated = False
            if (
                self._current_token is None
                or self._token_expiry is None
                or now >= (self._token_expiry - timedelta(minutes=5))
            ):
                logger.info(
                    f"🔄 JWT 토큰 새로 생성 중... (계정: {self.account_identifier})"
                )
                if self._current_token is None:
                    logger.info(f"   이유: 토큰이 없음")
                elif self._token_expiry is None:
                    logger.info(f"   이유: 만료 시간이 없음")
                else:
                    logger.info(
                        f"   이유: 만료 임박 (현재: {now}, 만료: {self._token_expiry})"
                    )
                self._current_token, self._token_expiry = self._generate_jwt_token()
                logger.info(f"✅ JWT 토큰 생성 완료 (만료: {self._token_expiry})")
                token_was_regenerated = True

                # 새 토큰을 캐시에 저장
                self._save_token_to_cache()

                # JWT 토큰 propagation을 위한 대기 (force_register 요청이 아닌 경우에만)
                if not force_register:
                    logger.info(f"⏳ JWT 토큰 propagation 대기 중 (1초)...")
                    time.sleep(1)
            else:
                logger.debug(
                    f"♻️ 기존 JWT 토큰 재사용 (계정: {self.account_identifier}, 만료까지: {self._token_expiry - now})"
                )

            return self._current_token

    def _try_register_public_key(self):
        """공개키 자동 등록 시도 (패스워드가 있는 경우)"""
        try:
            password = self._get_password_for_account()
            if not password:
                logger.warning(
                    f"패스워드가 없어 공개키 등록 불가: {self.account_identifier}"
                )
                return False

            # 공개키 추출
            private_key_pem = self._load_private_key()
            public_key_one_line = self._extract_public_key_one_line(private_key_pem)

            # Snowflake에 등록
            success = self._register_public_key_to_snowflake(
                self.account_identifier,
                self.username,
                password,
                public_key_one_line,
            )

            if success:
                self._key_registered = True
                # 키 등록 상태를 캐시에 저장
                self._save_key_status()

                # 기존 토큰 무효화 (새 키로 생성해야 함)
                self._current_token = None
                self._token_expiry = None

                # 캐시 파일도 제거
                try:
                    if os.path.exists(self._token_cache_file):
                        os.remove(self._token_cache_file)
                except:
                    pass

                logger.info(f"공개키 재등록 성공: {self.account_identifier}")
                # Key propagation을 위해 추가 대기
                logger.info(f"Key propagation 대기 중 (15초)...")
                time.sleep(15)  # Optimized for parallel processing
                return True
            else:
                logger.warning(f"공개키 재등록 실패: {self.account_identifier}")
                return False

        except Exception as e:
            logger.error(f"공개키 등록 오류: {e}")
            return False

    def _try_enable_cross_region(self):
        """Cross-region inference 자동 활성화 시도 (패스워드가 있는 경우)"""
        try:
            password = self._get_password_for_account()
            if not password:
                logger.warning(
                    f"패스워드가 없어 cross-region 설정 불가: {self.account_identifier}"
                )
                return False

            # Cross-region inference 활성화
            success = self._enable_cross_region_inference(
                self.account_identifier,
                self.username,
                password,
            )

            if success:
                logger.info(
                    f"Cross-region inference 활성화 성공: {self.account_identifier}"
                )
                return True
            else:
                logger.warning(
                    f"Cross-region inference 활성화 실패: {self.account_identifier}"
                )
                return False

        except Exception as e:
            logger.error(f"Cross-region inference 활성화 오류: {e}")
            return False

    def _try_enable_cortex_user_role(self) -> bool:
        """CORTEX_USER role 자동 설정 시도 (패스워드가 있는 경우)."""
        try:
            password = self._get_password_for_account()
            if not password:
                logger.warning(
                    f"패스워드가 없어 CORTEX_USER role 설정 불가: {self.account_identifier}"
                )
                return False

            success = self._ensure_cortex_user_role(
                self.account_identifier,
                self.username,
                password,
            )

            if success:
                logger.info(f"CORTEX_USER role 설정 성공: {self.account_identifier}")
                return True

            logger.warning(f"CORTEX_USER role 설정 실패: {self.account_identifier}")
            return False

        except Exception as e:
            logger.error(f"CORTEX_USER role 설정 오류: {e}")
            return False


class SnowflakeCortexLLM(CustomLLM):
    """Snowflake Cortex AI Custom LLM Handler with Dynamic Multi-Account Support"""

    # 클래스 변수: 싱글톤을 위한 공유 상태
    _shared_jwt_managers: Dict[str, SnowflakeJWTManager] = {}
    _shared_account_configs: Dict[str, Dict[str, str]] = {}
    _shared_model_configs: Dict[str, Dict[str, str]] = {}
    _shared_request_timeout: float = 120.0
    _shared_lock = threading.Lock()
    _shared_initialized = False
    _shared_suspended_accounts: set = set()  # 무료 체험 종료/정지된 계정 추적

    def __init__(self):
        super().__init__()
        # 인스턴스 변수는 클래스 변수를 참조
        self._jwt_managers = SnowflakeCortexLLM._shared_jwt_managers
        self._account_configs = SnowflakeCortexLLM._shared_account_configs
        self._model_configs = SnowflakeCortexLLM._shared_model_configs
        self._lock = SnowflakeCortexLLM._shared_lock
        self._suspended_accounts = SnowflakeCortexLLM._shared_suspended_accounts

        # 설정이 아직 로드되지 않았으면 한 번만 로드
        with self._lock:
            if not SnowflakeCortexLLM._shared_initialized:
                logger.info("🔧 Snowflake handler 초기화 시작 (최초 1회만)")
                try:
                    self._load_account_configs()
                    self._initialize_jwt_managers()  # JWT 매니저 사전 초기화 및 키 등록
                    SnowflakeCortexLLM._shared_initialized = True
                    logger.info(
                        f"✅ Snowflake handler 초기화 완료 ({len(self._account_configs)}개 계정 로드됨)"
                    )
                except Exception as e:
                    logger.error(f"❌ Snowflake handler 초기화 실패: {e}")
                    raise
            else:
                logger.debug(
                    f"Snowflake handler는 이미 초기화되어 있습니다 (계정 수: {len(self._account_configs)})"
                )

    def _find_config_file(self) -> Optional[str]:
        """litellm_config.yaml 파일을 찾습니다"""
        possible_paths = [
            "/app/litellm_config.yaml",  # 컨테이너 내부 경로
            "/root/Projects/litellm/litellm_config.yaml",  # 호스트 경로
            "/data/interns/jjkim/Projects/litellm/litellm_config.yaml",
            "litellm_config.yaml",
            "../litellm_config.yaml",
            "../../litellm_config.yaml",
            "../../../litellm_config.yaml",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    def _load_account_configs(self):
        """litellm_config.yaml에서 Snowflake 계정 정보를 동적으로 로드하고 자동 완성"""
        config_path = self._find_config_file()
        if not config_path:
            logger.error("litellm_config.yaml 파일을 찾을 수 없습니다.")
            raise FileNotFoundError(
                "litellm_config.yaml 파일을 찾을 수 없습니다. 설정 파일이 필요합니다."
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # request_timeout 설정 읽기
            litellm_settings = config.get("litellm_settings", {})
            request_timeout = litellm_settings.get("request_timeout", 120)
            # 초 단위로 변환 (config는 초 단위일 수 있음)
            SnowflakeCortexLLM._shared_request_timeout = float(request_timeout)
            logger.info(
                f"Request timeout 설정: {SnowflakeCortexLLM._shared_request_timeout}초"
            )

            model_list = config.get("model_list", [])
            account_configs = {}
            model_configs = {}

            for i, model_item in enumerate(model_list):
                model_name = model_item.get("model_name", "")
                litellm_params = model_item.get("litellm_params", {})
                custom_config = litellm_params.get("custom_config", {})
                provider = litellm_params.get("custom_llm_provider", "")

                # Snowflake 모델인지 확인
                if provider in ["snowflake", "snowflake-cortex"] and custom_config:
                    # 계정 식별자를 직접 사용
                    account_identifier = custom_config.get("account", "")
                    if not account_identifier:
                        continue

                    # API 엔드포인트는 litellm_params에서 직접 가져옴 (필수)
                    api_endpoint = litellm_params.get("api_base")
                    if not api_endpoint:
                        logger.warning(f"api_base가 없는 모델 건너뜀: {model_name}")
                        continue

                    # 고유한 키 생성 (모델명 + 계정 식별자)
                    unique_key = f"{model_name}#{account_identifier}"

                    # 모델별 설정 저장 (고유 키 사용)
                    model_configs[unique_key] = custom_config

                    # 계정 설정 저장
                    if account_identifier not in account_configs:
                        # private_key_path는 기본 경로 사용 (자동 생성됨)
                        private_key_path = (
                            f"/app/keys/snowflake_{account_identifier.lower()}_key.pem"
                        )
                        account_configs[account_identifier] = {
                            "account_identifier": account_identifier,
                            "username": custom_config.get("user", ""),
                            "private_key_path": private_key_path,
                            "api_endpoint": api_endpoint,
                        }
                        logger.info(f"Snowflake 계정: {account_identifier}")

            # 자동 생성된 값들은 handler 내부에서만 사용 (파일 저장 안 함)
            # 중요: 딕셔너리를 재할당하지 않고 update()로 내용 복사
            self._account_configs.clear()
            self._account_configs.update(account_configs)
            self._model_configs.clear()
            self._model_configs.update(model_configs)

            if not account_configs:
                logger.warning(
                    "litellm_config.yaml에서 Snowflake 계정 정보를 찾을 수 없습니다."
                )

        except Exception as e:
            logger.error(f"litellm_config.yaml 로드 실패: {e}")
            raise RuntimeError(f"설정 파일 로드 실패: {e}")

    def _create_jwt_manager(self, account_id: str) -> SnowflakeJWTManager:
        """JWT 매니저를 생성합니다 (공통 로직)"""
        if account_id not in self._account_configs:
            raise ValueError(f"계정 정보를 찾을 수 없습니다: {account_id}")

        config = self._account_configs[account_id]
        return SnowflakeJWTManager(
            account_identifier=config["account_identifier"],
            username=config["username"],
            private_key_path=config["private_key_path"],
            parent=self,
        )

    async def _initialize_single_account(
        self, account_id: str, idx: int, total: int
    ) -> tuple[str, bool]:
        """단일 계정의 JWT 매니저를 초기화합니다 (비동기)"""
        try:
            logger.info(f"  [{idx}/{total}] {account_id} 계정 처리 중...")

            # JWT 매니저 생성 (공통 메서드 사용)
            jwt_manager = self._create_jwt_manager(account_id)

            # 공개키가 아직 등록되지 않은 경우에만 등록 시도
            if not jwt_manager._key_registered:
                logger.info(f"    🔐 {account_id}: 공개키 등록 시도 중...")
                # 동기 함수를 executor에서 실행
                await asyncio.to_thread(jwt_manager._try_register_public_key)
            else:
                logger.info(f"    ♻️ {account_id}: 이미 등록된 공개키 사용")

            logger.info(f"    🧩 {account_id}: CORTEX_USER role 설정 시도 중...")
            await asyncio.to_thread(jwt_manager._try_enable_cortex_user_role)

            try:
                # JWT 토큰 생성 시도 (캐시에서 로드하거나 새로 생성)
                token = await asyncio.to_thread(jwt_manager.get_valid_token)
                logger.info(f"    ✅ {account_id}: JWT 토큰 생성 성공")
                jwt_manager._key_registered = True

                # Cross-region inference 활성화 (JWT Manager가 자체적으로 처리)
                logger.info(
                    f"    🌍 {account_id}: Cross-region inference 활성화 시도 중..."
                )
                await asyncio.to_thread(jwt_manager._try_enable_cross_region)

            except Exception as token_error:
                logger.warning(f"    ⚠️ {account_id}: 초기화 실패 - {token_error}")
                logger.warning(f"       첫 API 요청 시 자동 재시도 예정")
                # 실패해도 JWT 매니저는 저장 (나중에 재시도 가능)

            # JWT 매니저를 딕셔너리에 저장
            self._jwt_managers[account_id] = jwt_manager
            return (account_id, True)

        except Exception as e:
            logger.error(f"    ❌ {account_id}: JWT 매니저 초기화 실패 - {e}")
            return (account_id, False)

    def _initialize_jwt_managers(self):
        """모든 계정에 대해 JWT 매니저를 병렬로 초기화하고 키를 등록합니다"""
        logger.info("🔑 JWT 매니저 사전 초기화 및 키 등록 시작 (병렬 처리)...")

        account_count = len(self._account_configs)

        # 비동기 작업 실행
        async def run_parallel_init():
            tasks = []
            for idx, account_id in enumerate(self._account_configs.keys(), 1):
                task = self._initialize_single_account(account_id, idx, account_count)
                tasks.append(task)

            # 모든 계정을 병렬로 처리 (최대 120초 타임아웃)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=120.0
                )
                return results
            except asyncio.TimeoutError:
                logger.error("⏱️ JWT 매니저 초기화 타임아웃 (120초 초과)")
                return []

        # 이벤트 루프 실행
        try:
            # 기존 이벤트 루프가 있으면 사용, 없으면 새로 생성
            try:
                loop = asyncio.get_running_loop()
                # 이미 실행 중인 루프에서는 새 태스크로 실행
                future = asyncio.ensure_future(run_parallel_init())
                results = loop.run_until_complete(future)
            except RuntimeError:
                # 실행 중인 루프가 없으면 새로 생성
                results = asyncio.run(run_parallel_init())

            # 결과 집계
            success_count = sum(1 for r in results if isinstance(r, tuple) and r[1])
            logger.info(
                f"✅ JWT 매니저 초기화 완료 ({success_count}/{account_count}개 성공)"
            )
        except Exception as e:
            logger.error(f"❌ JWT 매니저 초기화 중 오류 발생: {e}")
            logger.info("   일부 계정은 첫 API 요청 시 자동으로 초기화됩니다")

    def _extract_account_alias(self, model_name: str) -> str:
        """모델명에서 계정 별칭을 추출합니다 (현재는 사용되지 않음)"""
        # 이 메서드는 더 이상 사용되지 않습니다.
        # 계정 선택은 get_account_from_model에서 처리됩니다.
        return "main"

    def _get_jwt_manager(self, account_id: str) -> SnowflakeJWTManager:
        """계정별 JWT 매니저를 가져오거나 생성"""
        with self._lock:
            if account_id not in self._jwt_managers:
                # 공통 생성 메서드 사용
                self._jwt_managers[account_id] = self._create_jwt_manager(account_id)

            return self._jwt_managers[account_id]

    def get_account_from_model(
        self, model: str, request_kwargs: Optional[Dict] = None
    ) -> str:
        """
        LiteLLM이 선택한 deployment의 정보에서 계정 정보를 추출합니다.
        정지된 계정은 자동으로 제외됩니다.
        """
        if request_kwargs is None:
            request_kwargs = {}

        # 0. metadata에서 target_account 확인 (헬스체크 등에서 특정 계정 지정 시)
        metadata = request_kwargs.get("metadata", {})
        if metadata and "target_account" in metadata:
            target_account = metadata["target_account"]
            # 계정이 존재하고 정지되지 않았는지 확인
            if target_account in self._account_configs:
                if target_account not in self._suspended_accounts:
                    logger.debug(
                        f"🎯 metadata에서 지정된 계정 사용: {target_account} (모델: {model})"
                    )
                    return target_account
                else:
                    logger.warning(
                        f"⚠️ metadata에서 지정된 계정 {target_account}는 정지되어 있습니다"
                    )

        # api_base 찾기 - 여러 위치에서 시도
        api_base = None

        # 1. metadata에서 api_base 확인 (헬스체크 등에서 특정 엔드포인트 지정 시)
        if metadata and "api_base" in metadata:
            api_base = metadata["api_base"]

        # 2. litellm_params에서 찾기
        if not api_base:
            litellm_params = request_kwargs.get("litellm_params", {})
            api_base = litellm_params.get("api_base", "")

        # 3. kwargs 최상위에서 찾기
        if not api_base:
            api_base = request_kwargs.get("api_base", "")

        if api_base and isinstance(api_base, str):
            import re

            match = re.match(r"https://([^.]+)\.snowflakecomputing\.com", api_base)
            if match:
                account_from_url = match.group(1)
                account_upper = account_from_url.upper()

                # _account_configs에 저장된 키와 정확히 일치하는지 확인
                if account_upper in self._account_configs:
                    # 정지된 계정인지 확인
                    if account_upper in self._suspended_accounts:
                        logger.warning(
                            f"⚠️ 계정 {account_upper}는 정지되어 사용할 수 없습니다 (무료 체험 종료 또는 청구 정보 필요)"
                        )
                        raise InvalidRequestError(
                            message=f"Account {account_upper} is suspended (free trial ended or billing required)",
                            model=model,
                            llm_provider="snowflake-cortex",
                        )
                    return account_upper

                # 대소문자 무관 매칭
                for account_id in self._account_configs.keys():
                    if account_id.upper() == account_upper:
                        # 정지된 계정인지 확인
                        if account_id in self._suspended_accounts:
                            logger.warning(
                                f"⚠️ 계정 {account_id}는 정지되어 사용할 수 없습니다 (무료 체험 종료 또는 청구 정보 필요)"
                            )
                            raise InvalidRequestError(
                                message=f"Account {account_id} is suspended (free trial ended or billing required)",
                                model=model,
                                llm_provider="snowflake-cortex",
                            )
                        return account_id

        # 계정을 찾을 수 없으면 기본 계정 사용 (정지되지 않은 계정 중에서)
        if self._account_configs:
            # 정지되지 않은 계정 찾기
            available_accounts = [
                acc
                for acc in self._account_configs.keys()
                if acc not in self._suspended_accounts
            ]

            if not available_accounts:
                raise InvalidRequestError(
                    message="All Snowflake accounts are suspended (free trial ended or billing required)",
                    model=model,
                    llm_provider="snowflake-cortex",
                )

            default_account = available_accounts[0]
            logger.warning(
                f"계정 식별 실패, 사용 가능한 기본 계정 사용: {default_account}"
            )
            logger.warning(f"요청된 api_base: {api_base}")
            logger.warning(f"request_kwargs keys: {list(request_kwargs.keys())}")
            return default_account
        else:
            raise ValueError("사용 가능한 Snowflake 계정이 없습니다")

    def get_jwt_token(self, account_id: str) -> str:
        """계정별 JWT 토큰 가져오기 (자동 갱신)"""
        jwt_manager = self._get_jwt_manager(account_id)
        return jwt_manager.get_valid_token()

    def get_api_endpoint(self, account_id: str) -> str:
        """계정별 API 엔드포인트 가져오기"""
        if account_id not in self._account_configs:
            raise ValueError(f"계정 정보를 찾을 수 없습니다: {account_id}")
        return self._account_configs[account_id]["api_endpoint"]

    def prepare_headers(self, jwt_token: str) -> Dict[str, str]:
        """Snowflake API 요청 헤더 준비"""
        return {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "User-Agent": "LiteLLM-SnowflakeCortex/1.0",
        }

    def transform_request(
        self, model_name: str, messages: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        """LiteLLM 요청을 Snowflake Cortex API 형식으로 변환"""
        # 헬스체크 alias(model@account)는 Snowflake로 전달하지 않음
        # litellm_params.model에 설정된 base model만 전송
        actual_model = model_name.split("@", 1)[0] if "@" in model_name else model_name

        # Snowflake Cortex API 요청 형식
        request_data = {"model": actual_model, "messages": messages}

        # 선택적 파라미터 처리
        if "temperature" in kwargs:
            request_data["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            request_data["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            request_data["top_p"] = kwargs["top_p"]

        return request_data

    def _parse_sse_response(self, response_text: str, model: str) -> Dict:
        """SSE 형태의 응답을 파싱하여 완성된 응답 데이터로 변환"""
        lines = response_text.strip().split("\n")
        full_content = ""
        last_usage = None
        response_id = None
        model_name = None

        for line in lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # "data: " 제거

                    if not response_id:
                        response_id = data.get("id")
                        model_name = data.get("model")

                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "delta" in choice:
                            delta = choice["delta"]
                            if "content" in delta:
                                full_content += delta["content"]

                    if "usage" in data:
                        last_usage = data["usage"]

                except json.JSONDecodeError:
                    continue

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model_name or model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": last_usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _handle_api_response(self, response_text: str, model: str) -> Dict:
        """API 응답을 파싱 (SSE 또는 JSON)"""
        if response_text.startswith("data: "):
            return self._parse_sse_response(response_text, model)
        else:
            # 일반 JSON 응답
            return json.loads(response_text)

    async def _retry_with_key_registration(
        self,
        client: httpx.AsyncClient,
        account_id: str,
        api_endpoint: str,
        request_data: Dict,
        model: str,
    ) -> ModelResponse:
        """JWT 인증 실패 시 공개키 재등록 후 재시도"""
        jwt_manager = self._jwt_managers.get(account_id)

        if not jwt_manager:
            raise Exception("JWT manager를 찾을 수 없습니다")

        # 첫 번째 401 에러: 공개키 등록 시도
        if not jwt_manager._key_registered:
            logger.warning(f"JWT 인증 실패, 공개키 등록 시도: {account_id}")
            jwt_manager._try_register_public_key()

            if not jwt_manager._key_registered:
                raise Exception("공개키 등록 실패")
        else:
            # 이미 등록된 경우: 토큰 재생성만 시도
            logger.warning(f"JWT 인증 실패 (이미 등록됨), 토큰 재생성: {account_id}")
            jwt_manager._current_token = None
            jwt_manager._token_expiry = None

        # 새 토큰으로 재시도
        jwt_token = jwt_manager.get_valid_token()
        headers = self.prepare_headers(jwt_token)

        response = await client.post(api_endpoint, headers=headers, json=request_data)
        response.raise_for_status()

        # 응답 파싱
        response_data = self._handle_api_response(response.text, model)
        logger.info(f"JWT 재등록/재생성 후 성공: {account_id}")

        return self.transform_response(response_data, model)

    def transform_response(self, response_data: Dict, model: str) -> ModelResponse:
        """Snowflake Cortex 응답을 LiteLLM ModelResponse 형식으로 변환"""
        try:
            # Snowflake Cortex API 응답 구조 처리
            choices = []

            if "choices" in response_data:
                for i, choice in enumerate(response_data["choices"]):
                    message_content = choice.get("message", {}).get("content", "")
                    choices.append(
                        Choices(
                            index=i,
                            message=Message(role="assistant", content=message_content),
                            finish_reason=choice.get("finish_reason", "stop"),
                        )
                    )
            else:
                # 단일 응답 처리 (message나 content 필드)
                content = response_data.get("message", response_data.get("content", ""))
                choices.append(
                    Choices(
                        index=0,
                        message=Message(role="assistant", content=content),
                        finish_reason="stop",
                    )
                )

            # Usage 정보 처리
            usage_data = response_data.get("usage", {})
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
            logger.error(f"응답 변환 오류: {e}")
            raise ValueError(f"Failed to transform response: {e}")

    def _handle_http_status_error(
        self,
        e: httpx.HTTPStatusError,
        model: str,
        api_endpoint: str,
        client: httpx.AsyncClient,
        account_id: str,
        request_data: Dict,
    ) -> None:
        """HTTP 상태 코드 에러를 적절한 LiteLLM 예외로 변환"""
        status_code = e.response.status_code

        # 상세 디버그 로그 (요청/응답 전체 정보)
        try:
            request_headers = dict(e.request.headers)
            if "authorization" in request_headers:
                request_headers["authorization"] = "Bearer ***REDACTED***"

            response_headers = dict(e.response.headers or {})
            debug_log = {
                "debug": "snowflake_http_error_detail",
                "http_status": status_code,
                "account": account_id,
                "endpoint": api_endpoint,
                "model": model,
                "request": {
                    "method": e.request.method,
                    "url": str(e.request.url),
                    "headers": request_headers,
                    "payload": request_data,
                },
                "response": {
                    "headers": response_headers,
                    "body": (e.response.text or "")[:4000],
                },
            }
            logger.info(
                "[DEBUG] Snowflake HTTP error detail: %s",
                json.dumps(debug_log, ensure_ascii=False),
            )
        except Exception as debug_log_error:
            logger.info(
                "[DEBUG] Snowflake HTTP error detail logging failed: %s",
                debug_log_error,
            )

        # 413 에러 (Payload Too Large) - 원본 에러를 그대로 throw하여 Router가 처리
        if status_code == 413:
            logger.warning(f"Snowflake API 요청 크기 초과 (413): {api_endpoint}")
            logger.warning("다음 deployment로 자동 전환 시도...")
            raise

        # 에러 메시지 파싱
        error_msg = f"Snowflake API 오류 ({status_code})"
        error_text = e.response.text
        try:
            error_data = e.response.json()
            error_msg += f" - {error_data.get('message', error_text)}"
        except:
            error_msg += f" - {error_text}"

        # 429 에러 (Rate Limit) - 일시적인 제한
        if status_code == 429:
            logger.warning(f"🚦 Snowflake Rate Limit (429) - 일시적인 요청 제한")
            logger.warning(f"   계정: {account_id}")
            logger.warning(f"   엔드포인트: {api_endpoint}")
            logger.warning(f"   모델: {model}")
            logger.warning(f"   에러 상세: {error_msg}")

            # Retry-After 헤더 확인
            retry_after = e.response.headers.get("Retry-After")
            if retry_after:
                logger.warning(
                    f"   ⏰ Retry-After: {retry_after}초 (이 시간 후 자동 복구)"
                )
            else:
                logger.warning(f"   ⏰ 잠시 후 자동 복구됨 (일시적 제한)")

            logger.info(
                "💡 이것은 일시적인 rate limit입니다. LiteLLM Router가 자동으로 재시도하거나 다른 deployment로 전환합니다."
            )
            raise RateLimitError(
                message=f"Temporary rate limit exceeded for account {account_id}. Will retry automatically.",
                model=model,
                llm_provider="snowflake-cortex",
            )

        # 500번대 에러 (Internal Server Error)
        if 500 <= status_code < 600:
            logger.error(f"🔴 Snowflake Internal Server Error ({status_code})")
            logger.error(f"   계정: {account_id}")
            logger.error(f"   엔드포인트: {api_endpoint}")
            logger.error(f"   모델: {model}")

            # 응답 본문 상세 로그
            try:
                error_json = e.response.json()
                logger.error(f"   응답 JSON: {error_json}")
            except:
                logger.error(f"   응답 텍스트: {error_text}")

            # 요청 헤더 로그 (민감 정보 제외)
            headers = dict(e.request.headers)
            if "authorization" in headers:
                headers["authorization"] = "Bearer ***REDACTED***"
            logger.error(f"   요청 헤더: {headers}")

            # 요청 본문 로그
            logger.error(f"   요청 본문: {request_data}")

            raise InternalServerError(
                message=error_msg,
                model=model,
                llm_provider="snowflake-cortex",
                response=e.response,
            )

        # 400번대 에러 (Bad Request)
        if 400 <= status_code < 500:
            logger.error(f"⚠️  Snowflake Bad Request ({status_code})")
            logger.error(f"   계정: {account_id}")
            logger.error(f"   엔드포인트: {api_endpoint}")
            logger.error(f"   모델: {model}")
            logger.error(f"   에러 상세: {error_msg}")

            # 응답 본문 상세 로그
            try:
                error_json = e.response.json()
                logger.error(f"   응답 JSON: {error_json}")
            except:
                logger.error(f"   응답 텍스트: {error_text}")

            raise InvalidRequestError(
                message=error_msg,
                model=model,
                llm_provider="snowflake-cortex",
            )

        # 기타 HTTP 에러
        logger.error(f"❌ Snowflake HTTP Error ({status_code})")
        logger.error(f"   계정: {account_id}")
        logger.error(f"   엔드포인트: {api_endpoint}")
        logger.error(f"   에러 상세: {error_msg}")

        raise InvalidRequestError(
            message=error_msg,
            model=model,
            llm_provider="snowflake-cortex",
        )

    async def _handle_401_error(
        self,
        e: httpx.HTTPStatusError,
        client: httpx.AsyncClient,
        account_id: str,
        api_endpoint: str,
        request_data: Dict,
        model: str,
    ) -> Optional[ModelResponse]:
        """401 에러 처리 - JWT 토큰 재등록 및 재시도, 계정 정지 감지"""
        error_text = e.response.text

        # Snowflake trial 종료/청구 필요로 인한 계정 정지는 강한 신호가 있을 때만 마킹합니다.
        # ("suspended" 같은 단어는 웨어하우스/리소스 모니터 등 다른 맥락에서도 등장할 수 있어 오탐 가능)
        normalized_text = error_text or ""
        try:
            error_json = e.response.json()
            if isinstance(error_json, dict):
                normalized_text = (
                    (error_json.get("message") or "")
                    + "\n"
                    + (error_json.get("error") or "")
                    + "\n"
                    + (error_json.get("detail") or "")
                    + "\n"
                    + (error_text or "")
                )
        except Exception:
            pass

        lower_text = normalized_text.lower()
        is_trial_ended = "free trial has ended" in lower_text
        is_trial_expired = "trial has expired" in lower_text
        asks_for_billing = ("add billing information" in lower_text) or (
            "billing" in lower_text
        )

        # 계정 정지로 마킹하는 조건은 보수적으로: trial 종료/만료 + billing 요구
        if is_trial_ended or (is_trial_expired and asks_for_billing):
            # 계정을 정지 목록에 추가
            with self._lock:
                self._suspended_accounts.add(account_id)

            logger.error(
                f"❌❌❌ 계정 {account_id}가 영구적으로 정지되었습니다 (Rate Limit이 아님!)"
            )
            logger.error(f"   원인: 무료 체험 종료 또는 청구 정보 필요")
            logger.error(f"   모델: {model}")
            logger.error(f"   에러 메시지: {error_text}")
            logger.error(
                f"   ⚠️  이것은 일시적인 rate limit이 아닌 영구적인 계정 정지입니다!"
            )
            logger.warning(
                f"   📍 이 계정은 자동 복구되지 않으며, LiteLLM Router가 다른 deployment로 전환합니다."
            )
            logger.warning(
                f"   💡 해결 방법 1: lom validate-models --cleanup (자동 제거)"
            )
            logger.warning(f"   💡 해결 방법 2: Snowflake 콘솔에서 청구 정보 추가")

            # InvalidRequestError로 변환 (클라이언트 측 오류 - 400번대)
            # 이는 일시적이지 않은 오류이므로 재시도하지 않도록 합니다
            # LiteLLM Router는 이를 받아서 다른 deployment로 즉시 전환합니다
            raise InvalidRequestError(
                message=f"Snowflake account {account_id} is suspended (free trial ended or requires billing information). Please remove this account from configuration or add billing details in Snowflake console.",
                model=model,
                llm_provider="snowflake-cortex",
            )

        # JWT 토큰 관련 에러가 아니면 None 반환
        if not (
            "JWT token is invalid" in error_text or "invalid" in error_text.lower()
        ):
            return None

        try:
            return await self._retry_with_key_registration(
                client, account_id, api_endpoint, request_data, model
            )
        except httpx.HTTPStatusError as retry_http_error:
            # 재시도 중 429 에러 발생 시 RateLimitError로 변환
            if retry_http_error.response.status_code == 429:
                error_msg = f"Snowflake API 요청 제한 초과 (429)"
                try:
                    error_data = retry_http_error.response.json()
                    error_msg += f" - {error_data.get('message', retry_http_error.response.text)}"
                except:
                    error_msg += f" - {retry_http_error.response.text}"
                logger.error(error_msg)
                raise RateLimitError(
                    message=error_msg,
                    model=model,
                    llm_provider="snowflake-cortex",
                )
            else:
                logger.error(
                    f"재시도 중 HTTP 에러: {retry_http_error.response.status_code}"
                )
                raise
        except Exception as retry_error:
            logger.error(f"재시도 실패: {retry_error}")
            raise

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        """비동기 completion 요청"""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])

        # 계정 정보 추출
        account_id = self.get_account_from_model(model, kwargs)
        api_endpoint = kwargs.get("api_base") or self.get_api_endpoint(account_id)

        # JWT 토큰 및 헤더 준비
        jwt_token = self.get_jwt_token(account_id)
        headers = self.prepare_headers(jwt_token)

        # 요청 데이터 구성
        request_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["model", "messages"]
        }
        request_data = self.transform_request(model, messages, **request_kwargs)

        async with httpx.AsyncClient(
            timeout=SnowflakeCortexLLM._shared_request_timeout
        ) as client:
            try:
                # API 요청
                response = await client.post(
                    api_endpoint, headers=headers, json=request_data
                )
                response.raise_for_status()

                # 응답 파싱 및 변환
                response_data = self._handle_api_response(response.text, model)
                return self.transform_response(response_data, model)

            except httpx.HTTPStatusError as e:
                # 401 에러 처리 (JWT 토큰 재등록 시도)
                if e.response.status_code == 401:
                    result = await self._handle_401_error(
                        e, client, account_id, api_endpoint, request_data, model
                    )
                    if result:
                        return result

                # 기타 HTTP 상태 코드 에러 처리
                self._handle_http_status_error(
                    e, model, api_endpoint, client, account_id, request_data
                )

            except httpx.ReadTimeout:
                logger.error(f"⏱️  Snowflake API Read Timeout")
                logger.error(f"   계정: {account_id}")
                logger.error(f"   엔드포인트: {api_endpoint}")
                logger.error(f"   모델: {model}")
                logger.error(
                    f"   타임아웃 설정: {SnowflakeCortexLLM._shared_request_timeout}초"
                )
                logger.error(f"   요청 데이터 크기: {len(str(request_data))} bytes")
                logger.info(
                    "[DEBUG] Snowflake timeout request detail: %s",
                    json.dumps(
                        {
                            "debug": "snowflake_timeout_request",
                            "account": account_id,
                            "endpoint": api_endpoint,
                            "model": model,
                            "request": {
                                "method": "POST",
                                "headers": {
                                    **{
                                        k: (
                                            "Bearer ***REDACTED***"
                                            if k.lower() == "authorization"
                                            else v
                                        )
                                        for k, v in headers.items()
                                    }
                                },
                                "payload": request_data,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )

                error_msg = f"Snowflake API 타임아웃 ({SnowflakeCortexLLM._shared_request_timeout}초 초과): {api_endpoint}"
                raise Timeout(
                    message=error_msg,
                    model=model,
                    llm_provider="snowflake-cortex",
                )

            except httpx.ConnectTimeout:
                logger.error(f"⏱️  Snowflake API Connect Timeout")
                logger.error(f"   계정: {account_id}")
                logger.error(f"   엔드포인트: {api_endpoint}")
                logger.error(f"   모델: {model}")
                logger.error(
                    f"   타임아웃 설정: {SnowflakeCortexLLM._shared_request_timeout}초"
                )
                logger.info(
                    "[DEBUG] Snowflake connect timeout request detail: %s",
                    json.dumps(
                        {
                            "debug": "snowflake_connect_timeout_request",
                            "account": account_id,
                            "endpoint": api_endpoint,
                            "model": model,
                            "request": {
                                "method": "POST",
                                "headers": {
                                    **{
                                        k: (
                                            "Bearer ***REDACTED***"
                                            if k.lower() == "authorization"
                                            else v
                                        )
                                        for k, v in headers.items()
                                    }
                                },
                                "payload": request_data,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )

                error_msg = f"Snowflake API 연결 타임아웃: {api_endpoint}"
                raise Timeout(
                    message=error_msg,
                    model=model,
                    llm_provider="snowflake-cortex",
                )

            except httpx.ConnectError as e:
                logger.error(f"🔌 Snowflake API Connection Error")
                logger.error(f"   계정: {account_id}")
                logger.error(f"   엔드포인트: {api_endpoint}")
                logger.error(f"   모델: {model}")
                logger.error(f"   에러 타입: {type(e).__name__}")
                logger.error(f"   에러 상세: {str(e)}")
                logger.info(
                    "[DEBUG] Snowflake connection error request detail: %s",
                    json.dumps(
                        {
                            "debug": "snowflake_connection_error_request",
                            "account": account_id,
                            "endpoint": api_endpoint,
                            "model": model,
                            "request": {
                                "method": "POST",
                                "headers": {
                                    **{
                                        k: (
                                            "Bearer ***REDACTED***"
                                            if k.lower() == "authorization"
                                            else v
                                        )
                                        for k, v in headers.items()
                                    }
                                },
                                "payload": request_data,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )

                error_msg = f"Snowflake API 연결 실패: {api_endpoint} - {str(e)}"
                raise APIConnectionError(
                    message=error_msg,
                    model=model,
                    llm_provider="snowflake-cortex",
                )

            except httpx.ReadError as e:
                error_msg = f"Snowflake API 읽기 오류: {api_endpoint} - {str(e)}"
                logger.error(error_msg)
                logger.info(
                    "[DEBUG] Snowflake read error request detail: %s",
                    json.dumps(
                        {
                            "debug": "snowflake_read_error_request",
                            "account": account_id,
                            "endpoint": api_endpoint,
                            "model": model,
                            "request": {
                                "method": "POST",
                                "headers": {
                                    **{
                                        k: (
                                            "Bearer ***REDACTED***"
                                            if k.lower() == "authorization"
                                            else v
                                        )
                                        for k, v in headers.items()
                                    }
                                },
                                "payload": request_data,
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
                raise APIConnectionError(
                    message=error_msg,
                    model=model,
                    llm_provider="snowflake-cortex",
                )

            except Exception as e:
                # 이미 LiteLLM 예외인 경우 그대로 전파
                if isinstance(
                    e,
                    (
                        RateLimitError,
                        InternalServerError,
                        InvalidRequestError,
                        Timeout,
                        APIConnectionError,
                        AuthenticationError,
                    ),
                ):
                    raise

                # 기타 예외 로깅
                error_type = type(e).__name__
                error_str = str(e) if str(e) else repr(e)
                logger.error(f"Snowflake API 요청 실패 ({error_type}): {error_str}")
                raise

    def completion(self, *args, **kwargs) -> ModelResponse:
        """동기 completion 요청"""
        return asyncio.run(self.acompletion(*args, **kwargs))

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        """비동기 스트리밍 (현재는 일반 completion 결과를 스트리밍 형태로 변환)"""
        response = await self.acompletion(*args, **kwargs)

        # 응답을 스트리밍 청크로 변환
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

        # 비동기 제너레이터를 동기적으로 실행
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


# 기본 인스턴스 (하위 호환성용)
snowflake_cortex_llm = create_snowflake_handler()
