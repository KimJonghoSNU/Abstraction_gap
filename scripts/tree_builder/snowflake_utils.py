import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

snowflake_connector = importlib.import_module("snowflake.connector")
from snowflake.connector import DictCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError


@dataclass
class SnowflakeAccountConfig:
    account: str
    username: str
    password: str
    warehouse: str
    database: str
    schema: str
    role: Optional[str] = None


def _dequote(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _parse_account_tokens(raw: str) -> List[str]:
    if not raw:
        return []
    values: List[str] = []
    for token in re.split(r"[,\s;]+", str(raw).strip()):
        tok = _dequote(token)
        if not tok:
            continue
        values.append(tok)
    return values


def _read_plain_accounts_from_env_file(env_file: str) -> List[str]:
    if (not env_file) or (not os.path.exists(env_file)):
        return []

    out: List[str] = []
    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue
            if "=" in line:
                continue
            out.extend(_parse_account_tokens(line))
    return out


def _load_snowflake_account_configs(env_file: str) -> List[SnowflakeAccountConfig]:
    # Intent: reproducible account selection requires .env values to override pre-exported shell variables.
    load_dotenv(dotenv_path=env_file, override=True)

    shared_username = _dequote(os.getenv("SNOWFLAKE_USERNAME") or os.getenv("SNOWFLAKE_USER", ""))
    shared_password = _dequote(os.getenv("SNOWFLAKE_PASSWORD", ""))
    shared_warehouse = _dequote(os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"))
    shared_database = _dequote(os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE_SAMPLE_DATA"))
    shared_schema = _dequote(os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"))
    shared_role = _dequote(os.getenv("SNOWFLAKE_ROLE", ""))

    seen: Set[str] = set()
    account_configs: List[SnowflakeAccountConfig] = []

    def add_account(account_value: str, entry: Optional[Dict[str, object]] = None) -> None:
        account_id = _dequote(account_value)
        if not account_id:
            return

        key = account_id.lower()
        if key in seen:
            return

        source = entry or {}
        username = _dequote(str(source.get("username") or source.get("user") or shared_username))
        password = _dequote(str(source.get("password") or shared_password))
        warehouse = _dequote(str(source.get("warehouse") or shared_warehouse or "COMPUTE_WH"))
        database = _dequote(
            str(source.get("database") or shared_database or "SNOWFLAKE_SAMPLE_DATA")
        )
        schema = _dequote(str(source.get("schema") or shared_schema or "PUBLIC"))
        role = _dequote(str(source.get("role") or shared_role))
        if not username or not password:
            raise ValueError(
                "Snowflake username/password is required. "
                "Set SNOWFLAKE_USERNAME and SNOWFLAKE_PASSWORD in environment or .env."
            )

        account_configs.append(
            SnowflakeAccountConfig(
                account=account_id,
                username=username,
                password=password,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role=role or None,
            )
        )
        seen.add(key)

    raw_accounts = _dequote(os.getenv("SNOWFLAKE_ACCOUNTS", ""))
    if raw_accounts:
        parsed = None
        try:
            parsed = json.loads(raw_accounts)
        except Exception:
            parsed = raw_accounts

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    add_account(
                        str(item.get("account") or item.get("account_identifier") or ""),
                        entry=item,
                    )
                elif isinstance(item, str):
                    add_account(item)
        elif isinstance(parsed, str):
            for token in _parse_account_tokens(parsed):
                add_account(token)

    for env_key in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_ACCOUNT_LIST", "SNOWFLAKE_ACCOUNT_IDS"):
        for token in _parse_account_tokens(_dequote(os.getenv(env_key, ""))):
            add_account(token)

    ignored_keys = {
        "SNOWFLAKE_ACCOUNTS",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_ACCOUNT_LIST",
        "SNOWFLAKE_ACCOUNT_IDS",
    }
    for key, value in sorted(os.environ.items()):
        if key in ignored_keys:
            continue
        if not key.startswith("SNOWFLAKE_ACCOUNT_"):
            continue
        for token in _parse_account_tokens(_dequote(value)):
            add_account(token)

    for token in _read_plain_accounts_from_env_file(env_file):
        add_account(token)

    if not account_configs:
        raise ValueError(
            "No Snowflake account found. "
            "Use SNOWFLAKE_ACCOUNTS JSON, SNOWFLAKE_ACCOUNT list, "
            "SNOWFLAKE_ACCOUNT_* variables, or plain account lines in .env."
        )

    return account_configs


class SnowflakeCortexRouter:
    def __init__(
        self,
        *,
        account_configs: List[SnowflakeAccountConfig],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        request_timeout: int,
        max_input_prompt_chars: int,
    ):
        if not account_configs:
            raise ValueError("account_configs must not be empty")

        self._account_configs = list(account_configs)
        self._model = model
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = int(max_tokens)
        self._request_timeout = int(request_timeout)
        self._max_input_prompt_chars = int(max_input_prompt_chars)

        self._connections: Dict[str, object] = {}
        self._suspended_accounts: Set[str] = set()
        self._next_account_idx = 0
        self._cross_region_value = _dequote(
            os.getenv("SNOWFLAKE_CORTEX_CROSS_REGION", "ANY_REGION")
        ).upper()
        self._cross_region_applied: Set[str] = set()

    def close_all(self) -> None:
        for account_id, conn in list(self._connections.items()):
            try:
                if conn is not None and (not conn.is_closed()):
                    conn.close()
            except Exception:
                pass
            finally:
                self._connections.pop(account_id, None)

    def _candidate_accounts(self) -> List[SnowflakeAccountConfig]:
        active = [
            cfg
            for cfg in self._account_configs
            if cfg.account.lower() not in self._suspended_accounts
        ]
        if not active:
            return []

        start = self._next_account_idx % len(active)
        return active[start:] + active[:start]

    def _advance_round_robin(self) -> None:
        active_count = len(
            [
                cfg
                for cfg in self._account_configs
                if cfg.account.lower() not in self._suspended_accounts
            ]
        )
        if active_count <= 0:
            self._next_account_idx = 0
            return
        self._next_account_idx = (self._next_account_idx + 1) % active_count

    def _connection_for(self, cfg: SnowflakeAccountConfig):
        conn = self._connections.get(cfg.account)
        if conn is not None and (not conn.is_closed()):
            return conn

        kwargs = {
            "account": cfg.account,
            "user": cfg.username,
            "password": cfg.password,
            "warehouse": cfg.warehouse,
            "database": cfg.database,
            "schema": cfg.schema,
            "login_timeout": max(1, self._request_timeout),
            "network_timeout": max(1, self._request_timeout),
        }
        if cfg.role:
            kwargs["role"] = cfg.role

        conn = snowflake_connector.connect(**kwargs)
        self._connections[cfg.account] = conn
        self._try_enable_cross_region(conn=conn, cfg=cfg)
        return conn

    def _close_connection(self, account_id: str) -> None:
        conn = self._connections.pop(account_id, None)
        if conn is None:
            return
        try:
            if not conn.is_closed():
                conn.close()
        except Exception:
            pass

    def _build_cortex_complete_sql(
        self,
        *,
        messages: List[Dict[str, str]],
        region: Optional[str] = None,
    ) -> Tuple[str, Tuple[str, str, str]]:
        options = {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }
        if region:
            options["region"] = region

        sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            %s,
            PARSE_JSON(%s),
            PARSE_JSON(%s)
        ) AS response
        """
        actual_model = self._model.split("@", 1)[0] if "@" in self._model else self._model
        params = (
            actual_model,
            json.dumps(messages),
            json.dumps(options, ensure_ascii=False),
        )
        return sql, params

    def _is_auth_expired_error(self, exc: Exception) -> bool:
        errno = getattr(exc, "errno", None)
        if errno == 390114:
            return True
        return "authentication token has expired" in str(exc).lower()

    def _is_region_unavailable_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("unavailable in your region" in msg) or ("cross region inference" in msg)

    def _region_fallback_order(self) -> List[str]:
        raw = _dequote(os.getenv("SNOWFLAKE_CORTEX_FALLBACK_REGIONS", "AWS_US,ANY_REGION"))
        out: List[str] = []
        for token in raw.split(","):
            value = token.strip().upper()
            if not value:
                continue
            if value not in out:
                out.append(value)
        return out

    def _query_complete(self, conn, *, messages: List[Dict[str, str]], region: Optional[str] = None) -> str:
        sql, params = self._build_cortex_complete_sql(messages=messages, region=region)
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            raise RuntimeError("Snowflake returned empty response.")

        row = rows[0]
        response_data = row.get("RESPONSE") or row.get("response")
        if isinstance(response_data, str):
            response_data = json.loads(response_data)

        choices = response_data.get("choices", []) if isinstance(response_data, dict) else []
        if not choices:
            return ""

        content = choices[0].get("messages", choices[0].get("message", ""))
        if isinstance(content, dict):
            content = content.get("content", str(content))
        return str(content or "")

    def _run_with_region_fallback(self, conn, *, messages: List[Dict[str, str]]) -> str:
        try:
            return self._query_complete(conn, messages=messages)
        except ProgrammingError as first_error:
            if not self._is_region_unavailable_error(first_error):
                raise

            last_error = first_error
            for region in self._region_fallback_order():
                try:
                    print(f"[Snowflake] region fallback={region}")
                    return self._query_complete(conn, messages=messages, region=region)
                except ProgrammingError as retry_error:
                    last_error = retry_error
                    if not self._is_region_unavailable_error(retry_error):
                        raise

            raise last_error

    def _try_enable_cross_region(self, *, conn, cfg: SnowflakeAccountConfig) -> None:
        if cfg.account in self._cross_region_applied:
            return
        if not self._cross_region_value:
            return
        if not re.match(r"^[A-Z_]+(,[A-Z_]+)*$", self._cross_region_value):
            return

        cursor = conn.cursor()
        try:
            sql = (
                "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = "
                f"'{self._cross_region_value}'"
            )
            cursor.execute(sql)
            self._cross_region_applied.add(cfg.account)
        except ProgrammingError:
            # Intent: cross-region 설정 실패는 치명적이지 않으므로 요청 실행을 계속합니다.
            pass
        finally:
            cursor.close()

    def _is_account_suspended_error(self, message: str, account_id: str) -> bool:
        msg = str(message or "").lower()
        if not msg:
            return False

        account_lower = str(account_id or "").lower()
        account_in_msg = bool(account_lower) and (account_lower in msg)
        has_account_keyword = "account" in msg

        if (
            "trial" in msg
            and ("ended" in msg or "expired" in msg)
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

    def _is_limit_exceeded_error(self, message: str, account_id: str) -> bool:
        msg = str(message or "").lower()
        if not msg:
            return False

        account_lower = str(account_id or "").lower()
        account_in_msg = bool(account_lower) and (account_lower in msg)

        if any(term in msg for term in ("429", "too many requests", "rate limit", "request limit")):
            return True
        if "quota" in msg and any(term in msg for term in ("exceeded", "reached", "limit")):
            return True
        if "credit" in msg and any(term in msg for term in ("exceeded", "insufficient", "limit")):
            return True
        if ("resource monitor" in msg) and ("suspend" in msg):
            return True
        if (
            any(term in msg for term in ("limit exceeded", "exceeded your limit", "limit has been reached"))
            and (account_in_msg or ("account" in msg) or ("quota" in msg))
        ):
            return True
        return False

    def complete(self, prompt: str) -> str:
        prompt_text = str(prompt or "")
        if self._max_input_prompt_chars > 0 and len(prompt_text) > self._max_input_prompt_chars:
            # Intent: enforce a strict request-size cap before Snowflake API calls to avoid oversized prompt failures.
            prompt_text = prompt_text[: self._max_input_prompt_chars].rstrip()
            if " " in prompt_text:
                prompt_text = prompt_text.rsplit(" ", 1)[0]
        messages = [{"role": "user", "content": prompt_text}]
        candidates = self._candidate_accounts()
        if not candidates:
            raise RuntimeError("All Snowflake accounts are suspended or unavailable.")

        last_error: Optional[Exception] = None
        for cfg in candidates:
            try:
                conn = self._connection_for(cfg)
                text = self._run_with_region_fallback(conn, messages=messages)
                self._advance_round_robin()
                return text
            except ProgrammingError as exc:
                if self._is_auth_expired_error(exc):
                    self._close_connection(cfg.account)
                    try:
                        conn = self._connection_for(cfg)
                        text = self._run_with_region_fallback(conn, messages=messages)
                        self._advance_round_robin()
                        return text
                    except Exception as retry_exc:
                        exc = retry_exc
                last_error = exc
                message = str(exc)
                if self._is_account_suspended_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] account suspended: {cfg.account}")
                    continue
                if self._is_limit_exceeded_error(message, cfg.account):
                    # Intent: 계정별 limit 초과는 전체 빌드 실패가 아니라 다음 account failover 트리거로 처리합니다.
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] account limit exceeded, failover: {cfg.account}")
                    continue
                raise
            except DatabaseError as exc:
                last_error = exc
                message = str(exc)
                # Intent: database-level trial/suspension signals should also trigger account failover, not hard-stop the run.
                if self._is_account_suspended_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] database suspended/trial signal, failover: {cfg.account}")
                    continue
                if self._is_limit_exceeded_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] database error with limit signal, failover: {cfg.account}")
                    continue
                raise
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if self._is_account_suspended_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] runtime suspended/trial signal, failover: {cfg.account}")
                    continue
                if self._is_limit_exceeded_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] runtime error with limit signal, failover: {cfg.account}")
                    continue
                raise

        # Intent: callers should receive a deterministic failure when all configured accounts are exhausted.
        raise RuntimeError("All configured Snowflake accounts failed for this request.") from last_error
