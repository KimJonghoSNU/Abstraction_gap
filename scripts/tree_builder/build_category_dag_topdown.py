import argparse
import datetime
import importlib
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from json_repair import repair_json

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != _THIS_SCRIPT_DIR
]

snowflake_connector = importlib.import_module("snowflake.connector")
from snowflake.connector import DictCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError


PROMPT_TEMPLATE = (
    "You are an expert in information retrieval and keyword generation.\n\n"
    "Your task is to analyze ONE informational passage and generate category-based "
    "hierarchical retrieval keywords, strictly following the 5-level rubric.\n\n"
    "Important constraints:\n"
    "- Category-first hierarchy: classify by SUPPORT ROLE / TYPE, not by topic.\n"
    "- Avoid content-topic labels like 'Evolution', 'Darwin', 'Neural Network'.\n"
    "- Prefer reusable role/category labels like 'Theory/Principle', "
    "'Experimental Protocol', 'Empirical Entity/Evidence'.\n"
    "- Output must be actionable search phrases.\n"
    "- If the passage reasonably supports two role hierarchies, provide one alternate path.\n\n"
    "Keyword Generation Rules (5 Levels):\n"
    "Level 1: 1-2 words, broadest role/domain label.\n"
    "Level 2: 3-4 words, general role sub-domain.\n"
    "Level 3: 4-6 words, key support concepts/themes.\n"
    "Level 4: 7-10 words, concise passage support summary.\n"
    "Level 5: 11-20 words, most specific support summary.\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"passage_id\": \"{passage_id}\",\n"
    "  \"hierarchical_keywords\": [\"L1\", \"L2\", \"L3\", \"L4\", \"L5\"],\n"
    "  \"alternate_hierarchical_keywords\": [[\"L1\", \"L2\", \"L3\", \"L4\", \"L5\"]]\n"
    "}}\n\n"
    "Input Passage ID:\n"
    "{passage_id}\n\n"
    "Website Title:\n"
    "{website_title}\n\n"
    "Passage:\n"
    "{passage}\n"
)


DEFAULT_LEVEL_LABELS = {
    1: "General Role",
    2: "General Supporting Category",
    3: "Core Supporting Role Concept",
    4: "This passage provides broad support for related information needs",
    5: "This passage provides specific supporting details useful for retrieval and answer grounding",
}

LEVEL_WORD_MAX = {1: 2, 2: 4, 3: 6, 4: 10, 5: 20}


@dataclass
class LongDocResult:
    doc_id: str
    prefix: str
    website_title: str
    paths: List[List[str]]
    parse_success: bool
    parse_retry_count: int
    was_token_trimmed: bool


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
    load_dotenv(dotenv_path=env_file, override=False)

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
    ):
        if not account_configs:
            raise ValueError("account_configs must not be empty")

        self._account_configs = list(account_configs)
        self._model = model
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = int(max_tokens)
        self._request_timeout = int(request_timeout)

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
        messages = [{"role": "user", "content": prompt}]
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
                if self._is_limit_exceeded_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] database error with limit signal, failover: {cfg.account}")
                    continue
                raise
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if self._is_limit_exceeded_error(message, cfg.account):
                    self._suspended_accounts.add(cfg.account.lower())
                    self._close_connection(cfg.account)
                    print(f"[Snowflake] runtime error with limit signal, failover: {cfg.account}")
                    continue
                raise

        raise RuntimeError("All configured Snowflake accounts failed for this request.") from last_error


def _list_subsets(data_dir: str) -> List[str]:
    long_dir = os.path.join(data_dir, "long_documents")
    if not os.path.isdir(long_dir):
        return []
    subsets: List[str] = []
    for name in sorted(os.listdir(long_dir)):
        if not name.endswith("-00000-of-00001.parquet"):
            continue
        subsets.append(name.replace("-00000-of-00001.parquet", ""))
    return subsets


def _resolve_subsets(subset_arg: str, data_dir: str) -> List[str]:
    available = _list_subsets(data_dir)
    if subset_arg.strip().lower() == "all":
        return available
    requested = [x.strip() for x in subset_arg.split(",") if x.strip()]
    missing = [x for x in requested if x not in available]
    if missing:
        raise ValueError(f"Unknown subset(s): {missing}. Available: {available}")
    return requested


def _read_parquet_rows(path: str) -> List[Dict[str, str]]:
    table = pq.read_table(path, columns=["id", "content"])
    rows = table.to_pylist()
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "id": str(row.get("id", "")).strip(),
                "content": str(row.get("content", "") or "").strip(),
            }
        )
    return out


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _safe_title_case(text: str) -> str:
    words = [w for w in text.split(" ") if w]
    if not words:
        return ""
    return " ".join([w[:1].upper() + w[1:].lower() if w else w for w in words])


def _strip_extension(name: str) -> str:
    return re.sub(r"\.(txt|html?)$", "", name, flags=re.IGNORECASE)


def _website_title_from_doc_id(doc_id: str) -> str:
    tail = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
    title = _strip_extension(tail)
    title = re.sub(r"[_\-]+", " ", title)
    title = re.sub(r"\b\d+\b", " ", title)
    title = _normalize_space(title)
    if not title:
        return "Untitled Document"
    return title


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return _normalize_space(text)
    words = _normalize_space(text).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _clean_json_candidate(text: str) -> str:
    cleaned = str(text or "")
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    cleaned = cleaned.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        fenced = [parts[i].strip() for i in range(1, len(parts), 2)]
        if fenced:
            cleaned = fenced[-1]
    return cleaned.strip()


def _normalize_label(label: str, level: int) -> str:
    raw = _normalize_space(label)
    raw = re.sub(r"\.(txt|html?)\b", "", raw, flags=re.IGNORECASE)
    raw = raw.strip(" \t\r\n\"'`.,:;|[]{}()")
    words = [w for w in raw.split(" ") if w]
    max_words = LEVEL_WORD_MAX[level]
    if len(words) > max_words:
        words = words[:max_words]
    normalized = " ".join(words).strip()
    if not normalized:
        normalized = DEFAULT_LEVEL_LABELS[level]
    if level <= 3:
        normalized = _safe_title_case(normalized)
    elif normalized:
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def _normalize_path(raw_path: Iterable[str]) -> Optional[List[str]]:
    if not isinstance(raw_path, list):
        return None
    if len(raw_path) < 5:
        return None
    norm: List[str] = []
    for level in range(1, 6):
        value = raw_path[level - 1] if level - 1 < len(raw_path) else ""
        norm.append(_normalize_label(str(value or ""), level))
    return norm


def _parse_paths_from_output(
    text: str,
    doc_id: str,
    max_alt_paths: int,
) -> Tuple[List[List[str]], bool]:
    cleaned = _clean_json_candidate(text)
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None

    parsed_paths: List[List[str]] = []
    if isinstance(obj, dict):
        primary_raw = obj.get("hierarchical_keywords", obj.get("keywords", []))
        primary = _normalize_path(primary_raw)
        if primary:
            parsed_paths.append(primary)
        alt_raw = obj.get("alternate_hierarchical_keywords", [])
        if isinstance(alt_raw, list):
            for item in alt_raw[: max(0, int(max_alt_paths))]:
                alt = _normalize_path(item)
                if alt:
                    parsed_paths.append(alt)

    dedup: List[List[str]] = []
    seen: Set[str] = set()
    for path in parsed_paths:
        key = "||".join(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)

    if dedup:
        return dedup, True

    fallback_title = _website_title_from_doc_id(doc_id)
    fallback = [
        DEFAULT_LEVEL_LABELS[1],
        DEFAULT_LEVEL_LABELS[2],
        DEFAULT_LEVEL_LABELS[3],
        DEFAULT_LEVEL_LABELS[4],
        _normalize_label(f"Specific support details from {fallback_title}", 5),
    ]
    return [fallback], False


def _build_prompt(
    *,
    passage_id: str,
    website_title: str,
    passage: str,
) -> str:
    return PROMPT_TEMPLATE.format(
        passage_id=passage_id,
        website_title=website_title,
        passage=passage,
    )


def _build_guarded_prompt(
    *,
    passage_id: str,
    website_title: str,
    passage: str,
    tokenizer=None,
    prompt_token_limit: Optional[int] = None,
) -> Tuple[str, bool]:
    prompt = _build_prompt(
        passage_id=passage_id,
        website_title=website_title,
        passage=passage,
    )
    if tokenizer is None or prompt_token_limit is None:
        return prompt, False
    if prompt_token_limit <= 0:
        return prompt, False
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= prompt_token_limit:
        return prompt, False

    words = passage.split(" ")
    lo = 0
    hi = len(words)
    best_prompt = prompt
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_passage = " ".join(words[:mid])
        cand_prompt = _build_prompt(
            passage_id=passage_id,
            website_title=website_title,
            passage=cand_passage,
        )
        cand_len = len(tokenizer.encode(cand_prompt, add_special_tokens=False))
        if cand_len <= prompt_token_limit:
            best_prompt = cand_prompt
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_prompt, best_len < len(words)


def _label_key(level: int, label: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "", str(label or "").lower())
    if not norm:
        norm = "misc"
    return f"L{level}|{norm}"


def _make_internal_node_id(level: int, label: str) -> str:
    return _label_key(level, label)


def _split_prefix(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else ""


def _strip_chunk_suffix(name: str) -> str:
    base = _strip_extension(name)
    # Intent: split-doc leaf IDs often end with chunk indices; removing numeric tails aligns them to long-document titles.
    while re.search(r"_[0-9]+$", base):
        base = re.sub(r"_[0-9]+$", "", base)
    return base


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _build_internal_desc(level: int, label: str, num_children: int, num_parents: int) -> str:
    return (
        f"Level {level} category node. Label: {label}. "
        f"Parent links: {num_parents}. Child links: {num_children}."
    )


def _build_projection_root_desc(subset: str, version: str) -> str:
    return (
        f"ROOT Node: category-first top-down projection tree for subset={subset}, version={version}. "
        "Built from long_documents hierarchy and attached split-document leaves."
    )


def _generate_long_doc_paths(
    *,
    generate_fn: Callable[[List[str]], List[str]],
    tokenizer=None,
    prompt_token_limit: Optional[int] = None,
    long_rows: List[Dict[str, str]],
    batch_size: int,
    max_desc_words: int,
    parse_retry_max: int,
    max_alt_paths: int,
) -> List[LongDocResult]:
    results: List[LongDocResult] = []
    pending: List[Dict[str, str]] = []

    def flush_batch(batch_rows: List[Dict[str, str]]) -> None:
        if not batch_rows:
            return
        prompts: List[str] = []
        trimmed_flags: List[bool] = []
        for row in batch_rows:
            doc_id = row["id"]
            website_title = _website_title_from_doc_id(doc_id)
            desc = _truncate_words(row["content"], max_desc_words)
            prompt, was_trimmed = _build_guarded_prompt(
                tokenizer=tokenizer,
                prompt_token_limit=prompt_token_limit,
                passage_id=doc_id,
                website_title=website_title,
                passage=desc,
            )
            prompts.append(prompt)
            trimmed_flags.append(was_trimmed)

        outputs = generate_fn(prompts)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Generation function returned {len(outputs)} outputs for {len(prompts)} prompts."
            )

        for row, text, was_trimmed, base_prompt in zip(batch_rows, outputs, trimmed_flags, prompts):
            doc_id = row["id"]
            paths, parse_ok = _parse_paths_from_output(
                text=text,
                doc_id=doc_id,
                max_alt_paths=max_alt_paths,
            )
            retry_count = 0
            if (not parse_ok) and parse_retry_max > 0:
                strict_prompt = (
                    base_prompt
                    + "\n\nSTRICT OUTPUT FORMAT: Return exactly one valid JSON object only."
                )
                for _ in range(parse_retry_max):
                    retry_out = generate_fn([strict_prompt])
                    retry_text = retry_out[0] if retry_out else ""
                    retry_count += 1
                    paths, parse_ok = _parse_paths_from_output(
                        text=retry_text,
                        doc_id=doc_id,
                        max_alt_paths=max_alt_paths,
                    )
                    if parse_ok:
                        break

            results.append(
                LongDocResult(
                    doc_id=doc_id,
                    prefix=_split_prefix(doc_id),
                    website_title=_website_title_from_doc_id(doc_id),
                    paths=paths,
                    parse_success=parse_ok,
                    parse_retry_count=retry_count,
                    was_token_trimmed=was_trimmed,
                )
            )

    for row in long_rows:
        pending.append(row)
        if len(pending) < batch_size:
            continue
        flush_batch(pending)
        pending = []
    if pending:
        flush_batch(pending)
    return results


def _apply_branching_cap(
    long_results: List[LongDocResult],
    max_branching: int,
) -> List[LongDocResult]:
    if max_branching <= 0:
        return long_results

    records: List[Dict[str, object]] = []
    for item in long_results:
        for path in item.paths:
            records.append(
                {
                    "doc_id": item.doc_id,
                    "prefix": item.prefix,
                    "website_title": item.website_title,
                    "path": list(path),
                }
            )
    if not records:
        return long_results

    for level in range(1, 6):
        parent_to_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        for rec in records:
            labels = rec["path"]
            parent_key = tuple(labels[: level - 1])
            child_label = labels[level - 1]
            parent_to_counts[parent_key][child_label] += 1

        parent_to_map: Dict[Tuple[str, ...], Dict[str, str]] = {}
        for parent_key, counts in parent_to_counts.items():
            mapping: Dict[str, str] = {}
            child_labels = sorted(counts.keys(), key=lambda x: (-counts[x], x))
            if len(child_labels) <= max_branching:
                for label in child_labels:
                    mapping[label] = label
                parent_to_map[parent_key] = mapping
                continue

            anchors = child_labels[:max_branching]
            for label in child_labels:
                if label in anchors:
                    mapping[label] = label
                    continue
                best_anchor = anchors[0]
                best_score = -1.0
                for anchor in anchors:
                    score = _similarity(_normalize_key(label), _normalize_key(anchor))
                    if score > best_score:
                        best_score = score
                        best_anchor = anchor
                mapping[label] = best_anchor
            parent_to_map[parent_key] = mapping

        for rec in records:
            labels = rec["path"]
            parent_key = tuple(labels[: level - 1])
            child_label = labels[level - 1]
            labels[level - 1] = parent_to_map[parent_key].get(child_label, child_label)

    by_doc: Dict[str, List[List[str]]] = defaultdict(list)
    for rec in records:
        by_doc[rec["doc_id"]].append(rec["path"])

    updated: List[LongDocResult] = []
    for item in long_results:
        seen: Set[str] = set()
        dedup_paths: List[List[str]] = []
        for path in by_doc.get(item.doc_id, item.paths):
            key = "||".join(path)
            if key in seen:
                continue
            seen.add(key)
            dedup_paths.append(path)
        if not dedup_paths:
            dedup_paths = item.paths
        updated.append(
            LongDocResult(
                doc_id=item.doc_id,
                prefix=item.prefix,
                website_title=item.website_title,
                paths=dedup_paths,
                parse_success=item.parse_success,
                parse_retry_count=item.parse_retry_count,
                was_token_trimmed=item.was_token_trimmed,
            )
        )
    return updated


def _compute_num_leaves(tree_node: Dict) -> int:
    children = tree_node.get("child") or []
    if not children:
        tree_node["num_leaves"] = 1
        return 1
    total = 0
    for child in children:
        total += _compute_num_leaves(child)
    tree_node["num_leaves"] = total
    return total


def _export_node_catalog(tree_dict: Dict, out_jsonl: str) -> None:
    _compute_num_leaves(tree_dict)
    records: List[Dict] = []

    def walk(node: Dict, path: Tuple[int, ...]) -> None:
        child = node.get("child") or []
        rec = {
            "path": list(path),
            "depth": len(path),
            "is_leaf": len(child) == 0,
            "num_children": len(child),
            "num_leaves": int(node.get("num_leaves", 1)),
            "id": node.get("id"),
            "desc": node.get("desc", ""),
        }
        records.append(rec)
        for idx, c in enumerate(child):
            walk(c, (*path, idx))

    walk(tree_dict, ())
    for idx, rec in enumerate(records):
        rec["registry_idx"] = idx

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    depth1_path = os.path.splitext(out_jsonl)[0] + "_depth1.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f_all, open(depth1_path, "w", encoding="utf-8") as f_d1:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            f_all.write(line + "\n")
            if rec["depth"] == 1:
                f_d1.write(line + "\n")


def _build_subset_artifacts(
    *,
    dataset: str,
    subset: str,
    long_rows: List[Dict[str, str]],
    split_rows: List[Dict[str, str]],
    long_results: List[LongDocResult],
    leaf_parent_cap: int,
    version: str,
    out_dir: str,
) -> Dict[str, object]:
    nodes: Dict[str, Dict[str, object]] = {}
    edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_types: Dict[Tuple[str, str], str] = {}
    incoming: Dict[str, Set[str]] = defaultdict(set)
    outgoing: Dict[str, Set[str]] = defaultdict(set)

    root_id = "L0|root"
    nodes[root_id] = {
        "id": root_id,
        "level": 0,
        "kind": "root",
        "label": "Root",
        "display_id": None,
        "desc": _build_projection_root_desc(subset=subset, version=version),
        "long_support_ids": set(),
        "split_support_ids": set(),
    }

    split_content_by_id = {row["id"]: row["content"] for row in split_rows}

    long_doc_l5_candidates: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    long_doc_title_key: Dict[str, str] = {}
    long_doc_prefix: Dict[str, str] = {}

    def ensure_internal_node(level: int, label: str) -> str:
        node_id = _make_internal_node_id(level, label)
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "level": level,
                "kind": "category",
                "label": label,
                "display_id": f"[L{level}] {label}",
                "desc": "",
                "long_support_ids": set(),
                "split_support_ids": set(),
            }
        return node_id

    def ensure_leaf_node(doc_id: str, content: str) -> str:
        if doc_id not in nodes:
            nodes[doc_id] = {
                "id": doc_id,
                "level": 6,
                "kind": "leaf",
                "label": doc_id,
                "display_id": doc_id,
                "desc": content,
                "long_support_ids": set(),
                "split_support_ids": {doc_id},
            }
        return doc_id

    def add_edge(parent_id: str, child_id: str, weight: float, edge_type: str) -> None:
        key = (parent_id, child_id)
        edge_weights[key] += float(weight)
        edge_types[key] = edge_type
        incoming[child_id].add(parent_id)
        outgoing[parent_id].add(child_id)

    for result in long_results:
        long_doc_title_key[result.doc_id] = _normalize_key(result.website_title)
        long_doc_prefix[result.doc_id] = result.prefix
        path_score_base = 1.0
        for path_idx, path in enumerate(result.paths):
            parent_id = root_id
            for level, label in enumerate(path, start=1):
                node_id = ensure_internal_node(level, label)
                add_edge(parent_id, node_id, weight=path_score_base, edge_type="category")
                nodes[node_id]["long_support_ids"].add(result.doc_id)
                parent_id = node_id
            # Intent: slight penalty keeps alternate paths for DAG coverage but preserves primary path preference.
            score = 1.0 - (0.05 * path_idx)
            long_doc_l5_candidates[result.doc_id].append((parent_id, score, path_idx))
            path_score_base = max(0.5, score)

    long_doc_by_prefix: Dict[str, List[str]] = defaultdict(list)
    for doc_id, pref in long_doc_prefix.items():
        long_doc_by_prefix[pref].append(doc_id)

    leaf_membership_rows: List[Dict[str, object]] = []
    for row in split_rows:
        split_id = row["id"]
        split_content = row["content"]
        split_pref = _split_prefix(split_id)
        split_tail = split_id.split("/", 1)[-1] if "/" in split_id else split_id
        split_base_key = _normalize_key(_website_title_from_doc_id(_strip_chunk_suffix(split_tail)))

        long_candidates = long_doc_by_prefix.get(split_pref, [])
        scored_parent_candidates: List[Tuple[float, str, str]] = []
        for long_doc_id in long_candidates:
            title_key = long_doc_title_key.get(long_doc_id, "")
            sim = _similarity(split_base_key, title_key)
            for l5_node_id, base_score, path_idx in long_doc_l5_candidates.get(long_doc_id, []):
                cand_score = (sim * 0.9) + (base_score * 0.1) - (0.01 * path_idx)
                scored_parent_candidates.append((cand_score, l5_node_id, long_doc_id))

        scored_parent_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        selected_parents: List[Tuple[float, str, str]] = []
        seen_l5: Set[str] = set()
        for item in scored_parent_candidates:
            if item[1] in seen_l5:
                continue
            seen_l5.add(item[1])
            selected_parents.append(item)
            if len(selected_parents) >= max(1, int(leaf_parent_cap)):
                break

        if not selected_parents:
            # Intent: keep every split document reachable by attaching to at least one category parent in the same subset.
            fallback_parent = None
            if long_candidates:
                first_long = long_candidates[0]
                l5_options = long_doc_l5_candidates.get(first_long, [])
                if l5_options:
                    fallback_parent = l5_options[0][0]
            if fallback_parent is None:
                fallback_parent = root_id
            selected_parents = [(0.0, fallback_parent, long_candidates[0] if long_candidates else "")]

        leaf_node_id = ensure_leaf_node(split_id, split_content)
        for score, parent_l5, parent_long_doc_id in selected_parents:
            add_edge(parent_l5, leaf_node_id, weight=max(0.01, score), edge_type="leaf_attach")
            nodes[parent_l5]["split_support_ids"].add(split_id)
            nodes[leaf_node_id]["long_support_ids"].add(parent_long_doc_id)

        leaf_membership_rows.append(
            {
                "doc_id": split_id,
                "prefix": split_pref,
                "selected_parent_node_ids": [x[1] for x in selected_parents],
                "selected_parent_long_doc_ids": [x[2] for x in selected_parents],
                "selected_scores": [round(float(x[0]), 6) for x in selected_parents],
            }
        )

    parent_choice: Dict[str, str] = {}
    for child_id, parents in incoming.items():
        if child_id == root_id:
            continue
        best_parent = None
        best_tuple = None
        for parent_id in parents:
            key = (parent_id, child_id)
            weight = edge_weights.get(key, 0.0)
            parent_level = int(nodes.get(parent_id, {}).get("level", -1))
            rank_key = (weight, -parent_level, parent_id)
            if (best_tuple is None) or (rank_key > best_tuple):
                best_tuple = rank_key
                best_parent = parent_id
        if best_parent is None:
            best_parent = root_id
        parent_choice[child_id] = best_parent

    proj_children: Dict[str, List[str]] = defaultdict(list)
    for child_id, parent_id in parent_choice.items():
        proj_children[parent_id].append(child_id)

    visited: Set[str] = set()
    queue: List[str] = [root_id]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in proj_children.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)

    for node_id in nodes:
        if node_id == root_id:
            continue
        if node_id in visited:
            continue
        proj_children[root_id].append(node_id)
        parent_choice[node_id] = root_id

    for node_id, node in nodes.items():
        if node_id == root_id:
            continue
        in_deg = len(incoming.get(node_id, set()))
        out_deg = len(outgoing.get(node_id, set()))
        if node["kind"] == "category":
            node["desc"] = _build_internal_desc(
                level=int(node["level"]),
                label=str(node["label"]),
                num_children=out_deg,
                num_parents=in_deg,
            )

    def child_sort_key(node_id: str) -> Tuple[int, int, str]:
        node = nodes[node_id]
        is_leaf = 1 if node["kind"] == "leaf" else 0
        return (int(node["level"]), is_leaf, str(node["display_id"]))

    def build_tree_dict(node_id: str) -> Dict:
        node = nodes[node_id]
        children_ids = sorted(proj_children.get(node_id, []), key=child_sort_key)
        child_nodes = [build_tree_dict(cid) for cid in children_ids]
        return {
            "id": node["display_id"],
            "desc": node["desc"],
            "child": child_nodes if child_nodes else None,
        }

    tree_dict = build_tree_dict(root_id)
    tree_dict["id"] = None

    dag_nodes_out: List[Dict[str, object]] = []
    for node_id, node in sorted(nodes.items(), key=lambda x: (int(x[1]["level"]), str(x[0]))):
        dag_nodes_out.append(
            {
                "id": node_id,
                "display_id": node["display_id"],
                "level": int(node["level"]),
                "kind": node["kind"],
                "label": node["label"],
                "desc": node["desc"],
                "num_parents": len(incoming.get(node_id, set())),
                "num_children": len(outgoing.get(node_id, set())),
                "long_support_count": len(node["long_support_ids"]),
                "split_support_count": len(node["split_support_ids"]),
            }
        )

    dag_edges_out: List[Dict[str, object]] = []
    for (parent_id, child_id), weight in sorted(edge_weights.items(), key=lambda x: (x[0][0], x[0][1])):
        dag_edges_out.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "weight": float(weight),
                "edge_type": edge_types.get((parent_id, child_id), "unknown"),
                "is_projection_parent": parent_choice.get(child_id) == parent_id,
            }
        )

    level_counter = Counter([int(x["level"]) for x in dag_nodes_out])
    multi_parent_counter = Counter(
        [
            int(node["level"])
            for node_id, node in nodes.items()
            if len(incoming.get(node_id, set())) > 1
        ]
    )
    leaf_parent_hist = Counter([len(x["selected_parent_node_ids"]) for x in leaf_membership_rows])

    report = {
        "meta": {
            "dataset": dataset,
            "subset": subset,
            "version": version,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "leaf_parent_cap": int(leaf_parent_cap),
        },
        "counts": {
            "num_long_documents": len(long_rows),
            "num_split_documents": len(split_rows),
            "num_dag_nodes": len(dag_nodes_out),
            "num_dag_edges": len(dag_edges_out),
            "num_projection_nodes": len(dag_nodes_out),
            "num_projection_edges": max(0, len(dag_nodes_out) - 1),
        },
        "level_distribution": {str(k): int(v) for k, v in sorted(level_counter.items())},
        "multi_parent_level_distribution": {
            str(k): int(v) for k, v in sorted(multi_parent_counter.items())
        },
        "leaf_parent_histogram": {str(k): int(v) for k, v in sorted(leaf_parent_hist.items())},
    }

    os.makedirs(out_dir, exist_ok=True)
    version_u = version.replace("-", "_")
    dag_json_path = os.path.join(out_dir, f"category_dag_topdown_{version_u}.json")
    dag_edge_jsonl_path = os.path.join(out_dir, f"category_dag_edges_topdown_{version_u}.jsonl")
    leaf_membership_path = os.path.join(out_dir, f"category_leaf_membership_topdown_{version_u}.jsonl")
    report_path = os.path.join(out_dir, f"category_build_report_topdown_{version_u}.json")
    projection_tree_path = os.path.join(out_dir, f"category_tree_projection_{version_u}.pkl")
    runtime_tree_path = os.path.join(out_dir, f"tree-category-topdown-{version}.pkl")
    node_catalog_path = os.path.join(out_dir, f"category_node_catalog_topdown_{version_u}.jsonl")

    with open(dag_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": report["meta"],
                "nodes": dag_nodes_out,
                "edges": dag_edges_out,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(dag_edge_jsonl_path, "w", encoding="utf-8") as f:
        for row in dag_edges_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(leaf_membership_path, "w", encoding="utf-8") as f:
        for row in leaf_membership_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pickle.dump(tree_dict, open(projection_tree_path, "wb"))
    pickle.dump(tree_dict, open(runtime_tree_path, "wb"))
    _export_node_catalog(tree_dict, node_catalog_path)

    return {
        "dag_json_path": dag_json_path,
        "dag_edge_jsonl_path": dag_edge_jsonl_path,
        "leaf_membership_path": leaf_membership_path,
        "report_path": report_path,
        "projection_tree_path": projection_tree_path,
        "runtime_tree_path": runtime_tree_path,
        "node_catalog_path": node_catalog_path,
        "report": report,
    }


def _save_long_results(
    out_dir: str,
    version: str,
    long_results: List[LongDocResult],
) -> str:
    version_u = version.replace("-", "_")
    out_path = os.path.join(out_dir, f"category_longdoc_paths_topdown_{version_u}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in long_results:
            f.write(
                json.dumps(
                    {
                        "doc_id": row.doc_id,
                        "prefix": row.prefix,
                        "website_title": row.website_title,
                        "paths": row.paths,
                        "parse_success": row.parse_success,
                        "parse_retry_count": row.parse_retry_count,
                        "was_token_trimmed": row.was_token_trimmed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build category-first top-down DAG from BRIGHT long_documents and "
            "export runtime-compatible projected tree."
        )
    )
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True, help="Subset name, comma list, or 'all'")
    parser.add_argument("--data_dir", type=str, default="data/BRIGHT")
    parser.add_argument("--trees_root", type=str, default="trees")
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--llm", type=str, required=True, help="Snowflake Cortex model name")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--snowflake_request_timeout", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_desc_words", type=int, default=4096)
    parser.add_argument("--parse_retry_max", type=int, default=1)
    parser.add_argument("--max_alt_paths", type=int, default=1)
    parser.add_argument("--leaf_parent_cap", type=int, default=2)
    parser.add_argument("--max_branching", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subsets = _resolve_subsets(args.subset, args.data_dir)
    if not subsets:
        raise ValueError(f"No subsets available under {args.data_dir}/long_documents")

    account_configs = _load_snowflake_account_configs(args.env_file)
    print(
        f"[Snowflake] model={args.llm} "
        f"accounts={len(account_configs)} env_file={args.env_file}"
    )
    router = SnowflakeCortexRouter(
        account_configs=account_configs,
        model=args.llm,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        request_timeout=args.snowflake_request_timeout,
    )

    def generate_fn(prompts: List[str]) -> List[str]:
        return [router.complete(prompt) for prompt in prompts]

    try:
        for subset in subsets:
            out_dir = os.path.join(args.trees_root, args.dataset, subset)
            version_u = args.version.replace("-", "_")
            report_path = os.path.join(out_dir, f"category_build_report_topdown_{version_u}.json")
            if os.path.exists(report_path) and (not args.overwrite):
                print(f"[Skip] {subset}: {report_path} already exists. Use --overwrite to rebuild.")
                continue

            long_path = os.path.join(args.data_dir, "long_documents", f"{subset}-00000-of-00001.parquet")
            split_path = os.path.join(args.data_dir, "documents", f"{subset}-00000-of-00001.parquet")
            if not os.path.exists(long_path):
                raise FileNotFoundError(f"Missing long_documents parquet: {long_path}")
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Missing documents parquet: {split_path}")

            print(f"[Build] subset={subset}")
            long_rows = _read_parquet_rows(long_path)
            split_rows = _read_parquet_rows(split_path)
            print(f"[Load] long_docs={len(long_rows)} split_docs={len(split_rows)}")

            long_results = _generate_long_doc_paths(
                generate_fn=generate_fn,
                long_rows=long_rows,
                batch_size=args.batch_size,
                max_desc_words=args.max_desc_words,
                parse_retry_max=args.parse_retry_max,
                max_alt_paths=args.max_alt_paths,
            )
            # Intent: keep top-down fanout bounded to the paper-style branching regime (M up to ~10-20).
            long_results = _apply_branching_cap(
                long_results=long_results,
                max_branching=args.max_branching,
            )

            parse_ok = sum([1 for x in long_results if x.parse_success])
            parse_fail = len(long_results) - parse_ok
            print(f"[LLM] parse_success={parse_ok} parse_fail={parse_fail}")

            os.makedirs(out_dir, exist_ok=True)
            long_results_path = _save_long_results(out_dir=out_dir, version=args.version, long_results=long_results)
            artifact_paths = _build_subset_artifacts(
                dataset=args.dataset,
                subset=subset,
                long_rows=long_rows,
                split_rows=split_rows,
                long_results=long_results,
                leaf_parent_cap=args.leaf_parent_cap,
                version=args.version,
                out_dir=out_dir,
            )

            report = artifact_paths["report"]
            report_meta = report["meta"] if isinstance(report, dict) else {}
            report_counts = report["counts"] if isinstance(report, dict) else {}
            print(
                "[Done] subset={subset} dag_nodes={dag_nodes} dag_edges={dag_edges} tree={tree_path}".format(
                    subset=subset,
                    dag_nodes=report_counts.get("num_dag_nodes", -1),
                    dag_edges=report_counts.get("num_dag_edges", -1),
                    tree_path=artifact_paths["runtime_tree_path"],
                )
            )
            print(f"[Meta] {json.dumps(report_meta, ensure_ascii=False)}")
            print(f"[LongDocPaths] {long_results_path}")
    finally:
        router.close_all()


if __name__ == "__main__":
    main()
