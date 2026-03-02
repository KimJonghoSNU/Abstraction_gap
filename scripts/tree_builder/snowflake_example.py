import os
import json
import re
import string
import pdb
from typing import Any

from ftfy import fix_text
import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.errors import ProgrammingError

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def _convert_obj_to_sql_string(obj: Any) -> str:
    if isinstance(obj, str):
        s = obj.replace("'", "''")
        s = s.replace("\\", "\\\\")
        return f"'{s}'"
    elif isinstance(obj, dict):
        s = "{"
        for k, v in obj.items():
            s += f"'{k}': {_convert_obj_to_sql_string(v)},"
        s = s[:-1] + "}"
        return s
    elif isinstance(obj, list):
        s = "["
        for v in obj:
            s += f"{_convert_obj_to_sql_string(v)},"
        s = s[:-1] + "]"
        return s
    else:
        return json.dumps(obj).replace("'", "''")


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.strip().split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_rag_instuction(question, documents):
    instruction = f"You will answer a question based on the following snippet:\n\n{documents}\n\nUse the information provided in the snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\nReturn only the final answer with no additional explanation or reasoning."
    return instruction


def get_rewrite_instruction(question, documents):
    instruction = (
        "Think step by step to use the provided documents to answer a user's question.\n\n"
        f"Question:\n{question}\n\n"
        f"Documents:\n{documents}\n\n"
    )
    return instruction


def build_cortex_complete_sql(model: str, messages: list[dict], **kwargs) -> tuple[str, tuple]:
    actual_model = model.split("@", 1)[0] if "@" in model else model
    messages_json = json.dumps(messages)

    options = {}
    if "temperature" in kwargs:
        options["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        options["max_tokens"] = kwargs["max_tokens"]
    if "top_p" in kwargs:
        options["top_p"] = kwargs["top_p"]
    if "region" in kwargs:
        options["region"] = kwargs["region"]

    options_json = json.dumps(options) if options else "{}"

    sql = """
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        %s,
        PARSE_JSON(%s),
        PARSE_JSON(%s)
    ) AS response
    """
    return sql, (actual_model, messages_json, options_json)


def run_cortex_complete(conn, model: str, messages: list[dict], **kwargs) -> str:
    def _run_once(call_kwargs: dict) -> str:
        sql, params = build_cortex_complete_sql(model, messages, **call_kwargs)
        cursor = conn.cursor(DictCursor)
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            raise RuntimeError("Snowflake returned empty result")

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
        return content

    def _is_region_unavailable_error(err: Exception) -> bool:
        msg = str(err).lower()
        return "unavailable in your region" in msg or "cross region inference" in msg

    try:
        return _run_once(kwargs)
    except ProgrammingError as first_error:
        if not _is_region_unavailable_error(first_error):
            raise

        fallback_regions = [
            r.strip()
            for r in os.getenv(
                "SNOWFLAKE_CORTEX_FALLBACK_REGIONS", "AWS_US,ANY_REGION"
            ).split(",")
            if r.strip()
        ]

        last_error = first_error
        for region in fallback_regions:
            retry_kwargs = dict(kwargs)
            retry_kwargs["region"] = region
            try:
                print(f"Region 제한 감지: region={region} 재시도")
                return _run_once(retry_kwargs)
            except ProgrammingError as retry_error:
                last_error = retry_error
                if not _is_region_unavailable_error(retry_error):
                    raise

        raise last_error


def enable_cross_region_inference(conn, region_value: str = "ANY_REGION") -> None:
    cursor = conn.cursor()
    try:
        sql = (
            "ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = "
            f"'{region_value}'"
        )
        cursor.execute(sql)
        print(f"CORTEX_ENABLED_CROSS_REGION set to {region_value}")
    except ProgrammingError as e:
        # ACCOUNTADMIN 권한이 없으면 ALTER ACCOUNT는 실패할 수 있다.
        print(f"Could not set CORTEX_ENABLED_CROSS_REGION ({e})")
    finally:
        cursor.close()


connection_params = {
    "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
    "user": os.environ.get("SNOWFLAKE_USERNAME"),
    "password": os.environ.get("SNOWFLAKE_PASSWORD"),
    "role": os.environ.get("SNOWFLAKE_ROLE"),
    "database": os.environ.get("SNOWFLAKE_DATABASE"),
    "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
}

conn = snowflake.connector.connect(**connection_params)
enable_cross_region_inference(
    conn, os.getenv("SNOWFLAKE_CORTEX_CROSS_REGION", "ANY_REGION")
)

cache_path = "/data4/jaeyoung/Search-o1/cache/msmarco_passage/search_cache_train_tevatron_full_72k.json"
cache = json.load(open(cache_path, "r"))

for i, (query, docs) in enumerate(cache.items()):
    if i < 5:
        continue
    if i > 8:
        break

    messages = []
    system_prompt = (
        "You are an AI assistant that analyzes complex questions and identifies which documents best support answering them.\n\n"
        "Given a user's query and a set of documents, your task is to:\n"
        "1. Generate a reasoning trace, thinking step by step about what knowledge or types of information are necessary to answer the query. These should be abstract but specific enough to guide document selection.\n"
        "2. Select and rank at least 10 documents that best support the reasoning steps. Consider how each document contributes to the reasoning process. Order them from most to least useful using `>` between document IDs (e.g., [3] > [7]).\n\n"
        "Use the following format:\n"
        "[Reasoning Trace]\n"
        "Step 1: <First reasoning step>\n"
        "Step 2: <Second reasoning step>\n"
        "...\n"
        "Step N: <Final reasoning step>\n\n"
        "[Document Ranking]\n"
        "[9] > [5] > [6] > ... > [12]\n\n"
        "Only produce the output in the format shown above."
    )
    messages.append({"role": "system", "content": system_prompt})

    documents = ""
    for idx, doc in enumerate(docs):
        if "content" in doc:
            contents = doc["content"].strip()
        elif "contents" in doc:
            contents = doc["contents"].strip()
        else:
            raise ValueError("Document has neither 'content' nor 'contents'")

        contents = " ".join(contents.split(" ")[:400])
        documents += f"[{idx + 1}]: {contents}\\n"

    query = query.strip()
    documents = documents.strip()

    user_prompt = (
        "[Query]\\n"
        f"{query}\\n\\n"
        "[Documents]\\n"
        f"{documents}\\n\\n"
    )

    user_prompt = fix_text(user_prompt)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = run_cortex_complete(
            conn,
            # model="claude-4-5-sonnet",
            model="claude-sonnet-4-5",
            messages=messages,
            max_tokens=3072,
            temperature=0.0,
            top_p=1.0,
        )
        print(f"[{i}] response:\n{response}\n")
    except ProgrammingError as e:
        print(f"[{i}] Snowflake SQL error: {e}")
        raise

    pdb.set_trace()

conn.close()
