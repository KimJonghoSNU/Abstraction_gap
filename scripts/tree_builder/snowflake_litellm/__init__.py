#!/usr/bin/env python3
"""
Snowflake Cortex AI Custom Provider for LiteLLM

이 모듈은 Snowflake Cortex AI를 LiteLLM에서 사용할 수 있도록 하는 custom provider입니다.
SQL 기반의 다중 계정 지원을 통해 snowflake-connector-python을 사용합니다.

Features:
- SQL-based Snowflake CORTEX.COMPLETE() calls
- Multi-account support via environment variables
- Thread-safe connection management
- JSON message format support for LiteLLM
"""

from .handler import SnowflakeConnectionManager, SnowflakeCortexLLM

__all__ = [
    "SnowflakeCortexLLM",
    "SnowflakeConnectionManager",
]

# Provider 메타데이터
PROVIDER_NAME = "snowflake-cortex"
PROVIDER_VERSION = "1.0.0"
SUPPORTED_MODELS = [
    "mistral-7b",
    "mistral-large",
    "llama3.1-8b",
    "llama3.1-405b",
    "gemma-7b",
]

# 환경변수 패턴
ENV_PATTERNS = {
    "account_id": "SNOWFLAKE_ACCOUNT_ID_{alias}",
    "username": "SNOWFLAKE_USERNAME_{alias}",
    "private_key_path": "SNOWFLAKE_PRIVATE_KEY_PATH_{alias}",
}
