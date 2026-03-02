#!/usr/bin/env python3
"""
Snowflake Custom Provider Utilities

Snowflake Cortex AI provider에서 사용하는 유틸리티 함수들을 포함합니다.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def discover_snowflake_accounts() -> Dict[str, Dict[str, str]]:
    """
    litellm_config.yaml에서 Snowflake 계정들을 자동으로 발견합니다.

    Returns:
        Dict[str, Dict[str, str]]: 계정 별칭을 키로 하는 계정 정보 딕셔너리
    """
    return discover_from_litellm_config()


def discover_from_litellm_config() -> Dict[str, Dict[str, str]]:
    """
    litellm_config.yaml에서 Snowflake 계정들을 발견합니다.

    Returns:
        Dict[str, Dict[str, str]]: 계정 별칭을 키로 하는 계정 정보 딕셔너리
    """
    import yaml

    # 설정 파일 경로 찾기
    possible_paths = [
        "/app/litellm_config.yaml",  # 컨테이너 내부 경로
        "/root/Projects/litellm/litellm_config.yaml",  # 호스트 경로
        "/data/interns/jjkim/Projects/litellm/litellm_config.yaml",
        "litellm_config.yaml",
        "../litellm_config.yaml",
        "../../litellm_config.yaml",
        "../../../litellm_config.yaml",
    ]

    config_path = None
    for path in possible_paths:
        if Path(path).exists():
            config_path = path
            break

    if not config_path:
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_list = config.get("model_list", [])
        account_configs = {}

        for model_item in model_list:
            model_name = model_item.get("model_name", "")
            custom_config = model_item.get("custom_config", {})

            # Snowflake 모델인지 확인
            if model_name.startswith("snowflake/") and custom_config:
                # 계정 별칭 추출
                account_alias = extract_account_alias_from_model(model_name)

                if account_alias not in account_configs:
                    # API 엔드포인트 생성
                    account_identifier = custom_config.get("account_identifier", "")
                    api_endpoint = f"https://{account_identifier.lower()}.snowflakecomputing.com/api/v2/cortex/inference:complete"

                    account_configs[account_alias] = {
                        "account_identifier": account_identifier,
                        "username": custom_config.get("username", ""),
                        "private_key_path": custom_config.get("private_key_path", ""),
                        "api_endpoint": api_endpoint,
                    }

                    logger.info(
                        f"litellm_config.yaml에서 Snowflake 계정 발견: {account_alias} ({account_identifier})"
                    )

        return account_configs

    except Exception as e:
        logger.warning(f"litellm_config.yaml 로드 실패: {e}")
        return {}


def extract_account_alias_from_model(model_name: str) -> str:
    """
    모델명에서 계정 별칭을 추출합니다.

    Args:
        model_name: 전체 모델명 (예: "snowflake/mistral-7b-dev")

    Returns:
        str: 계정 별칭 (예: "dev")
    """
    clean_name = model_name.replace("snowflake/", "")

    for suffix in ["-dev", "-test", "-prod", "-staging"]:
        if clean_name.endswith(suffix):
            return suffix[1:]  # - 제거

    return "main"


def get_default_key_path(account_alias: str) -> str:
    """
    계정 별칭에 따른 기본 키 파일 경로를 반환합니다.

    Args:
        account_alias: 계정 별칭 (예: "main", "dev")

    Returns:
        str: 키 파일 경로
    """
    # 컨테이너 내부 경로 우선, 호스트 경로 fallback
    base_paths = ["/app/keys", "/root/Projects/litellm/keys"]

    for base_path in base_paths:
        if account_alias == "main":
            key_path = f"{base_path}/snowflake_rsa_key.pem"
        else:
            key_path = f"{base_path}/snowflake_{account_alias}_rsa_key.pem"

        if Path(key_path).exists():
            return key_path

    # 기본값 (컨테이너 경로)
    if account_alias == "main":
        return "/app/keys/snowflake_rsa_key.pem"
    else:
        return f"/app/keys/snowflake_{account_alias}_rsa_key.pem"


def validate_account_config(account_config: Dict[str, str]) -> bool:
    """
    계정 설정의 유효성을 검증합니다.

    Args:
        account_config: 계정 설정 딕셔너리

    Returns:
        bool: 유효성 검증 결과
    """
    required_fields = [
        "account_identifier",
        "username",
        "private_key_path",
        "api_endpoint",
    ]

    for field in required_fields:
        if field not in account_config or not account_config[field]:
            logger.error(f"필수 필드 누락: {field}")
            return False

    # 키 파일 존재 확인
    key_path = Path(account_config["private_key_path"])
    if not key_path.exists():
        logger.error(f"키 파일을 찾을 수 없습니다: {key_path}")
        return False

    # 키 파일 권한 확인 (Unix 시스템)
    if hasattr(os, "stat"):
        try:
            stat_info = key_path.stat()
            if stat_info.st_mode & 0o077:
                logger.warning(
                    f"키 파일 권한이 안전하지 않습니다: {key_path}. 권장: chmod 600 {key_path}"
                )
        except Exception as e:
            logger.warning(f"키 파일 권한 확인 실패: {e}")

    return True


def get_model_account_mapping(
    model_name: str, available_accounts: List[str]
) -> Optional[str]:
    """
    모델명에서 계정 정보를 추출합니다.

    Args:
        model_name: 모델명 (예: "snowflake/mistral-7b-dev")
        available_accounts: 사용 가능한 계정 목록

    Returns:
        Optional[str]: 매핑된 계정 별칭
    """
    # snowflake/ 접두사 제거
    clean_model = model_name.replace("snowflake/", "")

    # 계정 suffix 확인
    for suffix in ["-dev", "-test", "-prod", "-staging"]:
        if clean_model.endswith(suffix):
            account_alias = suffix[1:]  # - 제거
            if account_alias in available_accounts:
                return account_alias

    # 기본 계정 반환
    if "main" in available_accounts:
        return "main"
    elif available_accounts:
        return available_accounts[0]

    return None


def extract_model_name(full_model_name: str) -> str:
    """
    전체 모델명에서 실제 Snowflake 모델명을 추출합니다.

    Args:
        full_model_name: 전체 모델명 (예: "snowflake/mistral-7b-dev")

    Returns:
        str: 실제 모델명 (예: "mistral-7b")
    """
    # snowflake/ 접두사 제거
    model_name = full_model_name.replace("snowflake/", "")

    # 계정 suffix 제거
    for suffix in ["-dev", "-test", "-prod", "-staging"]:
        if model_name.endswith(suffix):
            model_name = model_name.replace(suffix, "")
            break

    return model_name


def setup_logging(level: str = "INFO") -> None:
    """
    Snowflake provider 로깅을 설정합니다.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Snowflake provider 전용 로거 설정
    snowflake_logger = logging.getLogger("litellm_ollama_manager.providers.snowflake")
    snowflake_logger.setLevel(log_level)

    # 핸들러가 없으면 추가
    if not snowflake_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        snowflake_logger.addHandler(handler)

    logger.info(f"Snowflake provider 로깅 설정 완료: {level}")


def get_environment_info() -> Dict[str, any]:
    """
    현재 환경 정보를 반환합니다.

    Returns:
        Dict[str, any]: 환경 정보
    """
    accounts = discover_snowflake_accounts()

    return {
        "provider_name": "snowflake-cortex",
        "provider_version": "1.0.0",
        "accounts_discovered": len(accounts),
        "account_aliases": list(accounts.keys()),
        "environment_variables": {
            key: "***" if "KEY" in key or "TOKEN" in key else value
            for key, value in os.environ.items()
            if key.startswith("SNOWFLAKE_")
        },
    }
