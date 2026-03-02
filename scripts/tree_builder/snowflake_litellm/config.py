"""
Snowflake Custom Provider Configuration

Snowflake Cortex AI provider의 설정 관리를 담당합니다.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SnowflakeAccountConfig:
    """Snowflake 계정 설정"""

    account_identifier: str
    username: str
    private_key_path: str
    api_endpoint: str
    alias: str

    def to_dict(self) -> Dict[str, str]:
        """딕셔너리로 변환"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str], alias: str) -> "SnowflakeAccountConfig":
        """딕셔너리에서 생성"""
        return cls(
            account_identifier=data["account_identifier"],
            username=data["username"],
            private_key_path=data["private_key_path"],
            api_endpoint=data["api_endpoint"],
            alias=alias,
        )


@dataclass
class SnowflakeProviderConfig:
    """Snowflake Provider 전체 설정"""

    accounts: Dict[str, SnowflakeAccountConfig]
    default_account: str
    jwt_expiry_minutes: int = 59  # JWT 토큰 만료 시간 (분)
    request_timeout: int = 30  # API 요청 타임아웃 (초)
    max_retries: int = 3  # 최대 재시도 횟수
    retry_delay: float = 1.0  # 재시도 지연 시간 (초)

    def get_account_config(
        self, alias: Optional[str] = None
    ) -> Optional[SnowflakeAccountConfig]:
        """계정 설정 반환"""
        target_alias = alias or self.default_account
        return self.accounts.get(target_alias)

    def list_accounts(self) -> List[str]:
        """사용 가능한 계정 목록 반환"""
        return list(self.accounts.keys())

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "accounts": {
                alias: config.to_dict() for alias, config in self.accounts.items()
            },
            "default_account": self.default_account,
            "jwt_expiry_minutes": self.jwt_expiry_minutes,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }


class SnowflakeConfigManager:
    """Snowflake Provider 설정 관리자"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[SnowflakeProviderConfig] = None

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로 반환"""
        return "/root/Projects/litellm/litellm_ollama_manager/providers/snowflake/config.json"

    def load_config(self) -> SnowflakeProviderConfig:
        """설정 로드"""
        if self._config is not None:
            return self._config

        # 환경변수에서 동적 생성
        config = self._create_config_from_env()

        # 파일에서 로드 (있는 경우)
        if Path(self.config_path).exists():
            try:
                file_config = self._load_config_from_file()
                # 환경변수 설정과 파일 설정 병합
                config = self._merge_configs(config, file_config)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패, 환경변수 설정 사용: {e}")

        self._config = config
        return config

    def _create_config_from_env(self) -> SnowflakeProviderConfig:
        """litellm_config.yaml에서 설정 생성"""
        accounts = self._load_from_litellm_config()

        if not accounts:
            raise ValueError(
                "Snowflake 계정 설정을 찾을 수 없습니다. litellm_config.yaml을 확인하세요."
            )

        # 기본 계정 결정
        default_account = "main" if "main" in accounts else list(accounts.keys())[0]

        return SnowflakeProviderConfig(
            accounts=accounts,
            default_account=default_account,
            jwt_expiry_minutes=59,  # 기본값
            request_timeout=30,  # 기본값
            max_retries=3,  # 기본값
            retry_delay=1.0,  # 기본값
        )

    def _find_config_file(self) -> Optional[str]:
        """litellm_config.yaml 파일을 찾습니다"""
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

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    def _extract_account_alias_from_model(self, model_name: str) -> str:
        """모델명에서 계정 별칭 추출"""
        clean_name = model_name.replace("snowflake/", "")

        for suffix in ["-dev", "-test", "-prod", "-staging"]:
            if clean_name.endswith(suffix):
                return suffix[1:]

        return "main"

    def _load_from_litellm_config(self) -> Dict[str, SnowflakeAccountConfig]:
        """litellm_config.yaml에서 계정 설정 로드"""
        from pathlib import Path

        import yaml

        # 설정 파일 경로 찾기
        config_path = self._find_config_file()
        if not config_path:
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            model_list = config.get("model_list", [])
            accounts = {}

            for model_item in model_list:
                model_name = model_item.get("model_name", "")
                custom_config = model_item.get("custom_config", {})

                # Snowflake 모델인지 확인
                if model_name.startswith("snowflake/") and custom_config:
                    # 계정 별칭 추출
                    account_alias = self._extract_account_alias_from_model(model_name)

                    if account_alias not in accounts:
                        # API 엔드포인트 생성
                        account_identifier = custom_config.get("account_identifier", "")
                        api_endpoint = f"https://{account_identifier.lower()}.snowflakecomputing.com/api/v2/cortex/inference:complete"

                        account_config_data = {
                            "account_identifier": account_identifier,
                            "username": custom_config.get("username", ""),
                            "private_key_path": custom_config.get(
                                "private_key_path", ""
                            ),
                            "api_endpoint": api_endpoint,
                        }

                        accounts[account_alias] = SnowflakeAccountConfig.from_dict(
                            account_config_data, account_alias
                        )

            return accounts

        except Exception as e:
            logger.warning(f"litellm_config.yaml 로드 실패: {e}")
            return {}

    def _load_config_from_file(self) -> SnowflakeProviderConfig:
        """파일에서 설정 로드"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        accounts = {}
        for alias, account_data in data["accounts"].items():
            accounts[alias] = SnowflakeAccountConfig.from_dict(account_data, alias)

        return SnowflakeProviderConfig(
            accounts=accounts,
            default_account=data["default_account"],
            jwt_expiry_minutes=data.get("jwt_expiry_minutes", 59),
            request_timeout=data.get("request_timeout", 30),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
        )

    def _merge_configs(
        self, env_config: SnowflakeProviderConfig, file_config: SnowflakeProviderConfig
    ) -> SnowflakeProviderConfig:
        """환경변수 설정과 파일 설정 병합"""
        # 환경변수의 계정 설정이 우선
        merged_accounts = file_config.accounts.copy()
        merged_accounts.update(env_config.accounts)

        # 기본 계정은 환경변수 우선
        default_account = (
            env_config.default_account
            if env_config.accounts
            else file_config.default_account
        )

        return SnowflakeProviderConfig(
            accounts=merged_accounts,
            default_account=default_account,
            jwt_expiry_minutes=env_config.jwt_expiry_minutes,
            request_timeout=env_config.request_timeout,
            max_retries=env_config.max_retries,
            retry_delay=env_config.retry_delay,
        )

    def save_config(self, config: SnowflakeProviderConfig) -> None:
        """설정 저장"""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Snowflake provider 설정 저장 완료: {self.config_path}")

    def reload_config(self) -> SnowflakeProviderConfig:
        """설정 다시 로드"""
        self._config = None
        return self.load_config()

    def validate_config(self, config: SnowflakeProviderConfig) -> List[str]:
        """설정 유효성 검증"""
        errors = []

        if not config.accounts:
            errors.append("계정 설정이 없습니다.")
            return errors

        if config.default_account not in config.accounts:
            errors.append(
                f"기본 계정 '{config.default_account}'이 계정 목록에 없습니다."
            )

        for alias, account in config.accounts.items():
            # 키 파일 존재 확인
            key_path = Path(account.private_key_path)
            if not key_path.exists():
                errors.append(
                    f"계정 '{alias}': 키 파일을 찾을 수 없습니다 - {key_path}"
                )

            # 필수 필드 확인
            if not account.account_identifier:
                errors.append(f"계정 '{alias}': account_identifier가 비어있습니다.")

            if not account.username:
                errors.append(f"계정 '{alias}': username이 비어있습니다.")

            if not account.api_endpoint:
                errors.append(f"계정 '{alias}': api_endpoint가 비어있습니다.")

        # 숫자 값 검증
        if config.jwt_expiry_minutes < 1 or config.jwt_expiry_minutes > 60:
            errors.append("jwt_expiry_minutes는 1-60 사이여야 합니다.")

        if config.request_timeout < 1:
            errors.append("request_timeout은 1초 이상이어야 합니다.")

        if config.max_retries < 0:
            errors.append("max_retries는 0 이상이어야 합니다.")

        if config.retry_delay < 0:
            errors.append("retry_delay는 0 이상이어야 합니다.")

        return errors


# 전역 설정 관리자 인스턴스
_config_manager: Optional[SnowflakeConfigManager] = None


def get_config_manager() -> SnowflakeConfigManager:
    """전역 설정 관리자 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = SnowflakeConfigManager()
    return _config_manager


def get_provider_config() -> SnowflakeProviderConfig:
    """Provider 설정 반환"""
    return get_config_manager().load_config()


def reload_provider_config() -> SnowflakeProviderConfig:
    """Provider 설정 다시 로드"""
    return get_config_manager().reload_config()
