"""기본 설정값"""

from .schema import ScenarioConfig


def get_default_config() -> ScenarioConfig:
    """기본 설정 반환"""
    return ScenarioConfig()
