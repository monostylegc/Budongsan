"""JSON → Python 설정 로더"""

import json
from pathlib import Path
from .schema import ScenarioConfig, AgentsPresetConfig, InstitutionsConfig


def load_scenario(preset_dir: str | Path) -> ScenarioConfig:
    """프리셋 디렉토리에서 시나리오 로드

    Args:
        preset_dir: 프리셋 디렉토리 경로 (scenario.json이 있는 곳)

    Returns:
        통합된 ScenarioConfig
    """
    preset_dir = Path(preset_dir)

    # 1. scenario.json (마스터 설정)
    scenario_path = preset_dir / "scenario.json"
    if scenario_path.exists():
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
    else:
        scenario_data = {}

    # 2. institutions.json (제도 환경)
    inst_path = preset_dir / "institutions.json"
    if inst_path.exists():
        with open(inst_path, 'r', encoding='utf-8') as f:
            inst_data = json.load(f)
        scenario_data.setdefault("institutions", {}).update(inst_data)

    # 3. agents.json (에이전트 설정)
    agents_path = preset_dir / "agents.json"
    if agents_path.exists():
        with open(agents_path, 'r', encoding='utf-8') as f:
            agents_data = json.load(f)
        scenario_data.setdefault("agents", {}).update(agents_data)

    return ScenarioConfig(**scenario_data)


def load_scenario_from_dict(data: dict) -> ScenarioConfig:
    """딕셔너리에서 직접 로드"""
    return ScenarioConfig(**data)
