"""CLI 엔트리 포인트

사용법:
    python -m realestate_abm --preset korea_2024 --steps 24
    python -m realestate_abm --preset simple_3city --steps 12
"""

import argparse
import sys
from pathlib import Path

from .simulation.engine import SimulationEngine


def main():
    parser = argparse.ArgumentParser(description="부동산 ABM 시뮬레이션")
    parser.add_argument("--preset", type=str, default="korea_2024",
                        help="프리셋 이름 (korea_2024, simple_3city)")
    parser.add_argument("--preset-dir", type=str, default=None,
                        help="프리셋 디렉토리 직접 지정")
    parser.add_argument("--steps", type=int, default=None,
                        help="시뮬레이션 스텝 수 (개월)")
    parser.add_argument("--agents", type=int, default=None,
                        help="에이전트 수 (기본: 프리셋에서 로드)")
    parser.add_argument("--seed", type=int, default=None,
                        help="랜덤 시드")
    parser.add_argument("--quiet", action="store_true",
                        help="진행 출력 끄기")

    args = parser.parse_args()

    # 프리셋 디렉토리 결정
    if args.preset_dir:
        preset_dir = Path(args.preset_dir)
    else:
        preset_dir = Path(__file__).parent / "presets" / args.preset

    if not preset_dir.exists():
        print(f"Error: Preset directory not found: {preset_dir}")
        sys.exit(1)

    # 엔진 생성
    engine = SimulationEngine.from_preset(preset_dir)

    # CLI 인자로 오버라이드
    if args.steps is not None:
        engine.config.simulation.num_steps = args.steps
    if args.agents is not None:
        engine.config.simulation.num_households = args.agents
    if args.seed is not None:
        engine.config.simulation.seed = args.seed
        engine.rng = __import__('numpy').random.default_rng(args.seed)

    # 실행
    summary = engine.run(progress=not args.quiet)

    if args.quiet:
        import json
        import numpy as np

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        print(json.dumps(summary, indent=2, ensure_ascii=False, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
