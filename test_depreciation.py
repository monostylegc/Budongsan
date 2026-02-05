"""노후화/멸실 기능 테스트"""

import sys
sys.path.insert(0, "src")

from realestate import Simulation, Config

def test_depreciation():
    print("=" * 60)
    print("노후화/멸실 기능 테스트")
    print("=" * 60)

    # 작은 규모로 테스트
    config = Config(
        num_households=5000,
        num_houses=3000,
        num_steps=60,  # 5년
        seed=42
    )

    sim = Simulation(config, arch="vulkan")
    sim.initialize()

    # 초기 상태 확인
    initial_stats = sim.houses.get_condition_stats()
    print(f"\n초기 상태:")
    print(f"  활성 주택: {initial_stats['active_count']:,}채")
    print(f"  평균 상태: {initial_stats['mean_condition']:.3f}")
    print(f"  평균 연식: {initial_stats['mean_age']:.1f}년")
    print(f"  30년 이상: {initial_stats['old_buildings_30y']:,}채")
    print(f"  40년 이상: {initial_stats['old_buildings_40y']:,}채")

    # 시뮬레이션 실행
    print(f"\n시뮬레이션 실행 (60개월)...")

    for month in range(60):
        sim.step()

        if (month + 1) % 12 == 0:
            year = (month + 1) // 12
            stats = sim.houses.get_condition_stats()
            demo_stats = sim.demolition_history[-1] if sim.demolition_history else {}
            total_demolished = sum(d['total_count'] for d in sim.demolition_history)

            print(f"\n  {year}년차:")
            print(f"    활성 주택: {stats['active_count']:,}채")
            print(f"    평균 상태: {stats['mean_condition']:.3f}")
            print(f"    평균 연식: {stats['mean_age']:.1f}년")
            print(f"    누적 멸실: {total_demolished:,}채")

    # 최종 상태
    final_stats = sim.houses.get_condition_stats()
    total_demolished = sum(d['total_count'] for d in sim.demolition_history)
    total_natural = sum(d['natural_count'] for d in sim.demolition_history)
    total_disaster = sum(d['disaster_count'] for d in sim.demolition_history)

    print(f"\n" + "=" * 60)
    print("최종 결과 (5년 후)")
    print("=" * 60)
    print(f"  활성 주택: {final_stats['active_count']:,}채 (초기 대비 {final_stats['active_count'] - initial_stats['active_count']:+,})")
    print(f"  멸실 주택: {final_stats['demolished_count']:,}채")
    print(f"  평균 상태: {final_stats['mean_condition']:.3f} (초기 대비 {final_stats['mean_condition'] - initial_stats['mean_condition']:+.3f})")
    print(f"  평균 연식: {final_stats['mean_age']:.1f}년 (초기 대비 {final_stats['mean_age'] - initial_stats['mean_age']:+.1f})")

    print(f"\n  멸실 통계:")
    print(f"    자연 멸실 (노후): {total_natural:,}채")
    print(f"    재해 멸실: {total_disaster:,}채")
    print(f"    총 멸실: {total_demolished:,}채")

    print("\n테스트 완료!")

if __name__ == "__main__":
    test_depreciation()
