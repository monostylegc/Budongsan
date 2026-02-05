"""전체 기능 통합 테스트 (노후화/멸실 포함)"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.realestate import Simulation, Config

def test_full_simulation():
    print("="*70)
    print("한국 부동산 ABM - 전체 기능 통합 테스트")
    print("="*70)

    # 작은 규모로 빠른 테스트
    config = Config(
        num_households=10000,
        num_houses=6000,
        num_steps=24,  # 2년
        seed=42
    )

    sim = Simulation(config, arch="vulkan")
    sim.initialize()

    # 시뮬레이션 실행
    print("\n시뮬레이션 실행 (24개월)...")
    for month in range(24):
        sim.step()

        if (month + 1) % 6 == 0:
            stats = sim.stats_history[-1]
            macro = sim.macro_history[-1]
            demo = sim.demolition_history[-1]

            print(f"\n{month+1}개월 결과:")
            print(f"  가격: 강남 {stats['price_gangnam']/10000:.1f}억, 전국 {stats['avg_price']/10000:.1f}억")
            print(f"  거래: {stats['transaction_total']}건, 수요/공급: {stats['demand_supply_ratio']:.2f}")
            print(f"  거시: 금리 {macro['policy_rate']*100:.2f}%, M2 {macro['m2_growth']*100:.1f}%")
            print(f"  주택: 활성 {stats['active_houses']:,}채, 멸실 {demo['total_count']}채")
            print(f"  상태: 평균 {stats['mean_building_condition']:.3f}")

    # 최종 결과
    print("\n" + "="*70)
    print("최종 결과 (2년 후)")
    print("="*70)

    final_stats = sim.stats_history[-1]
    initial_stats = sim.stats_history[0]

    price_change = (final_stats['avg_price'] - initial_stats['avg_price']) / initial_stats['avg_price'] * 100

    print(f"\n[가격]")
    print(f"  전국 평균: {final_stats['avg_price']/10000:.1f}억 ({price_change:+.1f}%)")
    print(f"  강남: {final_stats['price_gangnam']/10000:.1f}억")

    print(f"\n[시장]")
    print(f"  자가 보유율: {final_stats['homeowner_rate']*100:.1f}%")
    print(f"  다주택자율: {final_stats['multi_owner_rate']*100:.1f}%")
    print(f"  수요/공급 비율: {final_stats['demand_supply_ratio']:.2f}")

    print(f"\n[거시경제]")
    final_macro = sim.macro_history[-1]
    print(f"  기준금리: {final_macro['policy_rate']*100:.2f}%")
    print(f"  주담대금리: {final_macro['mortgage_rate']*100:.2f}%")
    print(f"  M2 증가율: {final_macro['m2_growth']*100:.1f}%")
    print(f"  인플레이션: {final_macro['inflation']*100:.2f}%")

    print(f"\n[노후화/멸실]")
    total_demolished = sum(d['total_count'] for d in sim.demolition_history)
    total_natural = sum(d['natural_count'] for d in sim.demolition_history)
    total_disaster = sum(d['disaster_count'] for d in sim.demolition_history)
    print(f"  활성 주택: {final_stats['active_houses']:,}채")
    print(f"  총 멸실: {total_demolished}채 (자연 {total_natural}, 재해 {total_disaster})")
    print(f"  평균 건물 상태: {final_stats['mean_building_condition']:.3f}")
    print(f"  30년 이상 노후: {final_stats['old_buildings_30y']}채")

    # 간단한 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('ABM 시뮬레이션 결과 (2년)', fontweight='bold')

    # 가격 추이
    prices = [s['avg_price']/10000 for s in sim.stats_history]
    axes[0, 0].plot(prices, linewidth=2, color='blue')
    axes[0, 0].set_title('전국 평균 가격')
    axes[0, 0].set_ylabel('억원')
    axes[0, 0].grid(True, alpha=0.3)

    # 거래량
    trans = [s['transaction_total'] for s in sim.stats_history]
    axes[0, 1].plot(trans, linewidth=2, color='green')
    axes[0, 1].set_title('월간 거래량')
    axes[0, 1].set_ylabel('건수')
    axes[0, 1].grid(True, alpha=0.3)

    # 건물 상태
    condition = [s['mean_building_condition'] for s in sim.stats_history]
    axes[1, 0].plot(condition, linewidth=2, color='orange')
    axes[1, 0].set_title('평균 건물 상태')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('상태 (0~1)')
    axes[1, 0].grid(True, alpha=0.3)

    # 활성 주택
    active = [s['active_houses'] for s in sim.stats_history]
    axes[1, 1].plot(active, linewidth=2, color='red')
    axes[1, 1].set_title('활성 주택 수')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('채')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_all_features.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n저장됨: test_all_features.png")
    print("\n테스트 완료!")


if __name__ == "__main__":
    test_full_simulation()
