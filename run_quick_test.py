"""빠른 테스트용 시뮬레이션 (축소 규모)"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.realestate import Simulation, Config


def print_agent_heterogeneity(sim: Simulation):
    """에이전트 이질성 통계 출력"""
    print("\n" + "="*70)
    print("에이전트 이질성 (표준편차) - 정규분포를 따르는 개인별 특성")
    print("="*70)

    # 손실 회피 계수 (λ)
    loss_aversion = sim.households.loss_aversion.to_numpy()
    print(f"\n1. 손실회피계수 (λ) [Kahneman & Tversky, 1992]")
    print(f"   평균: {loss_aversion.mean():.3f}, 표준편차: {loss_aversion.std():.3f}")
    print(f"   범위: [{loss_aversion.min():.2f}, {loss_aversion.max():.2f}]")
    print(f"   해석: 손실의 고통 = 이득의 기쁨 × λ (평균 2.5배)")

    # 현재 편향 (β)
    discount_beta = sim.households.discount_beta.to_numpy()
    print(f"\n2. 현재편향 (β) [Laibson, 1997]")
    print(f"   평균: {discount_beta.mean():.3f}, 표준편차: {discount_beta.std():.3f}")
    print(f"   범위: [{discount_beta.min():.2f}, {discount_beta.max():.2f}]")
    print(f"   해석: β < 1이면 미래 가치를 추가 할인 (현재 선호)")

    # 기하 할인율 (δ)
    discount_delta = sim.households.discount_delta.to_numpy()
    print(f"\n3. 기하할인율 (δ)")
    print(f"   평균: {discount_delta.mean():.4f}, 표준편차: {discount_delta.std():.4f}")
    print(f"   범위: [{discount_delta.min():.3f}, {discount_delta.max():.3f}]")
    print(f"   해석: 월간 할인율 (0.99 = 연 ~11% 할인)")

    # FOMO 민감도
    fomo = sim.households.fomo_sensitivity.to_numpy()
    print(f"\n4. FOMO 민감도")
    print(f"   평균: {fomo.mean():.3f}, 표준편차: {fomo.std():.3f}")
    print(f"   범위: [{fomo.min():.2f}, {fomo.max():.2f}]")
    print(f"   해석: 가격 상승 시 매수 욕구 증폭 정도")

    # 군집 성향
    herding = sim.households.herding_tendency.to_numpy()
    print(f"\n5. 군집성향 (Herding)")
    print(f"   평균: {herding.mean():.3f}, 표준편차: {herding.std():.3f}")
    print(f"   범위: [{herding.min():.2f}, {herding.max():.2f}]")
    print(f"   해석: 다른 사람들의 행동을 따라하는 경향")

    # 위험 허용도
    risk = sim.households.risk_tolerance.to_numpy()
    print(f"\n6. 위험허용도")
    print(f"   평균: {risk.mean():.3f}, 표준편차: {risk.std():.3f}")
    print(f"   범위: [{risk.min():.2f}, {risk.max():.2f}]")
    print(f"   해석: 나이가 많을수록 감소")

    # 가격 기대
    expectation = sim.households.price_expectation.to_numpy()
    print(f"\n7. 가격기대")
    print(f"   평균: {expectation.mean():.3f}, 표준편차: {expectation.std():.3f}")
    print(f"   범위: [{expectation.min():.2f}, {expectation.max():.2f}]")
    print(f"   해석: -1(하락 예상) ~ +1(상승 예상)")

    # 에이전트 유형 분포
    agent_type = sim.households.agent_type.to_numpy()
    print(f"\n8. 에이전트 유형 분포")
    print(f"   실수요자: {(agent_type==0).sum()/len(agent_type)*100:.1f}%")
    print(f"   투자자: {(agent_type==1).sum()/len(agent_type)*100:.1f}%")
    print(f"   투기자: {(agent_type==2).sum()/len(agent_type)*100:.1f}%")


def plot_heterogeneity(sim: Simulation):
    """이질성 분포 시각화"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 1. 손실 회피 계수
    data = sim.households.loss_aversion.to_numpy()
    axes[0, 0].hist(data, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'손실회피계수 (λ)\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 2. 현재 편향
    data = sim.households.discount_beta.to_numpy()
    axes[0, 1].hist(data, bins=40, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title(f'현재편향 (β)\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 3. FOMO 민감도
    data = sim.households.fomo_sensitivity.to_numpy()
    axes[0, 2].hist(data, bins=40, color='green', edgecolor='white', alpha=0.7)
    axes[0, 2].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 2].set_title(f'FOMO 민감도\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 4. 군집 성향
    data = sim.households.herding_tendency.to_numpy()
    axes[0, 3].hist(data, bins=40, color='purple', edgecolor='white', alpha=0.7)
    axes[0, 3].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 3].set_title(f'군집성향\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 5. 위험 허용도
    data = sim.households.risk_tolerance.to_numpy()
    axes[1, 0].hist(data, bins=40, color='orange', edgecolor='white', alpha=0.7)
    axes[1, 0].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title(f'위험허용도\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 6. 가격 기대
    data = sim.households.price_expectation.to_numpy()
    axes[1, 1].hist(data, bins=40, color='teal', edgecolor='white', alpha=0.7)
    axes[1, 1].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 1].axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_title(f'가격기대\n평균={data.mean():.2f}, std={data.std():.2f}', fontweight='bold')

    # 7. 나이 분포
    data = sim.households.age.to_numpy()
    axes[1, 2].hist(data, bins=40, color='brown', edgecolor='white', alpha=0.7)
    axes[1, 2].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 2].set_title(f'나이 분포\n평균={data.mean():.1f}, std={data.std():.1f}', fontweight='bold')

    # 8. 소득 분포
    data = sim.households.income.to_numpy()
    axes[1, 3].hist(data, bins=40, color='navy', edgecolor='white', alpha=0.7)
    axes[1, 3].axvline(x=data.mean(), color='red', linestyle='--', linewidth=2)
    axes[1, 3].set_title(f'월소득 (만원)\n평균={data.mean():.0f}, std={data.std():.0f}', fontweight='bold')

    plt.tight_layout()
    plt.savefig('heterogeneity_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n저장됨: heterogeneity_distribution.png")


def run_scenario(name: str, config: Config, steps: int = 36):
    """시나리오 실행"""
    print(f"\n{'='*60}")
    print(f"시나리오: {name}")
    print('='*60)

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def main():
    # 축소 규모 설정
    N_HOUSEHOLDS = 50_000
    N_HOUSES = 25_000
    STEPS = 36  # 3년

    print("="*70)
    print("한국 부동산 ABM 시뮬레이션 - 이질성 및 다중 시나리오 분석")
    print("="*70)
    print(f"\n규모: 가구 {N_HOUSEHOLDS:,}개, 주택 {N_HOUSES:,}채, 기간 {STEPS}개월")

    # 1. 기본 시나리오 + 이질성 분석
    print("\n" + "="*70)
    print("Phase 1: 에이전트 이질성 분석")
    print("="*70)

    config = Config(
        num_households=N_HOUSEHOLDS,
        num_houses=N_HOUSES,
        num_steps=STEPS,
        seed=42
    )
    sim = Simulation(config, arch="vulkan")
    sim.initialize()

    print_agent_heterogeneity(sim)
    plot_heterogeneity(sim)

    # 시나리오 결과 저장
    all_results = {}

    # 2. 기본 시나리오 실행
    config_baseline = Config(num_households=N_HOUSEHOLDS, num_houses=N_HOUSES, seed=42)
    all_results['기본'] = run_scenario('기본', config_baseline, STEPS)

    # 3. 금리 인상 시나리오
    config_rate_up = Config(num_households=N_HOUSEHOLDS, num_houses=N_HOUSES, seed=42)
    config_rate_up.policy.interest_rate = 0.055
    all_results['금리인상(5.5%)'] = run_scenario('금리인상 (5.5%)', config_rate_up, STEPS)

    # 4. 금리 인하 시나리오
    config_rate_down = Config(num_households=N_HOUSEHOLDS, num_houses=N_HOUSES, seed=42)
    config_rate_down.policy.interest_rate = 0.02
    all_results['금리인하(2.0%)'] = run_scenario('금리인하 (2.0%)', config_rate_down, STEPS)

    # 5. 규제 강화 시나리오
    config_strict = Config(num_households=N_HOUSEHOLDS, num_houses=N_HOUSES, seed=42)
    config_strict.policy.ltv_1house = 0.30
    config_strict.policy.ltv_2house = 0.0
    config_strict.policy.dti_limit = 0.30
    all_results['규제강화'] = run_scenario('규제강화 (LTV 30%)', config_strict, STEPS)

    # 6. 규제 완화 시나리오
    config_relaxed = Config(num_households=N_HOUSEHOLDS, num_houses=N_HOUSES, seed=42)
    config_relaxed.policy.ltv_1house = 0.70
    config_relaxed.policy.ltv_2house = 0.50
    config_relaxed.policy.dti_limit = 0.50
    all_results['규제완화'] = run_scenario('규제완화 (LTV 70%)', config_relaxed, STEPS)

    # 결과 시각화
    print("\n" + "="*70)
    print("Phase 2: 시나리오 비교 시각화")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. 강남 가격
    ax = axes[0, 0]
    for (name, results), color in zip(all_results.items(), colors):
        prices = [p[0]/10000 for p in results['price_history']]  # 억 단위
        ax.plot(prices, label=name, color=color, linewidth=2)
    ax.set_title('강남3구 가격 추이', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 전국 평균
    ax = axes[0, 1]
    for (name, results), color in zip(all_results.items(), colors):
        avg = [s['avg_price']/10000 for s in results['stats_history']]
        ax.plot(avg, label=name, color=color, linewidth=2)
    ax.set_title('전국 평균 가격', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 거래량
    ax = axes[0, 2]
    for (name, results), color in zip(all_results.items(), colors):
        trans = [s['transaction_total'] for s in results['stats_history']]
        ax.plot(trans, label=name, color=color, linewidth=2)
    ax.set_title('월간 거래량', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('건수')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 자가율
    ax = axes[1, 0]
    for (name, results), color in zip(all_results.items(), colors):
        rate = [s['homeowner_rate']*100 for s in results['stats_history']]
        ax.plot(rate, label=name, color=color, linewidth=2)
    ax.set_title('자가 보유율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. 다주택자 비율
    ax = axes[1, 1]
    for (name, results), color in zip(all_results.items(), colors):
        rate = [s['multi_owner_rate']*100 for s in results['stats_history']]
        ax.plot(rate, label=name, color=color, linewidth=2)
    ax.set_title('다주택자 비율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('%')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. 수요/공급 비율
    ax = axes[1, 2]
    for (name, results), color in zip(all_results.items(), colors):
        ds = [s['demand_supply_ratio'] for s in results['stats_history']]
        ax.plot(ds, label=name, color=color, linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('수요/공급 비율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("저장됨: scenario_comparison.png")

    # 요약 테이블
    print("\n" + "="*70)
    print("시나리오별 결과 요약 (3년 후)")
    print("="*70)
    print(f"{'시나리오':<18} {'강남(억)':>10} {'전국(억)':>10} {'변화율':>10} {'자가율':>8} {'다주택':>8}")
    print("-"*70)

    for name, results in all_results.items():
        s_start = results['stats_history'][0]
        s_end = results['stats_history'][-1]
        change = (s_end['avg_price'] - s_start['avg_price']) / s_start['avg_price'] * 100
        print(f"{name:<18} "
              f"{s_end['price_gangnam']/10000:>10.1f} "
              f"{s_end['avg_price']/10000:>10.1f} "
              f"{change:>+9.1f}% "
              f"{s_end['homeowner_rate']*100:>7.1f}% "
              f"{s_end['multi_owner_rate']*100:>7.1f}%")

    print("\n완료!")


if __name__ == "__main__":
    main()
