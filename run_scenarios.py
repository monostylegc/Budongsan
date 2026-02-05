"""다양한 시나리오 시뮬레이션 실행

에이전트 성향의 이질성(표준편차)을 반영한 시뮬레이션
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.realestate import Simulation, Config


def print_agent_heterogeneity(sim: Simulation):
    """에이전트 이질성 통계 출력"""
    print("\n=== 에이전트 이질성 (표준편차) 확인 ===")

    # 손실 회피 계수
    loss_aversion = sim.households.loss_aversion.to_numpy()
    print(f"손실회피계수 (λ): mean={loss_aversion.mean():.3f}, std={loss_aversion.std():.3f}, "
          f"range=[{loss_aversion.min():.2f}, {loss_aversion.max():.2f}]")

    # 현재 편향 (β)
    discount_beta = sim.households.discount_beta.to_numpy()
    print(f"현재편향 (β): mean={discount_beta.mean():.3f}, std={discount_beta.std():.3f}, "
          f"range=[{discount_beta.min():.2f}, {discount_beta.max():.2f}]")

    # 기하 할인율 (δ)
    discount_delta = sim.households.discount_delta.to_numpy()
    print(f"기하할인율 (δ): mean={discount_delta.mean():.4f}, std={discount_delta.std():.4f}, "
          f"range=[{discount_delta.min():.3f}, {discount_delta.max():.3f}]")

    # FOMO 민감도
    fomo = sim.households.fomo_sensitivity.to_numpy()
    print(f"FOMO 민감도: mean={fomo.mean():.3f}, std={fomo.std():.3f}, "
          f"range=[{fomo.min():.2f}, {fomo.max():.2f}]")

    # 군집 성향
    herding = sim.households.herding_tendency.to_numpy()
    print(f"군집성향: mean={herding.mean():.3f}, std={herding.std():.3f}, "
          f"range=[{herding.min():.2f}, {herding.max():.2f}]")

    # 위험 허용도
    risk = sim.households.risk_tolerance.to_numpy()
    print(f"위험허용도: mean={risk.mean():.3f}, std={risk.std():.3f}, "
          f"range=[{risk.min():.2f}, {risk.max():.2f}]")

    # 가격 기대
    expectation = sim.households.price_expectation.to_numpy()
    print(f"가격기대: mean={expectation.mean():.3f}, std={expectation.std():.3f}, "
          f"range=[{expectation.min():.2f}, {expectation.max():.2f}]")

    # 에이전트 유형 분포
    agent_type = sim.households.agent_type.to_numpy()
    print(f"\n에이전트 유형: 실수요자 {(agent_type==0).sum()/len(agent_type)*100:.1f}%, "
          f"투자자 {(agent_type==1).sum()/len(agent_type)*100:.1f}%, "
          f"투기자 {(agent_type==2).sum()/len(agent_type)*100:.1f}%")


def run_baseline_scenario(steps: int = 60) -> Dict[str, Any]:
    """기본 시나리오 실행"""
    print("\n" + "="*60)
    print("시나리오 1: 기본 (Baseline)")
    print("="*60)

    config = Config(
        num_households=100_000,  # 테스트용 축소
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )

    sim = Simulation(config, arch="vulkan")
    sim.initialize()

    # 이질성 확인
    print_agent_heterogeneity(sim)

    # 실행
    results = sim.run(steps=steps, verbose=True)
    return results


def run_rate_hike_scenario(steps: int = 60) -> Dict[str, Any]:
    """금리 인상 시나리오 (3.5% → 5.5%)"""
    print("\n" + "="*60)
    print("시나리오 2: 금리 인상 (3.5% → 5.5%)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    config.policy.interest_rate = 0.055  # 5.5%
    config.macro.neutral_real_rate = 0.035  # 중립금리 상승

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def run_rate_cut_scenario(steps: int = 60) -> Dict[str, Any]:
    """금리 인하 시나리오 (3.5% → 2.0%)"""
    print("\n" + "="*60)
    print("시나리오 3: 금리 인하 (3.5% → 2.0%)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    config.policy.interest_rate = 0.02  # 2.0%
    config.macro.neutral_real_rate = 0.01

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def run_regulation_strict_scenario(steps: int = 60) -> Dict[str, Any]:
    """규제 강화 시나리오 (LTV/DTI 하향)"""
    print("\n" + "="*60)
    print("시나리오 4: 규제 강화 (LTV 30%, DTI 30%)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    config.policy.ltv_1house = 0.30  # 50% → 30%
    config.policy.ltv_2house = 0.0   # 2주택 대출 금지
    config.policy.dti_limit = 0.30   # 40% → 30%
    config.policy.jongbu_rate = 0.03  # 종부세 강화

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def run_regulation_relaxed_scenario(steps: int = 60) -> Dict[str, Any]:
    """규제 완화 시나리오 (LTV/DTI 상향)"""
    print("\n" + "="*60)
    print("시나리오 5: 규제 완화 (LTV 70%, DTI 50%)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    config.policy.ltv_1house = 0.70  # 50% → 70%
    config.policy.ltv_2house = 0.50  # 30% → 50%
    config.policy.dti_limit = 0.50   # 40% → 50%
    config.policy.jongbu_rate = 0.01  # 종부세 완화

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def run_supply_expansion_scenario(steps: int = 60) -> Dict[str, Any]:
    """공급 확대 시나리오"""
    print("\n" + "="*60)
    print("시나리오 6: 공급 확대 (탄력성 2배)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    # 공급 탄력성 2배
    config.supply.elasticity_gangnam = 0.6
    config.supply.elasticity_seoul = 1.0
    config.supply.elasticity_gyeonggi = 3.0
    config.supply.elasticity_local = 4.0
    config.supply.base_supply_rate = 0.002  # 기본 공급률 2배

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def run_high_fomo_scenario(steps: int = 60) -> Dict[str, Any]:
    """FOMO 과열 시나리오"""
    print("\n" + "="*60)
    print("시나리오 7: FOMO 과열 (민감도 증가)")
    print("="*60)

    config = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=steps,
        seed=42
    )
    config.behavioral.fomo_intensity = 100.0  # 50 → 100
    config.behavioral.herding_intensity = 20.0  # 10 → 20
    config.behavioral.fomo_trigger_threshold = 0.03  # 5% → 3%

    sim = Simulation(config, arch="vulkan")
    results = sim.run(steps=steps, verbose=True)
    return results


def plot_scenario_comparison(all_results: Dict[str, Dict], save_path: str = "scenario_comparison.png"):
    """시나리오 비교 그래프"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. 강남 가격 추이
    ax = axes[0, 0]
    for (name, results), color in zip(all_results.items(), colors):
        prices = results['price_history']
        if len(prices) > 0:
            gangnam_prices = [p[0] for p in prices]
            ax.plot(gangnam_prices, label=name, color=color, linewidth=2)
    ax.set_title('강남3구 가격 추이', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (만원)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 전국 평균 가격
    ax = axes[0, 1]
    for (name, results), color in zip(all_results.items(), colors):
        stats = results['stats_history']
        avg_prices = [s['avg_price'] for s in stats]
        ax.plot(avg_prices, label=name, color=color, linewidth=2)
    ax.set_title('전국 평균 가격', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (만원)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 거래량
    ax = axes[0, 2]
    for (name, results), color in zip(all_results.items(), colors):
        stats = results['stats_history']
        transactions = [s['transaction_total'] for s in stats]
        # 6개월 이동평균
        if len(transactions) >= 6:
            transactions_ma = np.convolve(transactions, np.ones(6)/6, mode='valid')
            ax.plot(transactions_ma, label=name, color=color, linewidth=2)
    ax.set_title('월간 거래량 (6개월 이동평균)', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('거래 건수')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 자가 보유율
    ax = axes[1, 0]
    for (name, results), color in zip(all_results.items(), colors):
        stats = results['stats_history']
        homeowner = [s['homeowner_rate'] * 100 for s in stats]
        ax.plot(homeowner, label=name, color=color, linewidth=2)
    ax.set_title('자가 보유율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율 (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. 다주택자 비율
    ax = axes[1, 1]
    for (name, results), color in zip(all_results.items(), colors):
        stats = results['stats_history']
        multi = [s['multi_owner_rate'] * 100 for s in stats]
        ax.plot(multi, label=name, color=color, linewidth=2)
    ax.set_title('다주택자 비율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율 (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 수요/공급 비율
    ax = axes[1, 2]
    for (name, results), color in zip(all_results.items(), colors):
        stats = results['stats_history']
        ds_ratio = [s['demand_supply_ratio'] for s in stats]
        ax.plot(ds_ratio, label=name, color=color, linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='균형선')
    ax.set_title('수요/공급 비율', fontsize=12, fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n그래프 저장: {save_path}")


def plot_macro_comparison(all_results: Dict[str, Dict], save_path: str = "macro_comparison.png"):
    """거시경제 지표 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, results), color in zip(all_results.items(), colors):
        macro = results.get('macro_history', [])
        if not macro:
            continue

        # 1. 기준금리
        policy_rates = [m['policy_rate'] * 100 for m in macro]
        axes[0, 0].plot(policy_rates, label=name, color=color, linewidth=2)

        # 2. 주담대 금리
        mortgage_rates = [m['mortgage_rate'] * 100 for m in macro]
        axes[0, 1].plot(mortgage_rates, label=name, color=color, linewidth=2)

        # 3. GDP 성장률
        gdp = [m['gdp_growth'] * 100 for m in macro]
        axes[1, 0].plot(gdp, label=name, color=color, linewidth=2)

        # 4. 인플레이션
        inflation = [m['inflation'] * 100 for m in macro]
        axes[1, 1].plot(inflation, label=name, color=color, linewidth=2)

    axes[0, 0].set_title('기준금리', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('%')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('주담대 금리', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('GDP 성장률 (연율)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('인플레이션', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"거시경제 그래프 저장: {save_path}")


def plot_heterogeneity_distribution(sim: Simulation, save_path: str = "heterogeneity_dist.png"):
    """에이전트 이질성 분포 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 손실 회피 계수 분포
    loss_aversion = sim.households.loss_aversion.to_numpy()
    axes[0, 0].hist(loss_aversion, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0, 0].axvline(x=loss_aversion.mean(), color='red', linestyle='--', label=f'평균: {loss_aversion.mean():.2f}')
    axes[0, 0].set_title('손실회피계수 (λ) 분포', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('λ')
    axes[0, 0].legend()

    # 2. 현재 편향 (β) 분포
    discount_beta = sim.households.discount_beta.to_numpy()
    axes[0, 1].hist(discount_beta, bins=50, color='coral', edgecolor='white', alpha=0.7)
    axes[0, 1].axvline(x=discount_beta.mean(), color='red', linestyle='--', label=f'평균: {discount_beta.mean():.2f}')
    axes[0, 1].set_title('현재편향 (β) 분포', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('β')
    axes[0, 1].legend()

    # 3. FOMO 민감도 분포
    fomo = sim.households.fomo_sensitivity.to_numpy()
    axes[0, 2].hist(fomo, bins=50, color='green', edgecolor='white', alpha=0.7)
    axes[0, 2].axvline(x=fomo.mean(), color='red', linestyle='--', label=f'평균: {fomo.mean():.2f}')
    axes[0, 2].set_title('FOMO 민감도 분포', fontsize=11, fontweight='bold')
    axes[0, 2].set_xlabel('민감도')
    axes[0, 2].legend()

    # 4. 군집 성향 분포
    herding = sim.households.herding_tendency.to_numpy()
    axes[1, 0].hist(herding, bins=50, color='purple', edgecolor='white', alpha=0.7)
    axes[1, 0].axvline(x=herding.mean(), color='red', linestyle='--', label=f'평균: {herding.mean():.2f}')
    axes[1, 0].set_title('군집성향 분포', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('성향')
    axes[1, 0].legend()

    # 5. 위험 허용도 분포
    risk = sim.households.risk_tolerance.to_numpy()
    axes[1, 1].hist(risk, bins=50, color='orange', edgecolor='white', alpha=0.7)
    axes[1, 1].axvline(x=risk.mean(), color='red', linestyle='--', label=f'평균: {risk.mean():.2f}')
    axes[1, 1].set_title('위험허용도 분포', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('허용도')
    axes[1, 1].legend()

    # 6. 가격 기대 분포
    expectation = sim.households.price_expectation.to_numpy()
    axes[1, 2].hist(expectation, bins=50, color='teal', edgecolor='white', alpha=0.7)
    axes[1, 2].axvline(x=expectation.mean(), color='red', linestyle='--', label=f'평균: {expectation.mean():.2f}')
    axes[1, 2].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_title('가격기대 분포', fontsize=11, fontweight='bold')
    axes[1, 2].set_xlabel('기대 (-1 ~ 1)')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"이질성 분포 그래프 저장: {save_path}")


def print_summary_statistics(all_results: Dict[str, Dict]):
    """시나리오별 요약 통계"""
    print("\n" + "="*80)
    print("시나리오별 요약 통계 (최종 시점)")
    print("="*80)

    headers = ["시나리오", "강남가격(억)", "전국평균(억)", "거래량", "자가율(%)", "다주택(%)", "수급비율"]

    # 헤더 출력
    print(f"{'시나리오':<20} {'강남(억)':>10} {'전국(억)':>10} {'거래량':>8} {'자가율':>8} {'다주택':>8} {'수급비':>8}")
    print("-" * 80)

    for name, results in all_results.items():
        stats = results['stats_history'][-1]
        print(f"{name:<20} "
              f"{stats['price_gangnam']/10000:>10.1f} "
              f"{stats['avg_price']/10000:>10.1f} "
              f"{stats['transaction_total']:>8,} "
              f"{stats['homeowner_rate']*100:>7.1f}% "
              f"{stats['multi_owner_rate']*100:>7.1f}% "
              f"{stats['demand_supply_ratio']:>8.2f}")

    # 가격 변화율 계산
    print("\n" + "="*80)
    print("시나리오별 가격 변화율 (시작 → 종료)")
    print("="*80)

    print(f"{'시나리오':<20} {'강남':>12} {'경기':>12} {'지방':>12} {'전국평균':>12}")
    print("-" * 80)

    for name, results in all_results.items():
        stats_start = results['stats_history'][0]
        stats_end = results['stats_history'][-1]

        gangnam_change = (stats_end['price_gangnam'] - stats_start['price_gangnam']) / stats_start['price_gangnam'] * 100
        gyeonggi_change = (stats_end['price_gyeonggi'] - stats_start['price_gyeonggi']) / stats_start['price_gyeonggi'] * 100
        jibang_change = (stats_end['price_jibang'] - stats_start['price_jibang']) / stats_start['price_jibang'] * 100
        avg_change = (stats_end['avg_price'] - stats_start['avg_price']) / stats_start['avg_price'] * 100

        print(f"{name:<20} "
              f"{gangnam_change:>+11.1f}% "
              f"{gyeonggi_change:>+11.1f}% "
              f"{jibang_change:>+11.1f}% "
              f"{avg_change:>+11.1f}%")


def main():
    """메인 실행"""
    print("="*80)
    print("한국 부동산 ABM 시뮬레이션 - 다중 시나리오 분석")
    print("="*80)

    # 시뮬레이션 기간 설정 (5년 = 60개월)
    STEPS = 60

    all_results = {}

    # 1. 기본 시나리오 (이질성 확인용)
    results = run_baseline_scenario(steps=STEPS)
    all_results['기본'] = results

    # 이질성 분포 시각화를 위해 별도 시뮬레이션 생성
    config = Config(num_households=100_000, num_houses=50_000, seed=42)
    sim_for_plot = Simulation(config, arch="vulkan")
    sim_for_plot.initialize()
    plot_heterogeneity_distribution(sim_for_plot, "heterogeneity_dist.png")

    # 2. 금리 인상 시나리오
    all_results['금리인상'] = run_rate_hike_scenario(steps=STEPS)

    # 3. 금리 인하 시나리오
    all_results['금리인하'] = run_rate_cut_scenario(steps=STEPS)

    # 4. 규제 강화 시나리오
    all_results['규제강화'] = run_regulation_strict_scenario(steps=STEPS)

    # 5. 규제 완화 시나리오
    all_results['규제완화'] = run_regulation_relaxed_scenario(steps=STEPS)

    # 6. 공급 확대 시나리오
    all_results['공급확대'] = run_supply_expansion_scenario(steps=STEPS)

    # 7. FOMO 과열 시나리오
    all_results['FOMO과열'] = run_high_fomo_scenario(steps=STEPS)

    # 결과 시각화
    plot_scenario_comparison(all_results, "scenario_comparison.png")
    plot_macro_comparison(all_results, "macro_comparison.png")

    # 요약 통계
    print_summary_statistics(all_results)

    print("\n" + "="*80)
    print("모든 시나리오 실행 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - heterogeneity_dist.png: 에이전트 이질성 분포")
    print("  - scenario_comparison.png: 시나리오 비교 그래프")
    print("  - macro_comparison.png: 거시경제 지표 비교")


if __name__ == "__main__":
    main()
