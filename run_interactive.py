"""인터랙티브 파라미터 조절 시뮬레이션

공급량, 통화량, 금리, 규제 등 다양한 파라미터를 조절하며 시뮬레이션
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.realestate import Simulation, Config


@dataclass
class ScenarioParams:
    """시나리오 파라미터"""
    name: str

    # 공급 관련
    supply_elasticity_mult: float = 1.0  # 공급 탄력성 배수
    base_supply_rate: float = 0.001      # 기본 공급률

    # 통화량 관련
    m2_growth_target: float = 0.08       # M2 증가율 목표 (연율)

    # 금리 관련
    interest_rate: float = 0.035         # 기준금리

    # 규제 관련
    ltv_1house: float = 0.50
    ltv_2house: float = 0.30
    dti_limit: float = 0.40

    # 행동경제학
    fomo_intensity: float = 50.0
    loss_aversion_mean: float = 2.5

    # 노후화/멸실 관련
    depreciation_rate: float = 0.003     # 월간 감가상각률 (기본 0.3%)
    disaster_rate: float = 0.0001        # 재해 멸실률 (기본 0.01%)


def create_config_from_params(params: ScenarioParams, n_households: int = 50000) -> Config:
    """파라미터로부터 Config 생성"""
    config = Config(
        num_households=n_households,
        num_houses=n_households // 2,
        seed=42
    )

    # 공급 설정
    config.supply.base_supply_rate = params.base_supply_rate
    config.supply.elasticity_gangnam *= params.supply_elasticity_mult
    config.supply.elasticity_seoul *= params.supply_elasticity_mult
    config.supply.elasticity_gyeonggi *= params.supply_elasticity_mult
    config.supply.elasticity_local *= params.supply_elasticity_mult

    # 금리 설정
    config.policy.interest_rate = params.interest_rate

    # 규제 설정
    config.policy.ltv_1house = params.ltv_1house
    config.policy.ltv_2house = params.ltv_2house
    config.policy.dti_limit = params.dti_limit

    # 행동경제학 설정
    config.behavioral.fomo_intensity = params.fomo_intensity
    config.behavioral.loss_aversion_mean = params.loss_aversion_mean

    return config


def apply_depreciation_params(sim, params: ScenarioParams):
    """노후화/멸실 파라미터 적용"""
    sim.houses.depreciation_rate = params.depreciation_rate
    sim.houses.disaster_rate = params.disaster_rate


def run_simulation_with_params(params: ScenarioParams, steps: int = 36) -> Dict[str, Any]:
    """파라미터로 시뮬레이션 실행"""
    print(f"\n{'='*60}")
    print(f"시나리오: {params.name}")
    print(f"  - 공급탄력성: {params.supply_elasticity_mult:.1f}x")
    print(f"  - M2 증가율: {params.m2_growth_target*100:.1f}%")
    print(f"  - 기준금리: {params.interest_rate*100:.1f}%")
    print(f"  - LTV: {params.ltv_1house*100:.0f}%/{params.ltv_2house*100:.0f}%")
    print(f"  - 감가상각: {params.depreciation_rate*100:.2f}%/월")
    print('='*60)

    config = create_config_from_params(params)
    sim = Simulation(config, arch="vulkan")

    # M2 증가율 설정
    sim.macro.set_m2_growth_target(params.m2_growth_target)

    # 노후화/멸실 파라미터 적용
    apply_depreciation_params(sim, params)

    results = sim.run(steps=steps, verbose=True)
    results['params'] = params
    return results


# ============================================================================
# 시나리오 정의
# ============================================================================

def get_supply_scenarios() -> List[ScenarioParams]:
    """공급량 시나리오"""
    return [
        ScenarioParams(name="공급부족", supply_elasticity_mult=0.3, base_supply_rate=0.0005),
        ScenarioParams(name="공급기본", supply_elasticity_mult=1.0, base_supply_rate=0.001),
        ScenarioParams(name="공급확대", supply_elasticity_mult=2.0, base_supply_rate=0.002),
        ScenarioParams(name="공급대량", supply_elasticity_mult=3.0, base_supply_rate=0.003),
    ]


def get_m2_scenarios() -> List[ScenarioParams]:
    """통화량 시나리오"""
    return [
        ScenarioParams(name="긴축(M2 2%)", m2_growth_target=0.02),
        ScenarioParams(name="기본(M2 8%)", m2_growth_target=0.08),
        ScenarioParams(name="완화(M2 12%)", m2_growth_target=0.12),
        ScenarioParams(name="양적완화(M2 20%)", m2_growth_target=0.20),
    ]


def get_interest_scenarios() -> List[ScenarioParams]:
    """금리 시나리오"""
    return [
        ScenarioParams(name="초저금리(1.5%)", interest_rate=0.015),
        ScenarioParams(name="저금리(2.5%)", interest_rate=0.025),
        ScenarioParams(name="기본(3.5%)", interest_rate=0.035),
        ScenarioParams(name="고금리(5.5%)", interest_rate=0.055),
    ]


def get_regulation_scenarios() -> List[ScenarioParams]:
    """규제 시나리오"""
    return [
        ScenarioParams(name="규제강화", ltv_1house=0.30, ltv_2house=0.0, dti_limit=0.30),
        ScenarioParams(name="규제기본", ltv_1house=0.50, ltv_2house=0.30, dti_limit=0.40),
        ScenarioParams(name="규제완화", ltv_1house=0.70, ltv_2house=0.50, dti_limit=0.50),
        ScenarioParams(name="규제해제", ltv_1house=0.80, ltv_2house=0.70, dti_limit=0.60),
    ]


def get_depreciation_scenarios() -> List[ScenarioParams]:
    """노후화/멸실 시나리오"""
    return [
        ScenarioParams(name="멸실없음", depreciation_rate=0.0, disaster_rate=0.0),
        ScenarioParams(name="느린노후화", depreciation_rate=0.002, disaster_rate=0.00005),
        ScenarioParams(name="기본노후화", depreciation_rate=0.003, disaster_rate=0.0001),
        ScenarioParams(name="빠른노후화", depreciation_rate=0.005, disaster_rate=0.0002),
    ]


def get_combined_scenarios() -> List[ScenarioParams]:
    """복합 시나리오"""
    return [
        # 2008 금융위기형: 고금리 + 긴축 + 공급과잉
        ScenarioParams(
            name="금융위기형",
            supply_elasticity_mult=2.0,
            m2_growth_target=0.02,
            interest_rate=0.055,
            ltv_1house=0.40, ltv_2house=0.20
        ),
        # 2020 코로나형: 저금리 + 양적완화 + 공급부족
        ScenarioParams(
            name="양적완화+공급부족",
            supply_elasticity_mult=0.5,
            m2_growth_target=0.15,
            interest_rate=0.015,
            ltv_1house=0.60, ltv_2house=0.40
        ),
        # 이상적 안정: 적정 공급 + 중립 통화정책
        ScenarioParams(
            name="균형정책",
            supply_elasticity_mult=1.5,
            m2_growth_target=0.06,
            interest_rate=0.03,
            ltv_1house=0.50, ltv_2house=0.30
        ),
        # 극단적 과열: 저금리 + 양적완화 + 공급부족 + 규제완화
        ScenarioParams(
            name="버블형성",
            supply_elasticity_mult=0.3,
            m2_growth_target=0.20,
            interest_rate=0.01,
            ltv_1house=0.80, ltv_2house=0.60,
            fomo_intensity=100.0
        ),
    ]


# ============================================================================
# 시각화
# ============================================================================

def plot_scenario_results(all_results: Dict[str, Dict], title: str, save_path: str):
    """시나리오 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # 1. 강남 가격
    ax = axes[0, 0]
    for (name, results), color in zip(all_results.items(), colors):
        prices = [p[0]/10000 for p in results['price_history']]
        ax.plot(prices, label=name, color=color, linewidth=2)
    ax.set_title('강남3구 가격', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 전국 평균
    ax = axes[0, 1]
    for (name, results), color in zip(all_results.items(), colors):
        avg = [s['avg_price']/10000 for s in results['stats_history']]
        ax.plot(avg, label=name, color=color, linewidth=2)
    ax.set_title('전국 평균 가격', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. 거래량
    ax = axes[0, 2]
    for (name, results), color in zip(all_results.items(), colors):
        trans = [s['transaction_total'] for s in results['stats_history']]
        ax.plot(trans, label=name, color=color, linewidth=2)
    ax.set_title('월간 거래량', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('건수')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 수요/공급 비율
    ax = axes[1, 0]
    for (name, results), color in zip(all_results.items(), colors):
        ds = [s['demand_supply_ratio'] for s in results['stats_history']]
        ax.plot(ds, label=name, color=color, linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('수요/공급 비율', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. M2/유동성 (거시경제 데이터가 있는 경우)
    ax = axes[1, 1]
    for (name, results), color in zip(all_results.items(), colors):
        macro = results.get('macro_history', [])
        if macro and 'm2_level' in macro[0]:
            m2 = [m['m2_level'] for m in macro]
            ax.plot(m2, label=name, color=color, linewidth=2)
    ax.set_title('M2 수준 (기준=1.0)', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('M2 수준')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. 자가율
    ax = axes[1, 2]
    for (name, results), color in zip(all_results.items(), colors):
        rate = [s['homeowner_rate']*100 for s in results['stats_history']]
        ax.plot(rate, label=name, color=color, linewidth=2)
    ax.set_title('자가 보유율', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장됨: {save_path}")


def plot_macro_details(all_results: Dict[str, Dict], title: str, save_path: str):
    """거시경제 상세 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, results), color in zip(all_results.items(), colors):
        macro = results.get('macro_history', [])
        if not macro:
            continue

        # 1. 기준금리
        rates = [m['policy_rate']*100 for m in macro]
        axes[0, 0].plot(rates, label=name, color=color, linewidth=2)

        # 2. M2 증가율
        if 'm2_growth' in macro[0]:
            m2g = [m['m2_growth']*100 for m in macro]
            axes[0, 1].plot(m2g, label=name, color=color, linewidth=2)

        # 3. 인플레이션
        inf = [m['inflation']*100 for m in macro]
        axes[0, 2].plot(inf, label=name, color=color, linewidth=2)

        # 4. GDP 성장률
        gdp = [m['gdp_growth']*100 for m in macro]
        axes[1, 0].plot(gdp, label=name, color=color, linewidth=2)

        # 5. 유동성 지수
        if 'liquidity_index' in macro[0]:
            liq = [m['liquidity_index'] for m in macro]
            axes[1, 1].plot(liq, label=name, color=color, linewidth=2)

        # 6. 주담대 금리
        mort = [m['mortgage_rate']*100 for m in macro]
        axes[1, 2].plot(mort, label=name, color=color, linewidth=2)

    axes[0, 0].set_title('기준금리 (%)', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('M2 증가율 (%, 연율)', fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title('인플레이션 (%)', fontweight='bold')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_title('GDP 성장률 (%, 연율)', fontweight='bold')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('유동성 지수', fontweight='bold')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_title('주담대 금리 (%)', fontweight='bold')
    axes[1, 2].set_xlabel('월')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장됨: {save_path}")


def plot_depreciation_results(all_results: Dict[str, Dict], title: str, save_path: str):
    """노후화/멸실 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, results), color in zip(all_results.items(), colors):
        stats = results.get('stats_history', [])
        demolition = results.get('demolition_history', [])

        if not stats:
            continue

        # 1. 가격 추이
        prices = [s['avg_price']/10000 for s in stats]
        axes[0, 0].plot(prices, label=name, color=color, linewidth=2)

        # 2. 활성 주택 수
        if 'active_houses' in stats[0]:
            active = [s['active_houses'] for s in stats]
            axes[0, 1].plot(active, label=name, color=color, linewidth=2)

        # 3. 건물 평균 상태
        if 'mean_building_condition' in stats[0]:
            condition = [s['mean_building_condition'] for s in stats]
            axes[0, 2].plot(condition, label=name, color=color, linewidth=2)

        # 4. 누적 멸실
        if demolition:
            cumulative = np.cumsum([d['total_count'] for d in demolition])
            axes[1, 0].plot(cumulative, label=name, color=color, linewidth=2)

        # 5. 월간 멸실
        if demolition:
            monthly = [d['total_count'] for d in demolition]
            axes[1, 1].plot(monthly, label=name, color=color, linewidth=2)

        # 6. 30년 이상 건물 수
        if 'old_buildings_30y' in stats[0]:
            old = [s['old_buildings_30y'] for s in stats]
            axes[1, 2].plot(old, label=name, color=color, linewidth=2)

    axes[0, 0].set_title('전국 평균 가격 (억원)', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('활성 주택 수', fontweight='bold')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title('평균 건물 상태 (0~1)', fontweight='bold')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_title('누적 멸실 (채)', fontweight='bold')
    axes[1, 0].set_xlabel('월')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('월간 멸실 (채)', fontweight='bold')
    axes[1, 1].set_xlabel('월')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_title('30년 이상 노후 건물', fontweight='bold')
    axes[1, 2].set_xlabel('월')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장됨: {save_path}")


def print_summary_table(all_results: Dict[str, Dict], title: str):
    """요약 테이블 출력"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

    # 멸실 데이터가 있는지 확인
    has_demolition = any('demolition_history' in r and r['demolition_history'] for r in all_results.values())

    if has_demolition:
        print(f"{'시나리오':<16} {'강남(억)':>8} {'전국(억)':>8} {'변화율':>8} {'거래량':>7} {'멸실':>6} {'상태':>6}")
        print('-'*80)
        for name, results in all_results.items():
            s_start = results['stats_history'][0]
            s_end = results['stats_history'][-1]
            change = (s_end['avg_price'] - s_start['avg_price']) / s_start['avg_price'] * 100

            total_demolished = sum(d['total_count'] for d in results.get('demolition_history', []))
            condition = s_end.get('mean_building_condition', 0)

            print(f"{name:<16} "
                  f"{s_end['price_gangnam']/10000:>8.1f} "
                  f"{s_end['avg_price']/10000:>8.1f} "
                  f"{change:>+7.1f}% "
                  f"{s_end['transaction_total']:>7} "
                  f"{total_demolished:>6} "
                  f"{condition:>6.3f}")
    else:
        print(f"{'시나리오':<20} {'강남(억)':>10} {'전국(억)':>10} {'변화율':>10} {'거래량':>8} {'자가율':>8}")
        print('-'*80)
        for name, results in all_results.items():
            s_start = results['stats_history'][0]
            s_end = results['stats_history'][-1]
            change = (s_end['avg_price'] - s_start['avg_price']) / s_start['avg_price'] * 100

            print(f"{name:<20} "
                  f"{s_end['price_gangnam']/10000:>10.1f} "
                  f"{s_end['avg_price']/10000:>10.1f} "
                  f"{change:>+9.1f}% "
                  f"{s_end['transaction_total']:>8} "
                  f"{s_end['homeowner_rate']*100:>7.1f}%")


# ============================================================================
# 메인 실행
# ============================================================================

def run_scenario_group(scenarios: List[ScenarioParams], steps: int = 36) -> Dict[str, Dict]:
    """시나리오 그룹 실행"""
    results = {}
    for params in scenarios:
        results[params.name] = run_simulation_with_params(params, steps)
    return results


def main():
    STEPS = 36  # 3년

    print("="*80)
    print("한국 부동산 ABM - 다중 파라미터 시뮬레이션")
    print("="*80)

    # 1. 공급량 시나리오
    print("\n" + "▶"*30)
    print("PART 1: 공급량에 따른 변화")
    print("▶"*30)
    supply_results = run_scenario_group(get_supply_scenarios(), STEPS)
    plot_scenario_results(supply_results, "공급량 시나리오 비교", "supply_scenarios.png")
    print_summary_table(supply_results, "공급량 시나리오 결과")

    # 2. 통화량 시나리오
    print("\n" + "▶"*30)
    print("PART 2: 통화량(M2)에 따른 변화")
    print("▶"*30)
    m2_results = run_scenario_group(get_m2_scenarios(), STEPS)
    plot_scenario_results(m2_results, "통화량(M2) 시나리오 비교", "m2_scenarios.png")
    plot_macro_details(m2_results, "통화량 시나리오 - 거시경제 상세", "m2_macro_details.png")
    print_summary_table(m2_results, "통화량 시나리오 결과")

    # 3. 금리 시나리오
    print("\n" + "▶"*30)
    print("PART 3: 금리에 따른 변화")
    print("▶"*30)
    rate_results = run_scenario_group(get_interest_scenarios(), STEPS)
    plot_scenario_results(rate_results, "금리 시나리오 비교", "rate_scenarios.png")
    print_summary_table(rate_results, "금리 시나리오 결과")

    # 4. 규제 시나리오
    print("\n" + "▶"*30)
    print("PART 4: 규제에 따른 변화")
    print("▶"*30)
    reg_results = run_scenario_group(get_regulation_scenarios(), STEPS)
    plot_scenario_results(reg_results, "규제 시나리오 비교", "regulation_scenarios.png")
    print_summary_table(reg_results, "규제 시나리오 결과")

    # 5. 노후화/멸실 시나리오
    print("\n" + "▶"*30)
    print("PART 5: 노후화/멸실에 따른 변화")
    print("▶"*30)
    depreciation_results = run_scenario_group(get_depreciation_scenarios(), STEPS)
    plot_depreciation_results(depreciation_results, "노후화/멸실 시나리오 비교", "depreciation_scenarios.png")
    plot_scenario_results(depreciation_results, "노후화 시나리오 - 시장 지표", "depreciation_market.png")
    print_summary_table(depreciation_results, "노후화/멸실 시나리오 결과")

    # 6. 복합 시나리오
    print("\n" + "▶"*30)
    print("PART 6: 복합 시나리오")
    print("▶"*30)
    combined_results = run_scenario_group(get_combined_scenarios(), STEPS)
    plot_scenario_results(combined_results, "복합 시나리오 비교", "combined_scenarios.png")
    plot_macro_details(combined_results, "복합 시나리오 - 거시경제 상세", "combined_macro_details.png")
    print_summary_table(combined_results, "복합 시나리오 결과")

    # 최종 분석
    print("\n" + "="*80)
    print("분석 결론")
    print("="*80)

    print("""
[공급량 효과]
- 공급 부족 → 가격 상승 압력, 거래량 감소
- 공급 확대 → 가격 안정화, 거래량 증가

[통화량(M2) 효과]
- M2 증가 → 유동성 확대 → 자산가격 상승 압력
- M2 긴축 → 유동성 축소 → 가격 하락 압력

[금리 효과]
- 저금리 → 대출 여력 증가 → 수요 증가 → 가격 상승
- 고금리 → 대출 비용 증가 → 수요 감소 → 가격 안정

[규제 효과]
- LTV/DTI 강화 → 대출 제한 → 수요 억제 → 가격 하락
- 규제 완화 → 대출 확대 → 수요 증가 → 가격 상승

[노후화/멸실 효과]
- 빠른 노후화 → 주택 재고 감소 → 가격 상승 압력
- 멸실 증가 → 공급 부족 심화 → 시장 불안정

[복합 효과]
- 버블 형성: 저금리 + 양적완화 + 공급부족 = 급격한 가격 상승
- 균형 정책: 적정 공급 + 중립 통화정책 = 안정적 시장
""")

    print("\n생성된 파일:")
    print("  - supply_scenarios.png: 공급량 시나리오")
    print("  - m2_scenarios.png: 통화량 시나리오")
    print("  - m2_macro_details.png: 통화량-거시경제 상세")
    print("  - rate_scenarios.png: 금리 시나리오")
    print("  - regulation_scenarios.png: 규제 시나리오")
    print("  - depreciation_scenarios.png: 노후화/멸실 시나리오")
    print("  - depreciation_market.png: 노후화-시장 지표")
    print("  - combined_scenarios.png: 복합 시나리오")
    print("  - combined_macro_details.png: 복합-거시경제 상세")


if __name__ == "__main__":
    main()
