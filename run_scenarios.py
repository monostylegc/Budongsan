"""2026년 한국 부동산 시장 시뮬레이션

현재 한국 상황 기반 시나리오 분석:
- 기준금리 2.5% (한국은행 2026.01 동결)
- GDP 성장률 ~1.5% (2025 1.0%, 2026 1.8%)
- 실업률 ~2.6%
- 서울 아파트 평균 15억 (2025.12 기준)
- 수도권 규제지역 LTV 40%, DSR 40%
- 2주택 이상 규제지역 대출 금지
- 가계부채/GDP ~90%

가구/주택 비율: 실제 한국 2,300만 가구 : 1,890만 호 = 1:100 축소
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.realestate import Simulation, Config

# ─────────────────────────────────────────────────────────
# 실제 한국 비율 (1:770 축소, 주택보급률 82% 유지)
# 총 가구 2,300만 → 30,000  |  주택 1,890만 → 24,600
# ─────────────────────────────────────────────────────────
N_HOUSEHOLDS = 30_000
N_HOUSES = 24_600
STEPS = 60  # 5년 (2026~2030)
ARCH = "cuda"


def make_korea_2026_config(seed=42) -> Config:
    """2026년 현재 한국 경제 상황 반영 Config 생성"""
    config = Config(
        num_households=N_HOUSEHOLDS,
        num_houses=N_HOUSES,
        num_steps=STEPS,
        seed=seed,
    )

    # ── 거시경제: 저성장 기조 ──
    config.macro.gdp_growth_mean = 0.015       # 연 1.5% (2026 전망)
    config.macro.initial_gdp_growth = 0.015
    config.macro.neutral_real_rate = 0.01       # 중립 실질금리 1%
    config.macro.initial_inflation = 0.025      # 인플레이션 2.5%

    # ── 금리: 인하 사이클 마무리 ──
    config.policy.interest_rate = 0.025         # 기준금리 2.5%
    config.policy.mortgage_spread = 0.01        # 스프레드 1% → 주담대 3.5%

    # ── 대출규제: 수도권 규제 강화 ──
    config.policy.ltv_first_time = 0.70         # 생애최초 70%
    config.policy.ltv_1house = 0.40             # 1주택자 40% (규제지역)
    config.policy.ltv_2house = 0.00             # 2주택 대출 금지
    config.policy.ltv_3house = 0.00             # 3주택 대출 금지
    config.policy.dti_limit = 0.40              # DTI 40%
    config.policy.dsr_limit = 0.40              # DSR 40%

    # ── 세금: 현행 유지 ──
    config.policy.acq_tax_1house = 0.01         # 취득세 1%
    config.policy.acq_tax_2house = 0.08         # 2주택 8%
    config.policy.acq_tax_3house = 0.12         # 3주택 12%
    config.policy.jongbu_rate = 0.02            # 종부세 2%
    config.policy.jongbu_threshold_1house = 110000  # 1주택 11억
    config.policy.jongbu_threshold_multi = 60000    # 다주택 6억

    return config


def run_scenario(name: str, config: Config, verbose=True) -> Dict[str, Any]:
    """시나리오 실행 + 시간 측정"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    start = time.time()
    sim = Simulation(config, arch=ARCH)
    results = sim.run(steps=STEPS, verbose=verbose)
    elapsed = time.time() - start
    print(f"  소요시간: {elapsed:.1f}초")

    return results


# ─────────────────────────────────────────────────────────
# 시나리오 정의
# ─────────────────────────────────────────────────────────

def scenario_baseline() -> Dict[str, Any]:
    """시나리오 1: 현재 한국 (2026 기본)
    - 기준금리 2.5%, GDP 1.5%, 현행 규제 유지
    """
    config = make_korea_2026_config()
    return run_scenario("시나리오 1: 현재 한국 (2026 기본)", config)


def scenario_rate_cut() -> Dict[str, Any]:
    """시나리오 2: 추가 금리 인하
    - 기준금리 2.5% → 1.5% (100bp 인하)
    - 경기 부양 목적, 주담대 2.5%
    """
    config = make_korea_2026_config()
    config.policy.interest_rate = 0.015         # 기준금리 1.5%
    config.macro.neutral_real_rate = 0.005      # 중립금리 하향
    return run_scenario("시나리오 2: 추가 금리 인하 (2.5→1.5%)", config)


def scenario_recession() -> Dict[str, Any]:
    """시나리오 3: 경기 침체
    - GDP 성장률 -1% (수출 부진, 글로벌 둔화)
    - 실업 상승, 기업 투자 감소
    """
    config = make_korea_2026_config()
    config.macro.gdp_growth_mean = -0.01        # 역성장
    config.macro.initial_gdp_growth = -0.01
    config.macro.gdp_growth_volatility = 0.015  # 변동성 증가
    return run_scenario("시나리오 3: 경기 침체 (GDP -1%)", config)


def scenario_deregulation() -> Dict[str, Any]:
    """시나리오 4: 규제 완화
    - LTV 60% (규제지역), 다주택 30% 허용
    - 종부세 인하, 양도세 중과 배제
    - 스트레스 DSR 완화
    """
    config = make_korea_2026_config()
    config.policy.ltv_1house = 0.60             # 1주택 60%
    config.policy.ltv_2house = 0.30             # 2주택 30% 허용
    config.policy.dti_limit = 0.50              # DTI 50%
    config.policy.jongbu_rate = 0.01            # 종부세 절반
    config.policy.transfer_tax_multi_long = 0.40  # 양도세 중과 배제
    return run_scenario("시나리오 4: 규제 완화 (LTV↑, 종부세↓)", config)


def scenario_supply_shortage() -> Dict[str, Any]:
    """시나리오 5: 공급 절벽
    - 2026년 수도권 입주물량 30% 감소 (16.1만 → 11.2만호)
    - 공급 탄력성 축소, 재건축 지연
    """
    config = make_korea_2026_config()
    config.supply.base_supply_rate = 0.0005     # 기본 공급률 절반
    config.supply.elasticity_gangnam = 0.15     # 강남 공급 극히 제한
    config.supply.elasticity_seoul = 0.25       # 서울 공급 축소
    config.supply.elasticity_gyeonggi = 0.8     # 경기도도 축소
    config.supply.elasticity_local = 1.5        # 지방만 여유
    config.supply.redevelopment_base_prob = 0.0003  # 재건축 지연
    return run_scenario("시나리오 5: 공급 절벽 (입주물량 30%↓)", config)


# ─────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────

def plot_all(all_results: Dict[str, Dict], save_path="korea_2026_scenarios.png"):
    """4x2 종합 비교 그래프"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
    names = list(all_results.keys())

    # 1. 강남3구 가격 추이
    ax = axes[0, 0]
    for i, (name, res) in enumerate(all_results.items()):
        prices = [s['price_gangnam'] / 10000 for s in res['stats_history']]
        ax.plot(prices, label=name, color=colors[i], linewidth=2)
    ax.set_title('강남3구 아파트 가격', fontweight='bold')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 2. 전국 평균 가격
    ax = axes[0, 1]
    for i, (name, res) in enumerate(all_results.items()):
        prices = [s['avg_price'] / 10000 for s in res['stats_history']]
        ax.plot(prices, label=name, color=colors[i], linewidth=2)
    ax.set_title('전국 평균 가격', fontweight='bold')
    ax.set_ylabel('가격 (억원)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 3. 실업률
    ax = axes[1, 0]
    for i, (name, res) in enumerate(all_results.items()):
        unemp = [s['unemployment_rate'] * 100 for s in res['stats_history']]
        ax.plot(unemp, label=name, color=colors[i], linewidth=2)
    ax.set_title('실업률', fontweight='bold')
    ax.set_ylabel('%')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 4. 거래량 (6개월 이동평균)
    ax = axes[1, 1]
    for i, (name, res) in enumerate(all_results.items()):
        trans = [s['transaction_total'] for s in res['stats_history']]
        if len(trans) >= 6:
            ma = np.convolve(trans, np.ones(6)/6, mode='valid')
            ax.plot(ma, label=name, color=colors[i], linewidth=2)
    ax.set_title('월간 거래량 (6개월 이동평균)', fontweight='bold')
    ax.set_ylabel('건수')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 5. 자가보유율
    ax = axes[2, 0]
    for i, (name, res) in enumerate(all_results.items()):
        rate = [s['homeowner_rate'] * 100 for s in res['stats_history']]
        ax.plot(rate, label=name, color=colors[i], linewidth=2)
    ax.set_title('자가보유율', fontweight='bold')
    ax.set_ylabel('%')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 6. 평균 소득
    ax = axes[2, 1]
    for i, (name, res) in enumerate(all_results.items()):
        inc = [s['avg_income'] for s in res['stats_history']]
        ax.plot(inc, label=name, color=colors[i], linewidth=2)
    ax.set_title('평균 소득 (만원/월)', fontweight='bold')
    ax.set_ylabel('만원')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 7. 수요/공급 비율
    ax = axes[3, 0]
    for i, (name, res) in enumerate(all_results.items()):
        ds = [s['demand_supply_ratio'] for s in res['stats_history']]
        ax.plot(ds, label=name, color=colors[i], linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='균형')
    ax.set_title('수요/공급 비율', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('비율')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # 8. 강제매도 위험
    ax = axes[3, 1]
    for i, (name, res) in enumerate(all_results.items()):
        risk = [s.get('at_risk_count', 0) for s in res['stats_history']]
        ax.plot(risk, label=name, color=colors[i], linewidth=2)
    ax.set_title('강제매도 위험 가구', fontweight='bold')
    ax.set_xlabel('월')
    ax.set_ylabel('가구 수')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    plt.suptitle('2026 한국 부동산 시장 시뮬레이션 (5년 전망)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n그래프 저장: {save_path}")


def print_summary(all_results: Dict[str, Dict]):
    """시나리오별 요약"""
    print("\n" + "=" * 100)
    print("  2026 한국 부동산 시뮬레이션 - 시나리오 비교 (5년 전망)")
    print("=" * 100)

    header = (f"{'시나리오':<25} {'강남(억)':>9} {'전국(억)':>9} {'변동률':>8} "
              f"{'거래량':>7} {'자가율':>7} {'실업률':>7} {'소득':>7} {'위험':>5}")
    print(header)
    print("-" * 100)

    for name, res in all_results.items():
        s0 = res['stats_history'][0]
        s = res['stats_history'][-1]
        gangnam_chg = (s['price_gangnam'] - s0['price_gangnam']) / s0['price_gangnam'] * 100

        print(f"{name:<25} "
              f"{s['price_gangnam']/10000:>8.1f} "
              f"{s['avg_price']/10000:>8.1f} "
              f"{gangnam_chg:>+7.1f}% "
              f"{s['transaction_total']:>7,} "
              f"{s['homeowner_rate']*100:>6.1f}% "
              f"{s['unemployment_rate']*100:>6.1f}% "
              f"{s['avg_income']:>6.0f} "
              f"{s.get('at_risk_count', 0):>5}")

    # 지역별 가격 변동 비교
    print("\n" + "=" * 100)
    print("  지역별 가격 변동률 (5년)")
    print("=" * 100)
    print(f"{'시나리오':<25} {'강남':>9} {'마용성':>9} {'기타서울':>9} {'경기':>9} {'지방':>9}")
    print("-" * 100)

    for name, res in all_results.items():
        ph = res['price_history']
        if len(ph) >= 2:
            p0 = ph[0]
            p1 = ph[-1]
            chg = (p1 - p0) / (p0 + 1e-6) * 100
            print(f"{name:<25} "
                  f"{chg[0]:>+8.1f}% "
                  f"{chg[1]:>+8.1f}% "
                  f"{chg[2]:>+8.1f}% "
                  f"{np.mean(chg[4:7]):>+8.1f}% "
                  f"{np.mean(chg[7:]):>+8.1f}%")


def main():
    """메인 실행"""
    total_start = time.time()

    print("=" * 60)
    print("  2026 한국 부동산 시장 ABM 시뮬레이션")
    print(f"  가구 {N_HOUSEHOLDS:,} | 주택 {N_HOUSES:,} | {STEPS}개월")
    print(f"  GPU: {ARCH.upper()}")
    print("=" * 60)

    all_results = {}

    all_results['기본 (현재)'] = scenario_baseline()
    all_results['금리인하'] = scenario_rate_cut()
    all_results['경기침체'] = scenario_recession()
    all_results['규제완화'] = scenario_deregulation()
    all_results['공급절벽'] = scenario_supply_shortage()

    # 시각화 + 요약
    plot_all(all_results)
    print_summary(all_results)

    total_elapsed = time.time() - total_start
    print(f"\n총 소요시간: {total_elapsed:.1f}초")
    print("생성 파일: korea_2026_scenarios.png")


if __name__ == "__main__":
    main()
