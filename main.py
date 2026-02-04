"""한국 부동산 시장 ABM 시뮬레이션 - 메인 엔트리포인트"""

import sys
sys.path.insert(0, "src")

from realestate import Simulation, Config
from realestate.config import PolicyConfig
from realestate.visualization import plot_price_trends, plot_comparison, plot_balloon_effect, print_summary
import numpy as np


def run_baseline():
    """기본 시나리오 실행"""
    print("\n" + "="*60)
    print("기본 시나리오 (현행 정책)")
    print("="*60)

    config = Config(
        num_households=100_000,  # 테스트용 축소 (10만)
        num_houses=50_000,
        num_steps=120,  # 10년
        seed=42,
    )

    sim = Simulation(config, arch="vulkan")
    results = sim.run(verbose=True)

    print_summary(results)
    plot_price_trends(results, save_path="output_baseline.png")

    return results


def run_hypothesis1_transfer_tax():
    """
    가설 1: 다주택자 규제의 역설
    - 양도세 중과가 매물 잠김을 유발하는지
    - 실험: 양도세 40%, 60%, 80% 비교
    """
    print("\n" + "="*60)
    print("가설 1: 다주택자 양도세 중과 효과")
    print("="*60)

    results_list = []
    labels = []
    tax_rates = [0.40, 0.60, 0.80]

    for tax_rate in tax_rates:
        print(f"\n[양도세 {int(tax_rate*100)}%]")
        config = Config(
            num_households=100_000,
            num_houses=50_000,
            num_steps=60,  # 5년
            seed=42,
            policy=PolicyConfig(
                transfer_tax_multi_short=tax_rate + 0.10,
                transfer_tax_multi_long=tax_rate,
            )
        )
        sim = Simulation(config, arch="vulkan")
        results = sim.run(verbose=True)
        results_list.append(results)
        labels.append(f"양도세 {int(tax_rate*100)}%")

    plot_comparison(results_list, labels, save_path="output_hypothesis1_transfer_tax.png")

    # 결과 분석
    print("\n" + "="*60)
    print("가설 1 결과 분석")
    print("="*60)

    for label, results in zip(labels, results_list):
        stats = results["stats_history"]
        price_history = results["price_history"]

        gangnam_change = (price_history[-1][0] - price_history[0][0]) / price_history[0][0] * 100
        total_trans = sum(s["transaction_total"] for s in stats)
        multi_rate = stats[-1]["multi_owner_rate"] * 100

        print(f"\n[{label}]")
        print(f"  강남 가격 변화: {gangnam_change:+.1f}%")
        print(f"  총 거래량: {total_trans:,}건")
        print(f"  다주택자 비율: {multi_rate:.1f}%")

    return results_list, labels


def run_hypothesis2_balloon_effect():
    """
    가설 2: 풍선효과의 비선형성
    - 서울 규제 강도에 따른 수도권/지방 가격 전파
    - 실험: 서울 LTV 0%, 20%, 40% 비교
    """
    print("\n" + "="*60)
    print("가설 2: 풍선효과 - 서울 LTV 규제 효과")
    print("="*60)

    results_list = []
    labels = []
    ltv_rates = [0.0, 0.20, 0.40]

    for ltv in ltv_rates:
        print(f"\n[서울 LTV {int(ltv*100)}%]")
        config = Config(
            num_households=100_000,
            num_houses=50_000,
            num_steps=60,  # 5년
            seed=42,
            policy=PolicyConfig(
                ltv_1house=ltv + 0.10,  # 1주택자는 약간 높게
                ltv_2house=ltv,
                ltv_3house=0.0,
            )
        )
        sim = Simulation(config, arch="vulkan")
        results = sim.run(verbose=True)
        results_list.append(results)
        labels.append(f"LTV {int(ltv*100)}%")

    plot_comparison(results_list, labels, save_path="output_hypothesis2_balloon.png")

    # 풍선효과 시각화
    for i, (results, label) in enumerate(zip(results_list, labels)):
        plot_balloon_effect(results, save_path=f"output_hypothesis2_heatmap_{i}.png")

    # 결과 분석
    print("\n" + "="*60)
    print("가설 2 결과 분석: 풍선효과")
    print("="*60)

    for label, results in zip(labels, results_list):
        price_history = results["price_history"]
        stats = results["stats_history"]

        seoul_change = np.mean([
            (price_history[-1][i] - price_history[0][i]) / price_history[0][i] * 100
            for i in range(3)
        ])
        gyeonggi_change = np.mean([
            (price_history[-1][i] - price_history[0][i]) / price_history[0][i] * 100
            for i in range(3, 7)
        ])
        jibang_change = np.mean([
            (price_history[-1][i] - price_history[0][i]) / price_history[0][i] * 100
            for i in range(7, 13)
        ])

        print(f"\n[{label}]")
        print(f"  서울 가격 변화: {seoul_change:+.1f}%")
        print(f"  수도권 가격 변화: {gyeonggi_change:+.1f}%")
        print(f"  지방 가격 변화: {jibang_change:+.1f}%")
        print(f"  풍선효과 지수: {jibang_change - seoul_change:+.1f}%p")

    return results_list, labels


def run_hypothesis3_interest_vs_regulation():
    """
    가설 3: 금리 vs 규제
    - 금리 인상 vs 규제 강화 중 어느 것이 효과적인지
    - 실험: 금리 3%/5%/7%, 규제 강/중/약 조합
    """
    print("\n" + "="*60)
    print("가설 3: 금리 vs 규제 효과 비교")
    print("="*60)

    results_list = []
    labels = []

    # 시나리오 1: 저금리 + 강규제
    print("\n[저금리 3% + 강규제]")
    config1 = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=60,
        seed=42,
        policy=PolicyConfig(
            interest_rate=0.03,
            ltv_2house=0.0,
            ltv_3house=0.0,
            transfer_tax_multi_long=0.70,
        )
    )
    sim1 = Simulation(config1, arch="vulkan")
    results1 = sim1.run(verbose=True)
    results_list.append(results1)
    labels.append("금리 3% + 강규제")

    # 시나리오 2: 중금리 + 중규제
    print("\n[중금리 5% + 중규제]")
    config2 = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=60,
        seed=42,
        policy=PolicyConfig(
            interest_rate=0.05,
            ltv_2house=0.20,
            ltv_3house=0.10,
            transfer_tax_multi_long=0.50,
        )
    )
    sim2 = Simulation(config2, arch="vulkan")
    results2 = sim2.run(verbose=True)
    results_list.append(results2)
    labels.append("금리 5% + 중규제")

    # 시나리오 3: 고금리 + 약규제
    print("\n[고금리 7% + 약규제]")
    config3 = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=60,
        seed=42,
        policy=PolicyConfig(
            interest_rate=0.07,
            ltv_2house=0.40,
            ltv_3house=0.20,
            transfer_tax_multi_long=0.40,
        )
    )
    sim3 = Simulation(config3, arch="vulkan")
    results3 = sim3.run(verbose=True)
    results_list.append(results3)
    labels.append("금리 7% + 약규제")

    # 시나리오 4: 고금리 + 강규제 (복합)
    print("\n[고금리 7% + 강규제]")
    config4 = Config(
        num_households=100_000,
        num_houses=50_000,
        num_steps=60,
        seed=42,
        policy=PolicyConfig(
            interest_rate=0.07,
            ltv_2house=0.0,
            ltv_3house=0.0,
            transfer_tax_multi_long=0.70,
        )
    )
    sim4 = Simulation(config4, arch="vulkan")
    results4 = sim4.run(verbose=True)
    results_list.append(results4)
    labels.append("금리 7% + 강규제")

    plot_comparison(results_list, labels, save_path="output_hypothesis3_interest_regulation.png")

    # 결과 분석
    print("\n" + "="*60)
    print("가설 3 결과 분석: 금리 vs 규제")
    print("="*60)

    for label, results in zip(labels, results_list):
        stats = results["stats_history"]
        price_history = results["price_history"]

        avg_change = np.mean([
            (price_history[-1][i] - price_history[0][i]) / price_history[0][i] * 100
            for i in range(13)
        ])
        total_trans = sum(s["transaction_total"] for s in stats)
        homeowner_rate = stats[-1]["homeowner_rate"] * 100

        print(f"\n[{label}]")
        print(f"  전국 평균 가격 변화: {avg_change:+.1f}%")
        print(f"  총 거래량: {total_trans:,}건")
        print(f"  자가보유율: {homeowner_rate:.1f}%")

    return results_list, labels


def run_hypothesis4_jongbu_tax():
    """
    가설 4: 종부세 효과
    - 종부세가 다주택자 매도를 유발하는지
    - 실험: 종부세율 1%, 3%, 6% 비교
    """
    print("\n" + "="*60)
    print("가설 4: 종부세 효과")
    print("="*60)

    results_list = []
    labels = []
    jongbu_rates = [0.01, 0.03, 0.06]

    for rate in jongbu_rates:
        print(f"\n[종부세 {int(rate*100)}%]")
        config = Config(
            num_households=100_000,
            num_houses=50_000,
            num_steps=60,  # 5년
            seed=42,
            policy=PolicyConfig(
                jongbu_rate=rate,
                jongbu_threshold_multi=60000,  # 6억 기준
            )
        )
        sim = Simulation(config, arch="vulkan")
        results = sim.run(verbose=True)
        results_list.append(results)
        labels.append(f"종부세 {int(rate*100)}%")

    plot_comparison(results_list, labels, save_path="output_hypothesis4_jongbu.png")

    # 결과 분석
    print("\n" + "="*60)
    print("가설 4 결과 분석: 종부세 효과")
    print("="*60)

    for label, results in zip(labels, results_list):
        stats = results["stats_history"]
        price_history = results["price_history"]

        gangnam_change = (price_history[-1][0] - price_history[0][0]) / price_history[0][0] * 100
        total_trans = sum(s["transaction_total"] for s in stats)
        multi_rate_start = stats[0]["multi_owner_rate"] * 100
        multi_rate_end = stats[-1]["multi_owner_rate"] * 100

        print(f"\n[{label}]")
        print(f"  강남 가격 변화: {gangnam_change:+.1f}%")
        print(f"  총 거래량: {total_trans:,}건")
        print(f"  다주택자 비율: {multi_rate_start:.1f}% -> {multi_rate_end:.1f}%")
        print(f"  다주택자 감소: {multi_rate_start - multi_rate_end:+.1f}%p")

    return results_list, labels


def run_all_experiments():
    """모든 실험 실행"""
    print("\n" + "="*70)
    print("한국 부동산 시장 ABM 시뮬레이션 - 전체 실험 수행")
    print("="*70)

    # 기본 시나리오
    print("\n[1/5] 기본 시나리오 실행")
    run_baseline()

    # 가설 1: 양도세
    print("\n[2/5] 가설 1: 양도세 효과")
    run_hypothesis1_transfer_tax()

    # 가설 2: 풍선효과
    print("\n[3/5] 가설 2: 풍선효과")
    run_hypothesis2_balloon_effect()

    # 가설 3: 금리 vs 규제
    print("\n[4/5] 가설 3: 금리 vs 규제")
    run_hypothesis3_interest_vs_regulation()

    # 가설 4: 종부세
    print("\n[5/5] 가설 4: 종부세 효과")
    run_hypothesis4_jongbu_tax()

    print("\n" + "="*70)
    print("모든 실험 완료!")
    print("="*70)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="한국 부동산 시장 ABM 시뮬레이션")
    parser.add_argument("--scenario", type=str, default="baseline",
                        choices=["baseline", "h1", "h2", "h3", "h4", "all"],
                        help="실행할 시나리오 (h1=양도세, h2=풍선효과, h3=금리vs규제, h4=종부세)")
    parser.add_argument("--arch", type=str, default="vulkan",
                        choices=["vulkan", "cuda", "cpu"],
                        help="Taichi 백엔드")

    args = parser.parse_args()

    if args.scenario == "baseline":
        run_baseline()
    elif args.scenario == "h1":
        run_hypothesis1_transfer_tax()
    elif args.scenario == "h2":
        run_hypothesis2_balloon_effect()
    elif args.scenario == "h3":
        run_hypothesis3_interest_vs_regulation()
    elif args.scenario == "h4":
        run_hypothesis4_jongbu_tax()
    elif args.scenario == "all":
        run_all_experiments()


if __name__ == "__main__":
    main()
