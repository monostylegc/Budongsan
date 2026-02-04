"""시각화 모듈"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Dict, Any, List, Optional
import os

from .config import REGIONS, NUM_REGIONS


def setup_korean_font():
    """한글 폰트 설정"""
    # 리눅스 한글 폰트 경로들
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams['font.family'] = font_name
            break
    else:
        # 폰트 못 찾으면 기본값
        plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.rcParams['axes.unicode_minus'] = False


def plot_price_trends(results: Dict[str, Any], save_path: Optional[str] = None):
    """지역별 가격 추이"""
    setup_korean_font()

    price_history = results["price_history"]
    regions = results["regions"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    months = np.arange(len(price_history))
    years = months / 12

    # 1. 주요 지역 가격 추이
    ax1 = axes[0, 0]
    key_regions = [0, 1, 2, 4, 7, 12]  # 강남, 마용성, 기타서울, 경기남부, 부산, 기타지방
    for r in key_regions:
        prices = price_history[:, r] / 10000  # 억원 단위
        ax1.plot(years, prices, label=regions[r]["name"], linewidth=2)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Price (억원)")
    ax1.set_title("주요 지역 매매가 추이")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. 수도권 vs 비수도권
    ax2 = axes[0, 1]
    seoul = np.mean(price_history[:, 0:3], axis=1) / 10000
    gyeonggi = np.mean(price_history[:, 3:7], axis=1) / 10000
    jibang = np.mean(price_history[:, 7:], axis=1) / 10000

    ax2.plot(years, seoul, label="서울", linewidth=2, color="red")
    ax2.plot(years, gyeonggi, label="경기/인천", linewidth=2, color="blue")
    ax2.plot(years, jibang, label="지방", linewidth=2, color="green")

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Price (억원)")
    ax2.set_title("권역별 평균 가격")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 거래량
    ax3 = axes[1, 0]
    trans_history = results["transaction_history"]
    total_trans = np.sum(trans_history, axis=1)
    ax3.bar(years, total_trans, width=0.08, alpha=0.7)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Transactions")
    ax3.set_title("월간 거래량")
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 전세가율
    ax4 = axes[1, 1]
    jeonse_ratio = results["jeonse_ratio_history"]
    avg_ratio = np.mean(jeonse_ratio, axis=1) * 100

    ax4.plot(years, avg_ratio, linewidth=2, color="purple")
    ax4.axhline(y=80, color='red', linestyle='--', label='위험 수준 (80%)')
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Jeonse Ratio (%)")
    ax4.set_title("평균 전세가율")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()

    return fig


def plot_comparison(results_list: List[Dict[str, Any]],
                    labels: List[str],
                    save_path: Optional[str] = None):
    """여러 시나리오 비교"""
    setup_korean_font()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    # 1. 강남 가격 비교
    ax1 = axes[0, 0]
    for i, (results, label) in enumerate(zip(results_list, labels)):
        price_history = results["price_history"]
        years = np.arange(len(price_history)) / 12
        gangnam = price_history[:, 0] / 10000
        ax1.plot(years, gangnam, label=label, linewidth=2, color=colors[i])

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Price (억원)")
    ax1.set_title("강남 매매가 비교")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 전국 평균 가격
    ax2 = axes[0, 1]
    for i, (results, label) in enumerate(zip(results_list, labels)):
        stats = results["stats_history"]
        years = np.array([s["step"] for s in stats]) / 12
        avg_price = np.array([s["avg_price"] for s in stats]) / 10000
        ax2.plot(years, avg_price, label=label, linewidth=2, color=colors[i])

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Price (억원)")
    ax2.set_title("전국 평균 가격 비교")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 거래량 비교
    ax3 = axes[1, 0]
    for i, (results, label) in enumerate(zip(results_list, labels)):
        stats = results["stats_history"]
        years = np.array([s["step"] for s in stats]) / 12
        trans = np.array([s["transaction_total"] for s in stats])
        # 이동 평균
        window = 6
        trans_smooth = np.convolve(trans, np.ones(window)/window, mode='valid')
        years_smooth = years[window-1:]
        ax3.plot(years_smooth, trans_smooth, label=label, linewidth=2, color=colors[i])

    ax3.set_xlabel("Year")
    ax3.set_ylabel("Transactions (6개월 평균)")
    ax3.set_title("거래량 비교")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 자가보유율
    ax4 = axes[1, 1]
    for i, (results, label) in enumerate(zip(results_list, labels)):
        stats = results["stats_history"]
        years = np.array([s["step"] for s in stats]) / 12
        rate = np.array([s["homeowner_rate"] for s in stats]) * 100
        ax4.plot(years, rate, label=label, linewidth=2, color=colors[i])

    ax4.set_xlabel("Year")
    ax4.set_ylabel("Homeowner Rate (%)")
    ax4.set_title("자가보유율 비교")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()

    return fig


def plot_balloon_effect(results: Dict[str, Any], save_path: Optional[str] = None):
    """풍선효과 시각화 (히트맵)"""
    setup_korean_font()

    price_history = results["price_history"]
    regions = results["regions"]

    # 가격 변화율 계산
    changes = np.diff(price_history, axis=0) / price_history[:-1]
    changes = changes * 100  # 퍼센트

    # 히트맵
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(changes.T, aspect='auto', cmap='RdYlGn_r',
                   vmin=-5, vmax=5)

    # 축 설정
    years = np.arange(len(changes)) / 12
    year_ticks = np.arange(0, len(changes), 12)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([f"{int(y)}년" for y in year_ticks / 12])

    region_names = [regions[i]["name"] for i in range(NUM_REGIONS)]
    ax.set_yticks(range(NUM_REGIONS))
    ax.set_yticklabels(region_names)

    ax.set_xlabel("Time")
    ax.set_ylabel("Region")
    ax.set_title("지역별 월간 가격변화율 (%) - 풍선효과 시각화")

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("가격 변화율 (%)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()

    return fig


def print_summary(results: Dict[str, Any]):
    """결과 요약 출력"""
    stats = results["stats_history"]
    price_history = results["price_history"]
    regions = results["regions"]

    print("\n" + "="*60)
    print("시뮬레이션 결과 요약")
    print("="*60)

    # 시작/종료 가격
    start_prices = price_history[0]
    end_prices = price_history[-1]
    change_rates = (end_prices - start_prices) / start_prices * 100

    print("\n[지역별 가격 변화]")
    print(f"{'지역':<12} {'시작가':<12} {'종료가':<12} {'변화율':<10}")
    print("-" * 46)
    for r in range(NUM_REGIONS):
        name = regions[r]["name"]
        start = start_prices[r] / 10000
        end = end_prices[r] / 10000
        change = change_rates[r]
        print(f"{name:<12} {start:>8.1f}억    {end:>8.1f}억    {change:>+6.1f}%")

    # 요약 통계
    print("\n[요약 통계]")
    total_trans = sum(s["transaction_total"] for s in stats)
    final_homeowner = stats[-1]["homeowner_rate"] * 100
    final_multi = stats[-1]["multi_owner_rate"] * 100

    print(f"총 거래량: {total_trans:,}건")
    print(f"최종 자가보유율: {final_homeowner:.1f}%")
    print(f"최종 다주택자 비율: {final_multi:.1f}%")
    print(f"서울 평균 상승률: {np.mean(change_rates[0:3]):.1f}%")
    print(f"지방 평균 상승률: {np.mean(change_rates[7:]):.1f}%")

    print("="*60 + "\n")
