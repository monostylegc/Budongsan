"""시뮬레이션 결과 시각화"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.family'] = 'NanumGothic'
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_price_trends(recorder, region_names: list[str], save_path: str = None):
    """지역별 가격 추이"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    prices = recorder.get_price_series()
    if len(prices) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    months = np.arange(len(prices))

    for i, name in enumerate(region_names):
        if i < prices.shape[1]:
            ax.plot(months, prices[:, i], label=name, linewidth=1.5)

    ax.set_xlabel("월")
    ax.set_ylabel("평균 가격 (만원)")
    ax.set_title("지역별 주택 가격 추이")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_agent_emotions(recorder, save_path: str = None):
    """에이전트 감정 추이"""
    if not HAS_MATPLOTLIB:
        return

    history = recorder.history
    months = [s.month for s in history]
    anxiety = [s.avg_anxiety for s in history]
    fomo = [s.avg_fomo for s in history]
    satisfaction = [s.avg_satisfaction for s in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(months, anxiety, label="불안", color='red', linewidth=1.5)
    ax.plot(months, fomo, label="FOMO", color='orange', linewidth=1.5)
    ax.plot(months, satisfaction, label="만족", color='green', linewidth=1.5)

    ax.set_xlabel("월")
    ax.set_ylabel("평균 수준")
    ax.set_title("에이전트 평균 감정 추이")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_ownership_distribution(recorder, save_path: str = None):
    """주택 보유 분포 추이"""
    if not HAS_MATPLOTLIB:
        return

    history = recorder.history
    months = [s.month for s in history]
    homeless = [s.homeless_rate for s in history]
    single = [s.homeowner_rate for s in history]
    multi = [s.multi_owner_rate for s in history]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(months, homeless, single, multi,
                 labels=['무주택', '1주택', '다주택'],
                 colors=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                 alpha=0.7)
    ax.set_xlabel("월")
    ax.set_ylabel("비율")
    ax.set_title("주택 보유 분포 변화")
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def plot_macro_indicators(recorder, save_path: str = None):
    """거시경제 지표 추이"""
    if not HAS_MATPLOTLIB:
        return

    history = recorder.history
    months = [s.month for s in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # GDP growth
    axes[0, 0].plot(months, [s.gdp_growth for s in history], color='blue')
    axes[0, 0].set_title("GDP 성장률")
    axes[0, 0].grid(True, alpha=0.3)

    # Inflation
    axes[0, 1].plot(months, [s.inflation for s in history], color='red')
    axes[0, 1].set_title("인플레이션")
    axes[0, 1].grid(True, alpha=0.3)

    # Interest rate
    axes[1, 0].plot(months, [s.interest_rate for s in history], color='green')
    axes[1, 0].set_title("모기지 금리")
    axes[1, 0].grid(True, alpha=0.3)

    # Unemployment
    axes[1, 1].plot(months, [s.unemployment_rate for s in history], color='orange')
    axes[1, 1].set_title("실업률")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("거시경제 지표")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
