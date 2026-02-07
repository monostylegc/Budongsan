"""감정 레이어 - 불안, 만족, FOMO, 후회

감정은 즉시 반응이 아니라 누적됨.
감정이 의사결정에 영향: 불안↑ → System1 지배 → 충동구매.
"""

import numpy as np


class EmotionEngine:
    """감정 엔진"""

    def __init__(self, cfg):
        """cfg: EmotionConfig"""
        self.cfg = cfg

    def update(self, agents, actual_prices: np.ndarray, price_changes: np.ndarray):
        """매 스텝 감정 업데이트

        Args:
            agents: AgentPopulation
            actual_prices: (n_regions,) 실제 가격
            price_changes: (n_regions,) 월간 가격 변화율
        """
        d = agents.data
        cfg = self.cfg

        homeless = d.owned_houses == 0
        owners = d.owned_houses > 0

        # === 불안감 (anxiety) ===
        # 무주택자: 매월 기본 증가 + 가격 상승 시 추가
        d.anxiety[homeless] += cfg.anxiety_base_homeless

        # 가격이 저축보다 빠르게 오르면 추가 불안
        region_changes = price_changes[d.region]
        savings_rate = np.where(d.income > 0, d.housing_fund / (d.income * 12 + 1e-6), 0)
        price_faster = region_changes > savings_rate * 0.01
        d.anxiety[homeless & price_faster] += region_changes[homeless & price_faster] * cfg.anxiety_price_sensitivity

        # 주택 보유자: 불안 감쇠
        d.anxiety[owners] -= cfg.anxiety_decay
        d.anxiety = np.clip(d.anxiety, 0, 1)

        # === 만족도 (satisfaction) ===
        # 주택 보유 + 가격 상승 → 만족도 증가
        d.satisfaction[owners] = np.clip(
            cfg.satisfaction_base_owner + region_changes[owners] * 2,
            0, 1
        )
        # 무주택자 만족도 하락
        d.satisfaction[homeless] = np.clip(
            d.satisfaction[homeless] - 0.01,
            0.1, 0.5
        )

        # === FOMO 누적 ===
        # 안 샀는데 가격 오르면 누적
        rising = region_changes[d.region] > 0.01
        d.fomo_level[homeless & rising] += cfg.fomo_accumulation_rate * region_changes[d.region[homeless & rising]]
        # 안 오르면 감쇠
        d.fomo_level[~rising] -= cfg.fomo_decay_rate
        d.fomo_level = np.clip(d.fomo_level, 0, 1)

        # === 후회 ===
        # 과거 결정 결과 기반 (memory에서 계산)
        # 여기서는 시간에 따른 자연 감쇠만
        d.regret *= 0.95
        d.regret = np.clip(d.regret, 0, 1)

    def get_emotional_intensity(self, agents) -> np.ndarray:
        """감정 강도 (의사결정 모드에 영향)

        Returns:
            (n,) 0~1 값. 높을수록 감정적 (System1 지배)
        """
        d = agents.data
        return np.clip(
            d.anxiety * 0.4 + d.fomo_level * 0.3 + d.regret * 0.1 + (1 - d.satisfaction) * 0.2,
            0, 1
        )
