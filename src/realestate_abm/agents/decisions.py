"""의사결정 레이어 - 트리거 기반 + 만족화

에이전트는 매달 평가하지 않음 (현실과 동일).
트리거 발동 → 탐색 → 만족화 기반 선택.
"""

import numpy as np


class DecisionEngine:
    """의사결정 엔진"""

    def __init__(self, cfg):
        """cfg: DecisionConfig"""
        self.cfg = cfg

    def check_triggers(self, agents, price_changes: np.ndarray, rng: np.random.Generator):
        """트리거 체크 - 행동 개시 여부 결정

        트리거 안 된 에이전트 → 대기 (현상유지 편향)

        Args:
            agents: AgentPopulation
            price_changes: (n_regions,) 가격 변화율
            rng: 난수 생성기
        """
        d = agents.data
        n = agents.n
        cfg = self.cfg

        triggered = np.zeros(n, dtype=bool)

        # 1. 생애 이벤트 (결혼, 출산 → 즉시 트리거)
        newly_married = (d.life_stage == 1) & (d.owned_houses == 0)
        triggered |= newly_married

        new_parent = (d.life_stage == 2) & (d.owned_houses == 0)
        triggered |= new_parent

        # 2. 시장 신호 (내가 아는 가격이 5%+ 변동)
        home_region_changes = price_changes[d.region]
        big_change = np.abs(home_region_changes) > cfg.trigger_price_change
        triggered |= big_change

        # 3. 사회적 압력 (이웃 30%+ 매수중)
        social_pressure = d.neighbor_buying_ratio > cfg.trigger_social_ratio
        triggered |= social_pressure

        # 4. 감정 임계 (불안 > 0.7 또는 FOMO > 0.8)
        anxiety_trigger = d.anxiety > cfg.trigger_anxiety_threshold
        fomo_trigger = d.fomo_level > cfg.trigger_fomo_threshold
        triggered |= anxiety_trigger | fomo_trigger

        # 5. 시간 압력 (무주택 24개월+)
        time_pressure = (d.homeless_months > cfg.trigger_homeless_months) & (d.owned_houses == 0)
        triggered |= time_pressure

        # 6. 랜덤 탐색 (매월 5% 확률)
        random_search = rng.random(n) < cfg.trigger_random_search_prob
        triggered |= random_search

        # 7. 은퇴기 다운사이징
        retirement = (d.life_stage == 5) & (d.owned_houses >= 1)
        triggered |= retirement & (rng.random(n) < 0.02)

        d.is_triggered = triggered.astype(np.int32)

    def apply_satisficing(
        self,
        agents,
        buy_scores: np.ndarray,
        sell_scores: np.ndarray,
        rng: np.random.Generator,
    ):
        """만족화 기반 최종 결정

        최적이 아닌 '충분히 좋은' 선택.
        경험 많은 에이전트(info_quality↑)는 기준 높음.

        Args:
            agents: AgentPopulation
            buy_scores: (n,) 구매 점수
            sell_scores: (n,) 매도 점수
            rng: 난수 생성기
        """
        d = agents.data
        cfg = self.cfg

        # 트리거 안 된 에이전트는 대기
        not_triggered = d.is_triggered == 0
        buy_scores[not_triggered] = 0
        sell_scores[not_triggered] = 0

        # 만족화 기준 (경험에 따라 상승)
        threshold = cfg.satisficing_base_threshold + d.info_quality * 0.2

        # 지연 확률: 인내심 있고 차분하면 "좀 더 지켜보자"
        delay_prob = d.patience * (1 - d.anxiety) * cfg.delay_patience_weight
        delays = rng.random(agents.n) < delay_prob

        # 매수 결정
        buy_decision = (buy_scores > threshold) & (d.is_triggered == 1) & ~delays
        # 무주택자는 기준 낮춤
        homeless = d.owned_houses == 0
        buy_decision |= (buy_scores > threshold * 0.7) & homeless & (d.is_triggered == 1)

        # 매도 결정
        sell_decision = (sell_scores > threshold * 0.8) & (d.is_triggered == 1) & (d.owned_houses >= 1)

        d.wants_to_buy = buy_decision.astype(np.int32)
        d.wants_to_sell = sell_decision.astype(np.int32)

        # 탐색 피로 회복 (비트리거 에이전트)
        d.search_fatigue[not_triggered] *= 0.9
