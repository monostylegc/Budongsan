"""인지 아키텍처 통합 - 4단계 파이프라인

매 스텝마다:
[인지] → [감정] → [사고] → [행동]
"""

import numpy as np
from .perception import PerceptionEngine
from .emotions import EmotionEngine
from .thinking import ThinkingEngine
from .decisions import DecisionEngine
from .memory import MemoryEngine


class CognitiveEngine:
    """4단계 인지 파이프라인 통합 엔진"""

    def __init__(self, cfg):
        """cfg: CognitiveConfig"""
        self.perception = PerceptionEngine(cfg.perception)
        self.emotions = EmotionEngine(cfg.emotion)
        self.thinking = ThinkingEngine(cfg.thinking)
        self.decisions = DecisionEngine(cfg.decision)
        self.memory = MemoryEngine(cfg.memory)

    def step(
        self,
        agents,
        world,
        actual_prices: np.ndarray,
        price_changes: np.ndarray,
        interest_rate: float,
        rng: np.random.Generator,
    ):
        """한 스텝 인지 파이프라인 실행

        Args:
            agents: AgentPopulation
            world: RegionSet
            actual_prices: (n_regions,) 실제 가격
            price_changes: (n_regions,) 가격 변화율
            interest_rate: 현재 금리
            rng: 난수 생성기
        """
        d = agents.data
        n = agents.n

        # Phase 1: 인지 (제한된 정보 수집)
        self.perception.update(agents, world, actual_prices, rng)

        # Phase 2: 감정 (누적/감쇠)
        self.emotions.update(agents, actual_prices, price_changes)

        # Phase 3: 트리거 체크 (행동 개시 여부)
        self.decisions.check_triggers(agents, price_changes, rng)

        # Phase 3.5: 타겟 지역 선택 (트리거된 에이전트만)
        self._select_target_regions(agents, world, actual_prices, rng)

        # Phase 4: 사고 (System1 + System2 블렌딩)
        emotional_intensity = self.emotions.get_emotional_intensity(agents)

        # 타겟 지역의 알려진 가격
        target_regions = d.target_region
        known_target_prices = d.known_prices[np.arange(n), target_regions]

        buy_scores = self.thinking.evaluate_purchase(
            agents, target_regions, known_target_prices,
            actual_prices, price_changes,
            emotional_intensity, interest_rate,
        )

        current_region_prices = actual_prices[d.region]
        sell_scores = self.thinking.evaluate_sale(
            agents, current_region_prices, emotional_intensity,
        )

        # Phase 5: 의사결정 (만족화)
        self.decisions.apply_satisficing(agents, buy_scores, sell_scores, rng)

        # Phase 6: 경험 학습
        self.memory.update_from_experience(agents)

    def _select_target_regions(self, agents, world, actual_prices, rng):
        """트리거된 에이전트의 타겟 지역 선택

        가격 대비 소득, 인접도, 프리미엄, 직장 밀도를 종합 고려.
        """
        d = agents.data
        n = agents.n
        nr = world.n
        triggered = np.where(d.is_triggered == 1)[0]
        if len(triggered) == 0:
            return

        # 지역별 점수 계산 (모든 트리거 에이전트에 대해)
        home_regions = d.region[triggered]

        # 기본 점수: 인접도 (현재 지역 근처 선호)
        adjacency_scores = world.adjacency[home_regions]  # (n_triggered, nr)

        # 가격 부담 점수 (저렴할수록 높음)
        available = d.housing_fund[triggered] + d.investment_fund[triggered] * 0.5
        price_ratio = available[:, None] / np.maximum(actual_prices[None, :], 1)  # (n_trig, nr)
        affordability_scores = np.clip(price_ratio, 0, 2) * 0.3

        # 프리미엄 선호 (학교, 교통 등)
        prestige_scores = world.prestige[None, :] * 0.2  # (1, nr)

        # 직장 밀도
        job_scores = world.job_density[None, :] * 0.15

        # 종합 (개인 특성 가중)
        total = (adjacency_scores * 0.35 +
                 affordability_scores +
                 prestige_scores +
                 job_scores)

        # 무주택자: 현재 지역 선호 가산
        homeless = d.owned_houses[triggered] == 0
        for i, agent_i in enumerate(triggered):
            total[i, d.region[agent_i]] += 0.2 if homeless[i] else 0

        # 확률적 선택 (상위 3개 중 랜덤)
        for i, agent_i in enumerate(triggered):
            top3 = np.argsort(total[i])[-3:]
            weights = total[i, top3]
            weights = np.maximum(weights, 0.01)
            weights /= weights.sum()
            d.target_region[agent_i] = rng.choice(top3, p=weights)
