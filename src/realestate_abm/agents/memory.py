"""기억과 학습 - 과거 결정/결과 기억

과거 매수 후 가격↑ → 자신감↑, 더 적극적
과거 매수 후 가격↓ → 후회↑, 더 신중
"""

import numpy as np


class MemoryEngine:
    """기억과 학습 엔진"""

    def __init__(self, cfg):
        """cfg: MemoryConfig"""
        self.cfg = cfg

    def record_decision(self, agents, agent_ids: np.ndarray, decision_type: int):
        """결정 기록"""
        d = agents.data
        for aid in agent_ids:
            count = d.decision_count[aid]
            idx = count % self.cfg.max_decisions
            d.past_decisions[aid, idx] = decision_type
            d.decision_count[aid] += 1

    def record_outcome(self, agents, agent_ids: np.ndarray, outcomes: np.ndarray):
        """결과 기록 (수익/손실)"""
        d = agents.data
        for i, aid in enumerate(agent_ids):
            count = d.decision_count[aid]
            idx = (count - 1) % self.cfg.max_decisions
            d.past_outcomes[aid, idx] = outcomes[i]

    def update_from_experience(self, agents):
        """경험에서 학습 → 감정/정보품질 업데이트"""
        d = agents.data
        cfg = self.cfg

        has_experience = d.decision_count > 0

        if not np.any(has_experience):
            return

        exp_idx = np.where(has_experience)[0]

        # 최근 결과의 가중 평균
        recent_outcomes = d.past_outcomes[exp_idx]
        weights = np.array([0.3, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01])
        counts = np.minimum(d.decision_count[exp_idx], self.cfg.max_decisions)

        avg_outcome = np.zeros(len(exp_idx), dtype=np.float32)
        for i, idx in enumerate(exp_idx):
            c = int(counts[i])
            if c > 0:
                w = weights[:c]
                w = w / w.sum()
                avg_outcome[i] = np.sum(w * recent_outcomes[i, :c])

        # 좋은 결과 → 후회 감소, 정보 품질 증가
        good = avg_outcome > 0
        d.regret[exp_idx[good]] = np.clip(
            d.regret[exp_idx[good]] - cfg.experience_learning_rate, 0, 1
        )

        # 나쁜 결과 → 후회 증가
        bad = avg_outcome < 0
        d.regret[exp_idx[bad]] = np.clip(
            d.regret[exp_idx[bad]] + cfg.experience_learning_rate, 0, 1
        )

        # 경험에 따른 정보 품질 증가
        d.info_quality[exp_idx] = np.clip(
            d.info_quality[exp_idx] + cfg.info_quality_growth, 0, 1
        )
