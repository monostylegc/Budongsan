"""통화정책 - Taylor Rule (기존 macro.py 포트)"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MonetaryState:
    policy_rate: float = 0.035
    mortgage_rate: float = 0.05
    inflation: float = 0.02
    credit_spread: float = 0.015


class MonetaryPolicy:
    """통화정책 (Taylor Rule)"""

    def __init__(self, cfg):
        """cfg: MonetaryConfig"""
        self.cfg = cfg
        self.state = MonetaryState(
            policy_rate=cfg.interest_rate,
            mortgage_rate=cfg.interest_rate + cfg.mortgage_spread,
            credit_spread=cfg.mortgage_spread,
        )

    def taylor_rule(self, inflation: float, output_gap: float) -> float:
        cfg = self.cfg
        target = (cfg.neutral_real_rate + cfg.inflation_target +
                  cfg.alpha_inflation * (inflation - cfg.inflation_target) +
                  cfg.alpha_output * output_gap)
        return np.clip(target, 0.0, 0.15)

    def step(self, inflation: float, output_gap: float):
        target = self.taylor_rule(inflation, output_gap)
        self.state.policy_rate = 0.9 * self.state.policy_rate + 0.1 * target
        self.state.inflation = inflation
        self.state.mortgage_rate = self.state.policy_rate + self.state.credit_spread

    def get_mortgage_rate(self) -> float:
        return self.state.mortgage_rate

    def get_jeonse_conversion_rate(self) -> float:
        return self.state.policy_rate + 0.02

    def reset(self):
        self.state = MonetaryState(
            policy_rate=self.cfg.interest_rate,
            mortgage_rate=self.cfg.interest_rate + self.cfg.mortgage_spread,
            credit_spread=self.cfg.mortgage_spread,
        )
