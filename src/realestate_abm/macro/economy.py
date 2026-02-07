"""거시경제 모델 - GDP, 인플레이션, 통화량"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MacroState:
    """거시경제 상태"""
    gdp_growth: float = 0.025
    inflation: float = 0.02
    output_gap: float = 0.0
    m2_growth: float = 0.06
    potential_gdp_growth: float = 0.025


class MacroEconomy:
    """거시경제 모델"""

    def __init__(self, cfg):
        """cfg: MacroEconomyConfig"""
        self.cfg = cfg
        self.state = MacroState(
            gdp_growth=cfg.initial_gdp_growth,
            inflation=cfg.initial_inflation,
        )
        self.history: list[dict] = []

    def step(self, housing_price_change: float, rng: np.random.Generator):
        """월간 거시경제 업데이트

        Args:
            housing_price_change: 전국 평균 주택 가격 변화율
            rng: 난수 생성기
        """
        cfg = self.cfg
        s = self.state

        # GDP growth - AR(1) with mean reversion
        shock = rng.normal(0, cfg.gdp_growth_volatility)
        s.gdp_growth = (cfg.gdp_growth_persistence * s.gdp_growth +
                        (1 - cfg.gdp_growth_persistence) * cfg.gdp_growth_mean +
                        shock)
        s.gdp_growth = np.clip(s.gdp_growth, -0.05, 0.10)

        # Output gap
        s.output_gap = s.gdp_growth - s.potential_gdp_growth

        # Inflation - Phillips curve style
        inflation_inertia = 0.7 * s.inflation
        demand_pull = 0.3 * max(0, s.output_gap)
        asset_effect = 0.1 * max(0, housing_price_change)
        supply_shock = rng.normal(0, 0.001)
        s.inflation = np.clip(
            inflation_inertia + demand_pull + asset_effect + supply_shock + 0.005,
            0.0, 0.15
        )

        # M2 growth
        s.m2_growth = 0.8 * s.m2_growth + 0.2 * (s.gdp_growth + s.inflation + 0.02)
        s.m2_growth = np.clip(s.m2_growth, 0.0, 0.20)

        self.history.append(self.get_state_dict())

    def get_income_growth_factor(self) -> float:
        """소득 성장률 팩터"""
        return 1.0 + self.cfg.income_gdp_beta * self.state.gdp_growth / 12.0

    def get_state_dict(self) -> dict:
        s = self.state
        return {
            'gdp_growth': s.gdp_growth,
            'inflation': s.inflation,
            'output_gap': s.output_gap,
            'm2_growth': s.m2_growth,
        }

    def reset(self):
        self.state = MacroState(
            gdp_growth=self.cfg.initial_gdp_growth,
            inflation=self.cfg.initial_inflation,
        )
        self.history = []
