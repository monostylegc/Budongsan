"""가격 결정 엔진 (기존 market.py의 update_prices 포트)"""

import numpy as np


class PriceEngine:
    """수급 기반 가격 결정"""

    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        self.historical_avg_prices = np.zeros(world.n, dtype=np.float32)
        self.dynamic_prestige = world.prestige.copy()
        self.prestige_momentum = np.zeros(world.n, dtype=np.float32)

    def update_prices(self, houses, demand: np.ndarray, supply: np.ndarray,
                       total_houses: np.ndarray, price_changes: np.ndarray):
        """가격 업데이트"""
        cfg = self.cfg
        world = self.world
        nr = world.n
        current_prices = houses.get_region_avg_prices()

        # 역사적 평균 업데이트
        if self.historical_avg_prices.sum() == 0:
            self.historical_avg_prices = current_prices.copy()
        else:
            self.historical_avg_prices = 0.98 * self.historical_avg_prices + 0.02 * current_prices

        # 수요/공급 비율
        effective_supply = np.maximum(supply, total_houses * 0.02)
        ds_ratio = demand.astype(np.float32) / effective_supply.astype(np.float32)
        excess_demand = ds_ratio - 1.0
        ds_change = cfg.price_sensitivity * np.clip(excess_demand, -2, 2)

        # 기대 효과
        expectation_effect = cfg.expectation_weight * price_changes

        # 풍선효과
        spillover = cfg.spillover_rate * (world.adjacency @ price_changes)

        # 기본 상승률 (tier 기반)
        tier_mult = np.zeros(nr, dtype=np.float32)
        for i in range(nr):
            t = world.tiers[i]
            if t == 1: tier_mult[i] = 1.3
            elif t == 2: tier_mult[i] = 0.8
            elif t == 3: tier_mult[i] = 0.3
            else: tier_mult[i] = 0.1

        appreciation_factor = np.clip((ds_ratio - 0.8) / 0.4, 0, 1)
        regional_appreciation = cfg.base_appreciation * tier_mult * appreciation_factor

        # 인플레이션 기대
        inflation_factor = tier_mult * 0.5
        regional_appreciation += 0.0017 * inflation_factor

        # 역사적 평균 회귀
        price_to_hist = np.divide(current_prices, self.historical_avg_prices,
                                   where=self.historical_avg_prices > 0,
                                   out=np.ones_like(current_prices))
        price_adj = np.clip(1.15 - (price_to_hist - 1.0) * 1.5, 0.5, 1.15)

        # 프리미엄 기반 지지/억제
        prestige = self.dynamic_prestige
        ds_change_supported = np.where(
            ds_change < 0,
            ds_change * (1.0 - prestige * 0.85),
            ds_change * np.clip(prestige ** 1.5, 0.1, 1.0),
        )

        # 최종
        total_change = (ds_change_supported * price_adj + regional_appreciation +
                        expectation_effect * 0.7 + spillover * 0.5)
        total_change = np.clip(total_change, -0.010, 0.010)

        # 개별 주택 가격 적용
        for r in range(nr):
            mask = (houses.region == r) & (houses.is_active == 1)
            houses.price[mask] *= (1.0 + total_change[r])
            houses.jeonse_price[mask] *= (1.0 + total_change[r] * 0.8)

        return total_change
