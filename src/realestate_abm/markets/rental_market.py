"""임대시장 (전세/월세)"""

import numpy as np


class RentalMarket:
    """전세/월세 시장"""

    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        self.jeonse_ratio = np.full(world.n, cfg.jeonse_ratio, dtype=np.float32)

    def update_conversion(self, houses, conversion_rate: float, rng):
        """전월세 전환"""
        threshold = 0.045
        for i in range(houses.n):
            if houses.owner_id[i] >= 0 and houses.is_active[i]:
                if conversion_rate > threshold:
                    if houses.is_jeonse[i] == 1 and rng.random() < 0.02:
                        houses.is_jeonse[i] = 0
                else:
                    if houses.is_jeonse[i] == 0 and rng.random() < 0.01:
                        houses.is_jeonse[i] = 1

        # 월세 재계산
        wolse = houses.is_jeonse == 0
        houses.wolse_deposit[wolse] = houses.jeonse_price[wolse] * 0.3
        houses.monthly_rent[wolse] = (houses.jeonse_price[wolse] - houses.wolse_deposit[wolse]) * conversion_rate / 12

    def reset(self):
        self.jeonse_ratio = np.full(self.world.n, self.cfg.jeonse_ratio, dtype=np.float32)
