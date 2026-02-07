"""주택 공급 모델 (기존 supply.py 포트)"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ConstructionProject:
    region: int
    num_units: int
    start_month: int
    completion_month: int
    is_redevelopment: bool = False


class SupplyModel:
    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        self.elasticity = world.supply_elasticity.copy()
        self.pipeline: list[ConstructionProject] = []
        self.history = []

    def step(self, houses, price_changes_12m: np.ndarray, current_stock: np.ndarray,
             current_month: int, rng: np.random.Generator) -> dict:
        # Complete construction
        completed = self._complete_construction(houses, current_month, rng)
        # New supply
        new_supply = self._calculate_new_supply(price_changes_12m, current_stock)
        self._add_construction(new_supply, current_month)

        stats = {'completed': int(completed.sum()), 'new_starts': int(new_supply.sum()),
                 'under_construction': sum(p.num_units for p in self.pipeline)}
        self.history.append(stats)
        return stats

    def _calculate_new_supply(self, price_changes: np.ndarray, stock: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ratio = np.clip(price_changes / cfg.price_threshold, 0, 5)
        rate = cfg.base_supply_rate * np.power(ratio + 1, self.elasticity)
        rate = np.clip(rate, 0, cfg.max_construction_ratio)
        return (stock * rate).astype(np.int32)

    def _add_construction(self, supply: np.ndarray, month: int):
        for r in range(len(supply)):
            if supply[r] > 0:
                self.pipeline.append(ConstructionProject(
                    region=r, num_units=supply[r],
                    start_month=month, completion_month=month + self.cfg.construction_period,
                ))

    def _complete_construction(self, houses, month: int, rng) -> np.ndarray:
        completed = np.zeros(self.world.n, dtype=np.int32)
        done = [p for p in self.pipeline if p.completion_month <= month]
        for p in done:
            completed[p.region] += p.num_units
            self.pipeline.remove(p)
            self._add_houses(houses, p.region, p.num_units, rng)
        return completed

    def _add_houses(self, houses, region: int, num: int, rng):
        empty = np.where((houses.owner_id == -1) & (houses.price == 0) & (houses.is_active == 0))[0]
        num = min(num, len(empty))
        if num == 0:
            return
        slots = empty[:num]
        base = self.world.base_prices[region]
        houses.price[slots] = base * rng.uniform(1.0, 1.2, num)
        houses.jeonse_price[slots] = houses.price[slots] * rng.uniform(0.6, 0.75, num)
        houses.region[slots] = region
        houses.building_age[slots] = 0
        houses.size[slots] = rng.uniform(20, 40, num)
        houses.is_for_sale[slots] = 1
        houses.owner_id[slots] = -1
        houses.is_active[slots] = 1
        houses.condition[slots] = 1.0

    def reset(self):
        self.pipeline = []
        self.history = []
