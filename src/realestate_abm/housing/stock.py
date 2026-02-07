"""주택 스톡 - NumPy SoA (기존 houses.py 포트)"""

import numpy as np


class HousingStock:
    """주택 스톡 관리"""

    def __init__(self, n: int, n_regions: int):
        self.n = n
        self.n_regions = n_regions

        # 기본 속성
        self.region = np.zeros(n, dtype=np.int32)
        self.price = np.zeros(n, dtype=np.float32)
        self.jeonse_price = np.zeros(n, dtype=np.float32)
        self.size = np.zeros(n, dtype=np.float32)
        self.building_age = np.zeros(n, dtype=np.int32)

        # 소유/상태
        self.owner_id = np.full(n, -1, dtype=np.int32)
        self.is_for_sale = np.zeros(n, dtype=np.int32)
        self.tenant_id = np.full(n, -1, dtype=np.int32)
        self.is_jeonse = np.zeros(n, dtype=np.int32)

        # 상태
        self.is_active = np.ones(n, dtype=np.int32)
        self.is_demolished = np.zeros(n, dtype=np.int32)
        self.condition = np.ones(n, dtype=np.float32)
        self.months_on_market = np.zeros(n, dtype=np.int32)

        # 월세
        self.monthly_rent = np.zeros(n, dtype=np.float32)
        self.wolse_deposit = np.zeros(n, dtype=np.float32)

        # 가격 이력
        self.price_history = np.zeros((n, 6), dtype=np.float32)

    def initialize(self, world, rng: np.random.Generator, active_ratio: float = 0.6):
        """초기화 (active_ratio만큼만 활성, 나머지는 신규 공급 버퍼)"""
        n = self.n
        n_active = int(n * active_ratio)

        # 비활성 슬롯 초기화 (공급 버퍼)
        self.is_active[:] = 0
        self.price[:] = 0
        self.owner_id[:] = -1

        # 활성 주택만 초기화
        active_idx = np.arange(n_active)
        self.is_active[active_idx] = 1

        # 지역 분포 (활성 주택만)
        self.region[active_idx] = rng.choice(world.n, size=n_active, p=world.household_ratio).astype(np.int32)

        # 가격 설정
        for i in range(world.n):
            mask = (self.region == i) & (self.is_active == 1)
            count = np.sum(mask)
            if count > 0:
                base = world.base_prices[i]
                self.price[mask] = rng.lognormal(np.log(base), 0.3, count).astype(np.float32)

        # 전세가
        self.jeonse_price[active_idx] = self.price[active_idx] * rng.uniform(0.6, 0.8, n_active).astype(np.float32)

        # 면적
        self.size[active_idx] = np.clip(rng.lognormal(np.log(25), 0.4, n_active), 10, 80).astype(np.float32)

        # 건물 연식
        self.building_age[active_idx] = rng.integers(0, 41, n_active, dtype=np.int32)

        # 매물
        self.is_for_sale[active_idx] = (rng.random(n_active) < 0.05).astype(np.int32)

        # 전세 여부
        self.is_jeonse[active_idx] = (rng.random(n_active) < 0.55).astype(np.int32)

        # 상태
        self.condition[active_idx] = np.clip(1.0 - self.building_age[active_idx] / 50.0 * 0.7, 0.3, 1.0).astype(np.float32)

        # 가격 이력
        self.price_history[:n_active] = np.tile(self.price[active_idx, None], (1, 6))

    def update_price_history(self):
        self.price_history[:, 1:] = self.price_history[:, :-1]
        self.price_history[:, 0] = self.price

    def update_depreciation(self, rate: float = 0.003, min_condition: float = 0.3):
        active = self.is_active == 1
        age_factor = 1.0 + self.building_age / 30.0
        self.condition[active] -= rate * age_factor[active]
        self.condition = np.clip(self.condition, min_condition, 1.0)

    def get_active_count_by_region(self) -> np.ndarray:
        counts = np.zeros(self.n_regions, dtype=np.int32)
        for r in range(self.n_regions):
            counts[r] = np.sum((self.region == r) & (self.is_active == 1))
        return counts

    def get_region_avg_prices(self) -> np.ndarray:
        avg = np.zeros(self.n_regions, dtype=np.float32)
        for r in range(self.n_regions):
            mask = (self.region == r) & (self.is_active == 1)
            if np.any(mask):
                avg[r] = np.mean(self.price[mask])
        return avg

    def mark_demolished(self, house_ids: np.ndarray, agents=None):
        if len(house_ids) == 0:
            return {'count': 0, 'owners_affected': 0}

        affected_owners = []
        for hid in house_ids:
            if self.owner_id[hid] >= 0:
                affected_owners.append(self.owner_id[hid])
            self.is_active[hid] = 0
            self.is_demolished[hid] = 1
            self.price[hid] = 0
            self.owner_id[hid] = -1
            self.tenant_id[hid] = -1
            self.is_for_sale[hid] = 0

        if agents is not None and affected_owners:
            for owner in affected_owners:
                agents.data.owned_houses[owner] = max(0, agents.data.owned_houses[owner] - 1)

        return {'count': len(house_ids), 'owners_affected': len(affected_owners)}
