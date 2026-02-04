"""주택 데이터 (Taichi fields)"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, REGIONS, REGION_HOUSEHOLD_RATIO


@ti.data_oriented
class Houses:
    """주택들을 관리하는 클래스"""

    def __init__(self, config: Config):
        self.n = config.num_houses
        self.config = config

        # 기본 속성
        self.region = ti.field(dtype=ti.i32, shape=self.n)
        self.price = ti.field(dtype=ti.f32, shape=self.n)         # 매매가 (만원)
        self.jeonse_price = ti.field(dtype=ti.f32, shape=self.n)  # 전세가 (만원)
        self.size = ti.field(dtype=ti.f32, shape=self.n)          # 전용면적 (평)
        self.building_age = ti.field(dtype=ti.i32, shape=self.n)  # 건물 연식

        # 소유 및 상태
        self.owner_id = ti.field(dtype=ti.i32, shape=self.n)      # 소유자 ID (-1이면 미분양)
        self.is_for_sale = ti.field(dtype=ti.i32, shape=self.n)   # 매물 여부
        self.tenant_id = ti.field(dtype=ti.i32, shape=self.n)     # 세입자 ID (-1이면 없음)
        self.is_jeonse = ti.field(dtype=ti.i32, shape=self.n)     # 전세 여부 (0: 자가/월세, 1: 전세)

        # 가격 이력 (최근 6개월)
        self.price_history = ti.field(dtype=ti.f32, shape=(self.n, 6))

        # 거래
        self.months_on_market = ti.field(dtype=ti.i32, shape=self.n)  # 매물 등록 후 경과 월

    def initialize(self, rng: np.random.Generator):
        """초기 상태 설정"""
        # 지역 분포 (가구 분포와 유사하게)
        regions = rng.choice(NUM_REGIONS, size=self.n, p=REGION_HOUSEHOLD_RATIO).astype(np.int32)

        # 가격 설정 (지역 기준가 기반 + 변동)
        prices = np.zeros(self.n, dtype=np.float32)
        for region_id, info in REGIONS.items():
            mask = regions == region_id
            count = np.sum(mask)
            if count > 0:
                base = info["base_price"]
                # 로그정규 분포로 가격 변동
                prices[mask] = rng.lognormal(
                    mean=np.log(base),
                    sigma=0.3,
                    size=count
                ).astype(np.float32)

        # 전세가 (전세가율 60-80%)
        jeonse_ratio = rng.uniform(0.6, 0.8, size=self.n).astype(np.float32)
        jeonse_prices = prices * jeonse_ratio

        # 면적 (10-80평, 중위값 25평)
        sizes = rng.lognormal(mean=np.log(25), sigma=0.4, size=self.n).astype(np.float32)
        sizes = np.clip(sizes, 10, 80)

        # 건물 연식 (0-40년)
        building_ages = rng.integers(0, 41, size=self.n, dtype=np.int32)

        # 소유자 (-1로 초기화, 나중에 매칭)
        owners = np.full(self.n, -1, dtype=np.int32)
        tenants = np.full(self.n, -1, dtype=np.int32)

        # 매물 (초기 5%만 매물)
        for_sale = (rng.random(self.n) < 0.05).astype(np.int32)

        # 전세 여부 (55%가 전세)
        is_jeonse = (rng.random(self.n) < 0.55).astype(np.int32)

        # Taichi 필드에 복사
        self.region.from_numpy(regions)
        self.price.from_numpy(prices)
        self.jeonse_price.from_numpy(jeonse_prices)
        self.size.from_numpy(sizes)
        self.building_age.from_numpy(building_ages)
        self.owner_id.from_numpy(owners)
        self.is_for_sale.from_numpy(for_sale)
        self.tenant_id.from_numpy(tenants)
        self.is_jeonse.from_numpy(is_jeonse)
        self.months_on_market.from_numpy(np.zeros(self.n, dtype=np.int32))

        # 가격 이력 초기화
        price_hist = np.tile(prices.reshape(-1, 1), (1, 6))
        self.price_history.from_numpy(price_hist)

    @ti.kernel
    def update_building_age(self):
        """건물 연식 증가 (연 1회 호출)"""
        for i in range(self.n):
            self.building_age[i] += 1

    @ti.kernel
    def update_price_history(self):
        """가격 이력 업데이트 (월 1회)"""
        for i in range(self.n):
            # 이력 시프트 (5->4->3->2->1->0)
            self.price_history[i, 5] = self.price_history[i, 4]
            self.price_history[i, 4] = self.price_history[i, 3]
            self.price_history[i, 3] = self.price_history[i, 2]
            self.price_history[i, 2] = self.price_history[i, 1]
            self.price_history[i, 1] = self.price_history[i, 0]
            self.price_history[i, 0] = self.price[i]

    @ti.kernel
    def count_for_sale_by_region(self, counts: ti.template()):
        """지역별 매물 수 계산"""
        for i in range(self.n):
            if self.is_for_sale[i] == 1:
                ti.atomic_add(counts[self.region[i]], 1)

    @ti.kernel
    def count_by_region(self, counts: ti.template()):
        """지역별 주택 수 계산"""
        for i in range(self.n):
            ti.atomic_add(counts[self.region[i]], 1)

    @ti.kernel
    def sum_prices_by_region(self, sums: ti.template(), counts: ti.template()):
        """지역별 가격 합계 및 개수"""
        for i in range(self.n):
            region = self.region[i]
            ti.atomic_add(sums[region], self.price[i])
            ti.atomic_add(counts[region], 1)

    @ti.kernel
    def update_months_on_market(self):
        """매물 등록 기간 업데이트"""
        for i in range(self.n):
            if self.is_for_sale[i] == 1:
                self.months_on_market[i] += 1
            else:
                self.months_on_market[i] = 0
