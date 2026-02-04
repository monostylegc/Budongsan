"""시장 메커니즘 (가격 결정, 매칭)"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, ADJACENCY


@ti.data_oriented
class Market:
    """부동산 시장"""

    def __init__(self, config: Config):
        self.config = config

        # 지역별 집계 데이터
        self.region_prices = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        self.region_jeonse_prices = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        self.region_price_changes = ti.field(dtype=ti.f32, shape=NUM_REGIONS)

        self.demand = ti.field(dtype=ti.i32, shape=NUM_REGIONS)
        self.supply = ti.field(dtype=ti.i32, shape=NUM_REGIONS)

        self.transactions = ti.field(dtype=ti.i32, shape=NUM_REGIONS)
        self.total_houses = ti.field(dtype=ti.i32, shape=NUM_REGIONS)

        # 가격 변화율 (업데이트용)
        self.price_change_rates = ti.field(dtype=ti.f32, shape=NUM_REGIONS)

        # 인접도 행렬 (풍선효과용)
        self.adjacency = ti.field(dtype=ti.f32, shape=(NUM_REGIONS, NUM_REGIONS))

        # 이력 저장 (시각화용)
        self.price_history = []
        self.transaction_history = []
        self.jeonse_ratio_history = []

    def initialize(self):
        """초기화"""
        self.adjacency.from_numpy(ADJACENCY)
        self.demand.fill(0)
        self.supply.fill(0)
        self.transactions.fill(0)

    def aggregate_prices(self, houses):
        """지역별 평균 가격 계산 (NumPy 사용)"""
        regions = houses.region.to_numpy()
        prices = houses.price.to_numpy()
        jeonse_prices = houses.jeonse_price.to_numpy()

        old_prices = self.region_prices.to_numpy()

        # 지역별 집계
        avg_prices = np.zeros(NUM_REGIONS, dtype=np.float32)
        avg_jeonse = np.zeros(NUM_REGIONS, dtype=np.float32)
        counts = np.zeros(NUM_REGIONS, dtype=np.int32)

        for r in range(NUM_REGIONS):
            mask = regions == r
            count = np.sum(mask)
            counts[r] = count
            if count > 0:
                avg_prices[r] = np.mean(prices[mask])
                avg_jeonse[r] = np.mean(jeonse_prices[mask])

        # 가격 변화율
        changes = np.zeros(NUM_REGIONS, dtype=np.float32)
        valid = old_prices > 0
        changes[valid] = (avg_prices[valid] - old_prices[valid]) / old_prices[valid]

        self.region_prices.from_numpy(avg_prices)
        self.region_jeonse_prices.from_numpy(avg_jeonse)
        self.region_price_changes.from_numpy(changes)
        self.total_houses.from_numpy(counts)

    def count_demand_supply(self, households, houses):
        """수요/공급 집계 (NumPy 사용)"""
        wants_buy = households.wants_to_buy.to_numpy()
        target_regions = households.target_region.to_numpy()
        is_for_sale = houses.is_for_sale.to_numpy()
        house_regions = houses.region.to_numpy()

        demand = np.zeros(NUM_REGIONS, dtype=np.int32)
        supply = np.zeros(NUM_REGIONS, dtype=np.int32)

        for r in range(NUM_REGIONS):
            demand[r] = np.sum((wants_buy == 1) & (target_regions == r))
            supply[r] = np.sum((is_for_sale == 1) & (house_regions == r))

        self.demand.from_numpy(demand)
        self.supply.from_numpy(supply)

    def update_prices(self, houses, sensitivity: float, expectation_weight: float):
        """가격 업데이트 (수요/공급 + 기대 + 풍선효과)"""
        demand_np = self.demand.to_numpy().astype(np.float32)
        supply_np = self.supply.to_numpy().astype(np.float32)
        total_np = self.total_houses.to_numpy().astype(np.float32)
        current_prices = self.region_prices.to_numpy()

        # 수요/공급 비율 (전체 주택수 대비)
        supply_np = np.maximum(supply_np, total_np * 0.02)  # 최소 2% 매물
        ds_ratio = demand_np / supply_np

        # 수요/공급 기반 가격 변화 (log scale로 완화)
        base_change = sensitivity * np.log1p(ds_ratio - 1.0)

        # 기대 효과
        price_changes = self.region_price_changes.to_numpy()
        expectation_effect = expectation_weight * price_changes

        # 풍선효과 (지역 간 격차에 따른 조정)
        adjacency = self.adjacency.to_numpy()
        spillover = self.config.spillover_rate * (adjacency @ price_changes)

        # 기본 가격 상승 (인플레이션 등) - 지역별 차등 적용
        # 고가 지역(서울)은 낮은 상승률, 저가 지역은 더 낮은 상승률 적용
        base_appreciation = self.config.base_appreciation

        # 지역별 기본 상승률 조정 (tier 기반)
        # 서울 핵심: 1.5x, 서울: 1.2x, 수도권: 1.0x, 지방광역시: 0.6x, 기타지방: 0.3x
        tier_multipliers = np.array([
            1.5, 1.3, 1.1,  # 서울 (강남, 마용성, 기타서울)
            1.2, 0.9, 0.7, 0.8,  # 수도권 (분당, 경기남부, 경기북부, 인천)
            0.5, 0.4, 0.4, 0.45, 0.6,  # 지방광역시 (부산, 대구, 광주, 대전, 세종)
            0.2,  # 기타지방
        ], dtype=np.float32)

        regional_appreciation = base_appreciation * tier_multipliers

        # 가격 수준에 따른 추가 조정 (평균대비 높으면 상승률 하락)
        avg_price = np.mean(current_prices[current_prices > 0])
        price_adjustment = np.where(
            current_prices > 0,
            np.clip(1.0 - (current_prices / avg_price - 1.0) * 0.3, 0.5, 1.5),
            1.0
        )

        # 최종 가격 변화율
        total_change = regional_appreciation * price_adjustment + base_change + expectation_effect + spillover

        # 월 최대 변화율 제한 (±1.5%)
        total_change = np.clip(total_change, -0.015, 0.015)

        # 필드에 저장
        self.price_change_rates.from_numpy(total_change.astype(np.float32))

        # 개별 주택 가격 업데이트 (Taichi 커널)
        self._apply_price_changes(houses.region, houses.price, houses.jeonse_price, houses.n)

    @ti.kernel
    def _apply_price_changes(
        self,
        regions: ti.template(),
        prices: ti.template(),
        jeonse_prices: ti.template(),
        n: ti.i32
    ):
        for i in range(n):
            region = regions[i]
            change = self.price_change_rates[region]
            prices[i] *= (1.0 + change)
            jeonse_prices[i] *= (1.0 + change * 0.8)

    def simple_matching(self, households, houses, rng: np.random.Generator):
        """단순 매칭 (NumPy 사용) - 하위 호환용"""
        self.enhanced_matching(households, houses, rng, 0)

    def enhanced_matching(self, households, houses, rng: np.random.Generator, current_month: int):
        """향상된 매칭 (매입가 기록 포함)

        행동경제학 요소를 위한 매입가 기록:
        - 손실 회피, 앵커링 계산에 사용
        """
        wants_buy = households.wants_to_buy.to_numpy()
        target_regions = households.target_region.to_numpy()
        buyer_assets = households.asset.to_numpy()
        owned_houses = households.owned_houses.to_numpy()

        is_for_sale = houses.is_for_sale.to_numpy()
        house_regions = houses.region.to_numpy()
        house_prices = houses.price.to_numpy()
        house_owners = houses.owner_id.to_numpy()

        # 매입가 기록용
        purchase_prices = households.purchase_price.to_numpy()
        purchase_months = households.purchase_month.to_numpy()
        total_purchase_prices = households.total_purchase_price.to_numpy()

        transactions = np.zeros(NUM_REGIONS, dtype=np.int32)

        for region in range(NUM_REGIONS):
            buyers = np.where((wants_buy == 1) & (target_regions == region))[0]
            sellers = np.where((is_for_sale == 1) & (house_regions == region))[0]

            if len(buyers) == 0 or len(sellers) == 0:
                continue

            rng.shuffle(buyers)
            rng.shuffle(sellers)

            matched = 0
            for buyer_id in buyers:
                if matched >= len(sellers):
                    break

                house_id = sellers[matched]
                price = house_prices[house_id]

                # 구매력 체크 (자산 50% + 대출 50%로 구매 가능)
                if buyer_assets[buyer_id] * 0.5 >= price * 0.3:  # 자산이 가격의 30% 이상
                    # 거래 성사
                    old_owner = house_owners[house_id]

                    is_for_sale[house_id] = 0
                    wants_buy[buyer_id] = 0
                    house_owners[house_id] = buyer_id
                    owned_houses[buyer_id] += 1

                    # 매수자 매입가 기록
                    if owned_houses[buyer_id] == 1:  # 첫 주택
                        purchase_prices[buyer_id] = price
                    total_purchase_prices[buyer_id] += price
                    purchase_months[buyer_id] = current_month

                    # 매도자 처리
                    if old_owner >= 0:
                        owned_houses[old_owner] -= 1
                        # 매도자 매입가 차감 (평균값 기준)
                        if owned_houses[old_owner] > 0:
                            avg_purchase = total_purchase_prices[old_owner] / (owned_houses[old_owner] + 1)
                            total_purchase_prices[old_owner] -= avg_purchase
                        else:
                            # 무주택자가 됨
                            purchase_prices[old_owner] = 0.0
                            total_purchase_prices[old_owner] = 0.0

                    matched += 1
                    transactions[region] += 1

        # 결과 반영
        houses.is_for_sale.from_numpy(is_for_sale.astype(np.int32))
        houses.owner_id.from_numpy(house_owners.astype(np.int32))
        households.wants_to_buy.from_numpy(wants_buy.astype(np.int32))
        households.owned_houses.from_numpy(owned_houses.astype(np.int32))
        households.purchase_price.from_numpy(purchase_prices.astype(np.float32))
        households.purchase_month.from_numpy(purchase_months.astype(np.int32))
        households.total_purchase_price.from_numpy(total_purchase_prices.astype(np.float32))
        self.transactions.from_numpy(transactions)

    def record_history(self):
        """이력 기록"""
        self.price_history.append(self.region_prices.to_numpy().copy())
        self.transaction_history.append(self.transactions.to_numpy().copy())

        prices = self.region_prices.to_numpy()
        jeonse = self.region_jeonse_prices.to_numpy()
        ratio = np.divide(jeonse, prices, where=prices > 0, out=np.zeros_like(prices))
        self.jeonse_ratio_history.append(ratio.copy())
