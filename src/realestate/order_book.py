"""호가 기반 거래 시스템 (Order Book) - Double Auction 메커니즘"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS


# 최대 호가 수
MAX_ORDERS = 100000


@ti.data_oriented
class OrderBook:
    """호가 기반 주문장 클래스 (Double Auction)

    학술적 근거:
    - Genesove & Mayer (2001, QJE): 손실 상황 매도자 호가 프리미엄 25-35%
    - 실제 부동산 시장의 호가-낙찰 메커니즘 반영

    구조:
    - 매수 호가 (Bid): 매수자의 최대 지불 의사 가격
    - 매도 호가 (Ask): 매도자의 최소 수취 의사 가격
    - 체결: bid >= ask 조건에서 매칭
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_orders = MAX_ORDERS

        # === 매수 호가 (Bid Orders) ===
        self.bid_price = ti.field(dtype=ti.f32, shape=MAX_ORDERS)
        self.bid_agent_id = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.bid_region = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.bid_active = ti.field(dtype=ti.i32, shape=MAX_ORDERS)  # 0: inactive, 1: active

        # === 매도 호가 (Ask Orders) ===
        self.ask_price = ti.field(dtype=ti.f32, shape=MAX_ORDERS)
        self.ask_agent_id = ti.field(dtype=ti.i32, shape=MAX_ORDERS)  # 소유자 ID
        self.ask_house_id = ti.field(dtype=ti.i32, shape=MAX_ORDERS)  # 매물 ID
        self.ask_region = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.ask_active = ti.field(dtype=ti.i32, shape=MAX_ORDERS)

        # 호가 협상 마진
        self.negotiation_margin_min = 0.03  # 최소 협상 마진 (3%)
        self.negotiation_margin_max = 0.10  # 최대 협상 마진 (10%)

        # 손실 프리미엄 (Genesove & Mayer)
        self.loss_premium_min = 0.25  # 손실 시 최소 프리미엄 (25%)
        self.loss_premium_max = 0.35  # 손실 시 최대 프리미엄 (35%)

        # 지역별 호가 카운트
        self.bid_count = ti.field(dtype=ti.i32, shape=NUM_REGIONS)
        self.ask_count = ti.field(dtype=ti.i32, shape=NUM_REGIONS)

        # 체결 기록
        self.matched_buyer = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.matched_seller = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.matched_house = ti.field(dtype=ti.i32, shape=MAX_ORDERS)
        self.matched_price = ti.field(dtype=ti.f32, shape=MAX_ORDERS)
        self.num_matched = ti.field(dtype=ti.i32, shape=())

    def clear_orders(self):
        """호가장 초기화"""
        self._clear_orders_kernel()

    @ti.kernel
    def _clear_orders_kernel(self):
        for i in range(self.max_orders):
            self.bid_active[i] = 0
            self.ask_active[i] = 0
            self.matched_buyer[i] = -1
            self.matched_seller[i] = -1
            self.matched_house[i] = -1
            self.matched_price[i] = 0.0

        for r in range(NUM_REGIONS):
            self.bid_count[r] = 0
            self.ask_count[r] = 0

        self.num_matched[None] = 0

    def generate_bid_orders(
        self,
        households,
        region_prices: np.ndarray,
        rng: np.random.Generator
    ):
        """매수 호가 생성 (Willingness to Pay)

        호가 = 인지된 가치 * (1 - 협상 마진)
        인지된 가치 = 시장가 * (1 + 기대 * 위험허용도)
        """
        wants_to_buy = households.wants_to_buy.to_numpy()
        target_region = households.target_region.to_numpy()
        expectations = households.price_expectation.to_numpy()
        risk_tolerance = households.risk_tolerance.to_numpy()
        assets = households.asset.to_numpy()

        # 매수 희망자 필터링
        buyers = np.where(wants_to_buy == 1)[0]

        if len(buyers) == 0:
            return

        # 호가 제한
        num_bids = min(len(buyers), MAX_ORDERS)
        buyers = buyers[:num_bids]

        # 협상 마진 생성
        negotiation_margin = rng.uniform(
            self.negotiation_margin_min,
            self.negotiation_margin_max,
            size=num_bids
        )

        # 호가 계산
        regions = target_region[buyers]
        base_prices = region_prices[regions]

        # 인지된 가치 = 시장가 * (1 + 기대 * 위험허용도)
        exp = expectations[buyers]
        risk = risk_tolerance[buyers]
        perceived_value = base_prices * (1.0 + exp * risk * 0.2)

        # 매수 호가 = 인지된 가치 * (1 - 협상 마진)
        bid_prices = perceived_value * (1.0 - negotiation_margin)

        # 구매력 제한 적용
        max_bid = assets[buyers] * 0.8  # 자산의 80%까지
        bid_prices = np.minimum(bid_prices, max_bid)
        bid_prices = bid_prices.astype(np.float32)

        # Taichi 필드에 저장
        self.bid_price.from_numpy(np.pad(bid_prices, (0, MAX_ORDERS - num_bids), constant_values=0))
        self.bid_agent_id.from_numpy(np.pad(buyers.astype(np.int32), (0, MAX_ORDERS - num_bids), constant_values=-1))
        self.bid_region.from_numpy(np.pad(regions.astype(np.int32), (0, MAX_ORDERS - num_bids), constant_values=-1))
        self.bid_active.from_numpy(np.pad(np.ones(num_bids, dtype=np.int32), (0, MAX_ORDERS - num_bids), constant_values=0))

        # 지역별 매수 호가 수 집계
        bid_counts = np.bincount(regions, minlength=NUM_REGIONS).astype(np.int32)
        self.bid_count.from_numpy(bid_counts)

    def generate_ask_orders(
        self,
        households,
        houses,
        region_prices: np.ndarray,
        rng: np.random.Generator
    ):
        """매도 호가 생성 (Reservation Price)

        손실 상황: ask = 매입가 * (1 + 손실 프리미엄) [Genesove & Mayer 2001]
        이익 상황: ask = 시장가 * (1 - 긴급도 할인)
        """
        is_for_sale = houses.is_for_sale.to_numpy()
        house_prices = houses.price.to_numpy()
        house_owners = houses.owner_id.to_numpy()
        house_regions = houses.region.to_numpy()

        # 매물 필터링
        for_sale_indices = np.where(is_for_sale == 1)[0]

        if len(for_sale_indices) == 0:
            return

        # 호가 제한
        num_asks = min(len(for_sale_indices), MAX_ORDERS)
        for_sale_indices = for_sale_indices[:num_asks]

        owners = house_owners[for_sale_indices]
        regions = house_regions[for_sale_indices]
        current_prices = house_prices[for_sale_indices]

        # 소유자 정보 가져오기
        valid_owners_mask = owners >= 0
        valid_indices = for_sale_indices[valid_owners_mask]
        valid_owners = owners[valid_owners_mask]
        valid_regions = regions[valid_owners_mask]
        valid_current = current_prices[valid_owners_mask]

        if len(valid_owners) == 0:
            return

        purchase_prices = households.purchase_price.to_numpy()[valid_owners]
        loss_aversion = households.loss_aversion.to_numpy()[valid_owners]

        # 손익 계산
        gain_loss_ratio = np.zeros(len(valid_owners), dtype=np.float32)
        mask = purchase_prices > 0
        gain_loss_ratio[mask] = (valid_current[mask] - purchase_prices[mask]) / purchase_prices[mask]

        # 호가 결정
        ask_prices = np.zeros(len(valid_owners), dtype=np.float32)

        # 손실 상황: 매입가 + 손실 프리미엄
        loss_mask = gain_loss_ratio < 0
        if np.any(loss_mask):
            loss_premium = rng.uniform(
                self.loss_premium_min,
                self.loss_premium_max,
                size=np.sum(loss_mask)
            )
            # 손실 회피 성향에 비례하여 프리미엄 증가
            loss_premium *= loss_aversion[loss_mask] / 2.5
            ask_prices[loss_mask] = purchase_prices[loss_mask] * (1.0 + loss_premium)

        # 이익 상황: 시장가 기준
        gain_mask = ~loss_mask
        if np.any(gain_mask):
            urgency_discount = rng.uniform(0.0, 0.05, size=np.sum(gain_mask))
            ask_prices[gain_mask] = valid_current[gain_mask] * (1.0 - urgency_discount)

        ask_prices = ask_prices.astype(np.float32)

        # Taichi 필드에 저장
        num_valid = len(valid_owners)
        self.ask_price.from_numpy(np.pad(ask_prices, (0, MAX_ORDERS - num_valid), constant_values=0))
        self.ask_agent_id.from_numpy(np.pad(valid_owners.astype(np.int32), (0, MAX_ORDERS - num_valid), constant_values=-1))
        self.ask_house_id.from_numpy(np.pad(valid_indices.astype(np.int32), (0, MAX_ORDERS - num_valid), constant_values=-1))
        self.ask_region.from_numpy(np.pad(valid_regions.astype(np.int32), (0, MAX_ORDERS - num_valid), constant_values=-1))
        self.ask_active.from_numpy(np.pad(np.ones(num_valid, dtype=np.int32), (0, MAX_ORDERS - num_valid), constant_values=0))

        # 지역별 매도 호가 수 집계
        ask_counts = np.bincount(valid_regions, minlength=NUM_REGIONS).astype(np.int32)
        self.ask_count.from_numpy(ask_counts)

    def match_orders(self) -> int:
        """Double Auction 매칭

        지역별로:
        1. 매수 호가 내림차순 정렬
        2. 매도 호가 오름차순 정렬
        3. bid >= ask 조건에서 매칭
        4. 체결가 = (bid + ask) / 2

        Returns:
            총 체결 건수
        """
        # NumPy에서 처리 (정렬 필요)
        bid_prices = self.bid_price.to_numpy()
        bid_agents = self.bid_agent_id.to_numpy()
        bid_regions = self.bid_region.to_numpy()
        bid_active = self.bid_active.to_numpy()

        ask_prices = self.ask_price.to_numpy()
        ask_agents = self.ask_agent_id.to_numpy()
        ask_houses = self.ask_house_id.to_numpy()
        ask_regions = self.ask_region.to_numpy()
        ask_active = self.ask_active.to_numpy()

        matched_buyers = []
        matched_sellers = []
        matched_houses = []
        matched_prices = []

        for region in range(NUM_REGIONS):
            # 해당 지역 활성 호가 필터링
            region_bids = np.where((bid_regions == region) & (bid_active == 1))[0]
            region_asks = np.where((ask_regions == region) & (ask_active == 1))[0]

            if len(region_bids) == 0 or len(region_asks) == 0:
                continue

            # 매수 호가 내림차순 정렬
            bid_order = np.argsort(-bid_prices[region_bids])
            sorted_bids = region_bids[bid_order]

            # 매도 호가 오름차순 정렬
            ask_order = np.argsort(ask_prices[region_asks])
            sorted_asks = region_asks[ask_order]

            # 매칭
            bid_idx = 0
            ask_idx = 0

            while bid_idx < len(sorted_bids) and ask_idx < len(sorted_asks):
                b = sorted_bids[bid_idx]
                a = sorted_asks[ask_idx]

                if bid_prices[b] >= ask_prices[a]:
                    # 체결
                    execution_price = (bid_prices[b] + ask_prices[a]) / 2.0

                    matched_buyers.append(bid_agents[b])
                    matched_sellers.append(ask_agents[a])
                    matched_houses.append(ask_houses[a])
                    matched_prices.append(execution_price)

                    # 호가 비활성화
                    bid_active[b] = 0
                    ask_active[a] = 0

                    bid_idx += 1
                    ask_idx += 1
                else:
                    # 더 이상 매칭 불가
                    break

        # 결과 저장
        num_matched = len(matched_buyers)
        if num_matched > 0:
            self.matched_buyer.from_numpy(
                np.pad(np.array(matched_buyers, dtype=np.int32),
                       (0, MAX_ORDERS - num_matched), constant_values=-1)
            )
            self.matched_seller.from_numpy(
                np.pad(np.array(matched_sellers, dtype=np.int32),
                       (0, MAX_ORDERS - num_matched), constant_values=-1)
            )
            self.matched_house.from_numpy(
                np.pad(np.array(matched_houses, dtype=np.int32),
                       (0, MAX_ORDERS - num_matched), constant_values=-1)
            )
            self.matched_price.from_numpy(
                np.pad(np.array(matched_prices, dtype=np.float32),
                       (0, MAX_ORDERS - num_matched), constant_values=0)
            )

        self.num_matched[None] = num_matched
        self.bid_active.from_numpy(bid_active)
        self.ask_active.from_numpy(ask_active)

        return num_matched

    def get_matched_transactions(self) -> tuple:
        """체결된 거래 정보 반환

        Returns:
            (buyers, sellers, houses, prices) 튜플
        """
        num = self.num_matched[None]
        if num == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32)
            )

        buyers = self.matched_buyer.to_numpy()[:num]
        sellers = self.matched_seller.to_numpy()[:num]
        houses = self.matched_house.to_numpy()[:num]
        prices = self.matched_price.to_numpy()[:num]

        return buyers, sellers, houses, prices

    def get_bid_ask_spread(self) -> np.ndarray:
        """지역별 호가 스프레드 계산

        Returns:
            지역별 (평균 매수호가 - 평균 매도호가) / 평균 매도호가
        """
        bid_prices = self.bid_price.to_numpy()
        bid_regions = self.bid_region.to_numpy()
        bid_active = self.bid_active.to_numpy()

        ask_prices = self.ask_price.to_numpy()
        ask_regions = self.ask_region.to_numpy()
        ask_active = self.ask_active.to_numpy()

        spreads = np.zeros(NUM_REGIONS, dtype=np.float32)

        for region in range(NUM_REGIONS):
            region_bids = bid_prices[(bid_regions == region) & (bid_active == 1)]
            region_asks = ask_prices[(ask_regions == region) & (ask_active == 1)]

            if len(region_bids) > 0 and len(region_asks) > 0:
                avg_bid = np.mean(region_bids)
                avg_ask = np.mean(region_asks)
                if avg_ask > 0:
                    spreads[region] = (avg_bid - avg_ask) / avg_ask

        return spreads
