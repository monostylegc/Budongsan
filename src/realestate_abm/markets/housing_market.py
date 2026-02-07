"""주택시장 - 매칭, 거래, 집계"""

import numpy as np


class HousingMarket:
    """주택시장 통합"""

    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        nr = world.n

        self.region_prices = np.zeros(nr, dtype=np.float32)
        self.region_jeonse = np.zeros(nr, dtype=np.float32)
        self.region_price_changes = np.zeros(nr, dtype=np.float32)
        self.demand = np.zeros(nr, dtype=np.int32)
        self.supply = np.zeros(nr, dtype=np.int32)
        self.transactions = np.zeros(nr, dtype=np.int32)
        self.total_houses = np.zeros(nr, dtype=np.int32)

        self.price_history = []
        self.transaction_history = []

        # 외부 시스템 참조 (engine에서 주입)
        self.lending = None
        self.tax_system = None
        self.mortgage_rate = 0.05

    def aggregate_prices(self, houses):
        nr = self.world.n
        old_prices = self.region_prices.copy()

        for r in range(nr):
            mask = (houses.region == r) & (houses.is_active == 1)
            count = np.sum(mask)
            self.total_houses[r] = count
            if count > 0:
                self.region_prices[r] = np.mean(houses.price[mask])
                self.region_jeonse[r] = np.mean(houses.jeonse_price[mask])

        valid = old_prices > 0
        self.region_price_changes[valid] = (self.region_prices[valid] - old_prices[valid]) / old_prices[valid]

    def count_demand_supply(self, agents, houses):
        nr = self.world.n
        d = agents.data

        self.demand[:] = 0
        self.supply[:] = 0
        for r in range(nr):
            self.demand[r] = np.sum((d.wants_to_buy == 1) & (d.target_region == r))
            self.supply[r] = np.sum((houses.is_for_sale == 1) & (houses.region == r))

    def simple_matching(self, agents, houses, rng: np.random.Generator, current_month: int) -> int:
        """LTV/DSR/세금 기반 매칭"""
        d = agents.data
        nr = self.world.n
        total_matched = 0
        self.transactions[:] = 0

        for region in range(nr):
            buyers = np.where((d.wants_to_buy == 1) & (d.target_region == region))[0]
            sellers = np.where((houses.is_for_sale == 1) & (houses.region == region) & (houses.is_active == 1))[0]

            if len(buyers) == 0 or len(sellers) == 0:
                continue

            rng.shuffle(buyers)
            rng.shuffle(sellers)

            si = 0  # seller index
            for buyer_id in buyers:
                if si >= len(sellers):
                    break
                house_id = sellers[si]
                price = houses.price[house_id]

                available = d.housing_fund[buyer_id] + d.investment_fund[buyer_id] * 0.5
                if d.owned_houses[buyer_id] == 0:
                    available += d.parent_support[buyer_id]

                # 구매 가능 여부 판단
                can_afford = self._check_can_afford(
                    buyer_id, d, price, available
                )

                if not can_afford:
                    continue  # 이 매수자 건너뜀, 매물은 유지

                # 취득세 계산
                acq_tax = 0.0
                if self.tax_system is not None:
                    new_count = d.owned_houses[buyer_id] + 1
                    acq_tax = float(self.tax_system.compute_acquisition_tax(
                        np.array([price]), np.array([new_count])
                    )[0])

                # 거래 성사
                old_owner = houses.owner_id[house_id]
                houses.is_for_sale[house_id] = 0
                houses.owner_id[house_id] = buyer_id
                d.wants_to_buy[buyer_id] = 0
                d.owned_houses[buyer_id] += 1

                if d.owned_houses[buyer_id] == 1:
                    d.purchase_price[buyer_id] = price
                d.total_purchase_price[buyer_id] += price
                d.purchase_month[buyer_id] = current_month

                # 실제 비용 차감 (자기자본 + 취득세)
                total_cost = min(price, available) + acq_tax
                fund_use = min(total_cost, d.housing_fund[buyer_id])
                d.housing_fund[buyer_id] -= fund_use
                remaining_cost = total_cost - fund_use
                if remaining_cost > 0:
                    inv_use = min(remaining_cost, d.investment_fund[buyer_id] * 0.5)
                    d.investment_fund[buyer_id] -= inv_use

                # 부모 지원 소진
                if d.owned_houses[buyer_id] == 1:
                    d.parent_support[buyer_id] = 0

                if old_owner >= 0:
                    d.owned_houses[old_owner] -= 1
                    # 매도 대금 지급
                    d.housing_fund[old_owner] += price * 0.5
                    d.investment_fund[old_owner] += price * 0.5
                    if d.owned_houses[old_owner] == 0:
                        d.purchase_price[old_owner] = 0
                        d.total_purchase_price[old_owner] = 0

                si += 1
                total_matched += 1
                self.transactions[region] += 1

        return total_matched

    def _check_can_afford(self, buyer_id, d, price, available):
        """구매 가능 여부 (LTV + DSR 종합)"""
        if self.lending is not None:
            house_count = d.owned_houses[buyer_id]
            ltv = self.lending.get_ltv(
                np.array([house_count], dtype=np.int32)
            )[0]
            max_loan = price * ltv
            required_down = price - max_loan

            if available >= price:
                return True  # 전액 현금

            if available < required_down:
                return False  # 자기자본 부족

            # DSR 체크
            actual_loan = min(price - available, max_loan)
            if actual_loan > 0:
                annual_income = float(d.income[buyer_id] * 12)
                if annual_income <= 0:
                    return False
                dsr = self.lending.compute_dsr(
                    np.array([actual_loan]),
                    np.array([annual_income]),
                    self.mortgage_rate,
                )[0]
                return dsr <= self.lending.cfg.dsr_limit
            return True
        else:
            return available >= price * 0.3

    def record_history(self):
        self.price_history.append(self.region_prices.copy())
        self.transaction_history.append(self.transactions.copy())
