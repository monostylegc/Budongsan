"""Double Auction 주문장 (기존 order_book.py 포트 → NumPy)"""

import numpy as np


class OrderBook:
    """NumPy 기반 Double Auction"""

    def __init__(self, max_orders: int = 100000):
        self.max_orders = max_orders
        self.bid_prices = np.zeros(max_orders, dtype=np.float32)
        self.bid_agents = np.full(max_orders, -1, dtype=np.int32)
        self.bid_regions = np.full(max_orders, -1, dtype=np.int32)
        self.bid_active = np.zeros(max_orders, dtype=np.int32)
        self.n_bids = 0

        self.ask_prices = np.zeros(max_orders, dtype=np.float32)
        self.ask_agents = np.full(max_orders, -1, dtype=np.int32)
        self.ask_houses = np.full(max_orders, -1, dtype=np.int32)
        self.ask_regions = np.full(max_orders, -1, dtype=np.int32)
        self.ask_active = np.zeros(max_orders, dtype=np.int32)
        self.n_asks = 0

    def clear(self):
        self.bid_active[:] = 0
        self.ask_active[:] = 0
        self.n_bids = 0
        self.n_asks = 0

    def generate_bids(self, agents, region_prices: np.ndarray, rng):
        d = agents.data
        buyers = np.where(d.wants_to_buy == 1)[0]
        if len(buyers) == 0:
            return
        n = min(len(buyers), self.max_orders)
        buyers = buyers[:n]

        margin = rng.uniform(0.03, 0.10, n)
        base = region_prices[d.target_region[buyers]]
        perceived = base * (1 + d.price_expectation[buyers] * d.risk_tolerance[buyers] * 0.2 if hasattr(d, 'price_expectation') else base)

        # Simple bid calculation
        bid = base * (1.0 - margin)
        max_bid = (d.housing_fund[buyers] + d.investment_fund[buyers] * 0.5) * 0.8
        bid = np.minimum(bid, max_bid).astype(np.float32)

        self.bid_prices[:n] = bid
        self.bid_agents[:n] = buyers
        self.bid_regions[:n] = d.target_region[buyers]
        self.bid_active[:n] = 1
        self.n_bids = n

    def generate_asks(self, agents, houses, region_prices: np.ndarray, rng):
        for_sale = np.where((houses.is_for_sale == 1) & (houses.is_active == 1))[0]
        if len(for_sale) == 0:
            return
        n = min(len(for_sale), self.max_orders)
        for_sale = for_sale[:n]

        owners = houses.owner_id[for_sale]
        valid = owners >= 0
        valid_houses = for_sale[valid]
        valid_owners = owners[valid]

        if len(valid_owners) == 0:
            return

        current = houses.price[valid_houses]
        purchase = agents.data.purchase_price[valid_owners]

        gain_loss = np.where(purchase > 0, (current - purchase) / purchase, 0)

        ask = np.where(
            gain_loss < 0,
            purchase * (1 + rng.uniform(0.25, 0.35, len(valid_owners))),
            current * (1 - rng.uniform(0, 0.05, len(valid_owners)))
        ).astype(np.float32)

        nv = len(valid_owners)
        self.ask_prices[:nv] = ask
        self.ask_agents[:nv] = valid_owners
        self.ask_houses[:nv] = valid_houses
        self.ask_regions[:nv] = houses.region[valid_houses]
        self.ask_active[:nv] = 1
        self.n_asks = nv

    def match(self, n_regions: int) -> tuple:
        """매칭 → (buyers, sellers, houses, prices)"""
        matched_b, matched_s, matched_h, matched_p = [], [], [], []

        for r in range(n_regions):
            rb = np.where((self.bid_regions[:self.n_bids] == r) & (self.bid_active[:self.n_bids] == 1))[0]
            ra = np.where((self.ask_regions[:self.n_asks] == r) & (self.ask_active[:self.n_asks] == 1))[0]
            if len(rb) == 0 or len(ra) == 0:
                continue

            bid_order = np.argsort(-self.bid_prices[rb])
            ask_order = np.argsort(self.ask_prices[ra])
            sorted_b = rb[bid_order]
            sorted_a = ra[ask_order]

            bi, ai = 0, 0
            while bi < len(sorted_b) and ai < len(sorted_a):
                b, a = sorted_b[bi], sorted_a[ai]
                if self.bid_prices[b] >= self.ask_prices[a]:
                    price = (self.bid_prices[b] + self.ask_prices[a]) / 2
                    matched_b.append(self.bid_agents[b])
                    matched_s.append(self.ask_agents[a])
                    matched_h.append(self.ask_houses[a])
                    matched_p.append(price)
                    self.bid_active[b] = 0
                    self.ask_active[a] = 0
                    bi += 1
                    ai += 1
                else:
                    break

        if not matched_b:
            return np.array([], np.int32), np.array([], np.int32), np.array([], np.int32), np.array([], np.float32)
        return (np.array(matched_b, np.int32), np.array(matched_s, np.int32),
                np.array(matched_h, np.int32), np.array(matched_p, np.float32))
