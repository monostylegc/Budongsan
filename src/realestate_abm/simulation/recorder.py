"""통계 기록"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MonthlyStats:
    """월간 통계"""
    month: int = 0
    region_prices: np.ndarray = None
    region_jeonse: np.ndarray = None
    region_price_changes: np.ndarray = None
    transactions: np.ndarray = None
    demand: np.ndarray = None
    supply: np.ndarray = None
    total_houses: np.ndarray = None

    # Agent stats
    avg_income: float = 0.0
    median_assets: float = 0.0
    homeless_rate: float = 0.0
    homeowner_rate: float = 0.0
    multi_owner_rate: float = 0.0
    avg_anxiety: float = 0.0
    avg_fomo: float = 0.0
    avg_satisfaction: float = 0.0
    triggered_ratio: float = 0.0

    # Macro
    gdp_growth: float = 0.0
    inflation: float = 0.0
    interest_rate: float = 0.0
    unemployment_rate: float = 0.0

    # Supply
    new_construction: int = 0
    completed_construction: int = 0

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d


class Recorder:
    """시뮬레이션 통계 기록기"""

    def __init__(self, n_regions: int):
        self.n_regions = n_regions
        self.history: list[MonthlyStats] = []

    def record(self, month: int, agents, houses, market, macro, monetary, labor) -> MonthlyStats:
        """한 달치 통계 기록"""
        d = agents.data
        n = agents.n

        stats = MonthlyStats(
            month=month,
            region_prices=market.region_prices.copy(),
            region_jeonse=market.region_jeonse.copy(),
            region_price_changes=market.region_price_changes.copy(),
            transactions=market.transactions.copy(),
            demand=market.demand.copy(),
            supply=market.supply.copy(),
            total_houses=market.total_houses.copy(),

            avg_income=float(np.mean(d.income)),
            median_assets=float(np.median(agents.total_assets)),
            homeless_rate=float(np.mean(d.owned_houses == 0)),
            homeowner_rate=float(np.mean(d.owned_houses == 1)),
            multi_owner_rate=float(np.mean(d.owned_houses >= 2)),
            avg_anxiety=float(np.mean(d.anxiety)),
            avg_fomo=float(np.mean(d.fomo_level)),
            avg_satisfaction=float(np.mean(d.satisfaction)),
            triggered_ratio=float(np.mean(d.is_triggered)),

            gdp_growth=macro.state.gdp_growth,
            inflation=macro.state.inflation,
            interest_rate=monetary.get_mortgage_rate(),
            unemployment_rate=float(np.mean(d.employment_status != 0)),
        )

        self.history.append(stats)
        return stats

    def get_price_series(self) -> np.ndarray:
        """(n_months, n_regions) 가격 시계열"""
        if not self.history:
            return np.array([])
        return np.array([s.region_prices for s in self.history])

    def get_summary(self) -> dict:
        """최종 요약"""
        if not self.history:
            return {}
        first = self.history[0]
        last = self.history[-1]
        n_months = len(self.history)

        price_changes = {}
        if first.region_prices is not None and last.region_prices is not None:
            for i in range(self.n_regions):
                if first.region_prices[i] > 0:
                    pct = (last.region_prices[i] - first.region_prices[i]) / first.region_prices[i] * 100
                    price_changes[i] = round(pct, 2)

        return {
            'months': n_months,
            'price_changes_pct': price_changes,
            'final_homeless_rate': last.homeless_rate,
            'final_homeowner_rate': last.homeowner_rate,
            'final_avg_income': last.avg_income,
            'final_interest_rate': last.interest_rate,
            'total_transactions': sum(int(s.transactions.sum()) for s in self.history if s.transactions is not None),
        }

    def reset(self):
        self.history = []
