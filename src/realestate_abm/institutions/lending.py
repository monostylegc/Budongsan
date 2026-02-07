"""대출 규제 시스템 - LTV/DSR 지역별 차등"""

import numpy as np


class LendingSystem:
    """대출 규제"""

    def __init__(self, cfg):
        """cfg: LendingConfig"""
        self.cfg = cfg
        self._build_ltv_table()

    def _build_ltv_table(self):
        """LTV 룩업 테이블 구축"""
        self._ltv_by_count = {}
        for rule in self.cfg.ltv_rules:
            self._ltv_by_count[rule.house_count] = {
                'ltv': rule.ltv,
                'overrides': rule.region_overrides or {}
            }

    def get_ltv(self, house_counts: np.ndarray, region_ids: np.ndarray = None) -> np.ndarray:
        """LTV 한도 조회 (벡터화)"""
        ltv = np.zeros(len(house_counts), dtype=np.float32)
        for count, info in self._ltv_by_count.items():
            mask = house_counts == count
            ltv[mask] = info['ltv']
        # 3주택 이상은 마지막 규칙 적용
        max_count = max(self._ltv_by_count.keys())
        ltv[house_counts > max_count] = self._ltv_by_count[max_count]['ltv']
        return ltv

    def compute_dsr(self, loan_amounts: np.ndarray, annual_incomes: np.ndarray,
                     interest_rate: float, loan_term_years: int = 30) -> np.ndarray:
        """DSR 계산 (벡터화)"""
        monthly_rate = interest_rate / 12.0
        n_payments = loan_term_years * 12

        monthly_payment = np.where(
            loan_amounts > 0,
            loan_amounts * monthly_rate * (1 + monthly_rate)**n_payments / ((1 + monthly_rate)**n_payments - 1),
            0
        )
        annual_payment = monthly_payment * 12
        dsr = np.where(annual_incomes > 0, annual_payment / annual_incomes, 999)
        return dsr

    def check_affordability(self, prices: np.ndarray, assets: np.ndarray,
                             annual_incomes: np.ndarray, house_counts: np.ndarray,
                             interest_rate: float) -> np.ndarray:
        """구매 가능 여부 종합 판단

        Returns:
            (n,) bool 배열
        """
        ltv = self.get_ltv(house_counts)
        max_loan = prices * ltv
        required_loan = np.maximum(prices - assets, 0)

        # LTV 체크
        ltv_ok = required_loan <= max_loan

        # DSR 체크
        dsr = self.compute_dsr(required_loan, annual_incomes, interest_rate)
        dsr_ok = dsr <= self.cfg.dsr_limit

        return ltv_ok & dsr_ok
