"""세금 시스템 - JSON 규칙 기반, 실제 차감"""

import numpy as np


class TaxSystem:
    """세금 계산 및 차감"""

    def __init__(self, cfg):
        """cfg: TaxConfig from InstitutionsConfig"""
        self.cfg = cfg

    def compute_acquisition_tax(self, prices: np.ndarray, house_counts: np.ndarray,
                                  region_ids: np.ndarray = None) -> np.ndarray:
        """취득세 계산 (벡터화)

        Args:
            prices: (n,) 매수 가격
            house_counts: (n,) 매수 후 보유 주택 수
            region_ids: (n,) 지역 ID (지역별 차등 적용)

        Returns:
            (n,) 취득세 금액
        """
        tax = np.zeros_like(prices)
        for rule in self.cfg.acquisition_tax:
            mask = (house_counts >= rule.house_count_min) & (house_counts <= rule.house_count_max)
            if rule.regions is not None and region_ids is not None:
                # 지역별 적용 (TODO: region_id → region_name 매핑 필요)
                pass  # 지역 필터링 추후 구현
            tax[mask] = prices[mask] * rule.rate
        return tax

    def compute_transfer_tax(self, gains: np.ndarray, holding_months: np.ndarray,
                              house_counts: np.ndarray) -> np.ndarray:
        """양도세 계산"""
        cfg = self.cfg
        tax = np.zeros_like(gains)

        positive_gain = gains > 0

        # 1주택자
        single = house_counts == 1
        # 장기 보유
        long_hold = holding_months >= 24

        tax[positive_gain & single & long_hold] = gains[positive_gain & single & long_hold] * cfg.transfer_tax_long
        tax[positive_gain & single & ~long_hold] = gains[positive_gain & single & ~long_hold] * cfg.transfer_tax_short

        # 다주택자
        multi = house_counts >= 2
        tax[positive_gain & multi & long_hold] = gains[positive_gain & multi & long_hold] * cfg.transfer_tax_multi_long
        tax[positive_gain & multi & ~long_hold] = gains[positive_gain & multi & ~long_hold] * cfg.transfer_tax_multi_short

        return tax

    def compute_jongbu_tax(self, total_values: np.ndarray, house_counts: np.ndarray) -> np.ndarray:
        """종합부동산세 (연간, 월간으로 변환하여 반환)"""
        cfg = self.cfg
        annual_tax = np.zeros_like(total_values)

        # 1주택자
        single = house_counts == 1
        exceed_1 = total_values > cfg.jongbu_threshold_1house
        annual_tax[single & exceed_1] = (total_values[single & exceed_1] - cfg.jongbu_threshold_1house) * cfg.jongbu_rate

        # 다주택자
        multi = house_counts >= 2
        exceed_m = total_values > cfg.jongbu_threshold_multi
        annual_tax[multi & exceed_m] = (total_values[multi & exceed_m] - cfg.jongbu_threshold_multi) * cfg.jongbu_rate * 1.5

        return annual_tax / 12.0  # 월간으로 변환

    def deduct_acquisition_tax(self, agents, buyer_ids: np.ndarray, prices: np.ndarray):
        """취득세 실제 차감"""
        house_counts = agents.data.owned_houses[buyer_ids]
        tax = self.compute_acquisition_tax(prices, house_counts)
        agents.data.housing_fund[buyer_ids] -= tax
        return tax

    def deduct_monthly_jongbu(self, agents, house_values_by_agent: np.ndarray):
        """월간 종부세 차감"""
        d = agents.data
        monthly_tax = self.compute_jongbu_tax(house_values_by_agent, d.owned_houses)
        has_tax = monthly_tax > 0
        d.investment_fund[has_tax] -= monthly_tax[has_tax]
        # 투자자금 부족 시 주거자금에서 차감
        deficit = d.investment_fund < 0
        d.housing_fund[deficit] += d.investment_fund[deficit]
        d.investment_fund[deficit] = 0
        return monthly_tax
