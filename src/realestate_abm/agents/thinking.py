"""사고 레이어 - System1(직감) + System2(분석) 이중처리

Kahneman의 이중 처리 모델:
- System1: 빠르고 직관적, 편향 있음 (FOMO, 사회적 증거, 앵커링)
- System2: 느리고 분석적, 합리적 (DSR, ROI, 세금)
- 블렌딩: analytical_tendency × emotional_state로 결정
"""

import numpy as np


class ThinkingEngine:
    """이중처리 사고 엔진"""

    def __init__(self, cfg):
        """cfg: ThinkingConfig"""
        self.cfg = cfg

    def evaluate_purchase(
        self,
        agents,
        region_idx: np.ndarray,
        known_prices: np.ndarray,
        actual_prices: np.ndarray,
        price_changes: np.ndarray,
        emotional_intensity: np.ndarray,
        interest_rate: float,
    ) -> np.ndarray:
        """구매 평가 점수 계산 (System1 + System2 블렌딩)

        Args:
            agents: AgentPopulation
            region_idx: (n,) 타겟 지역 인덱스
            known_prices: (n,) 에이전트가 아는 타겟 가격
            actual_prices: (n_regions,) 실제 가격 (System2만 사용)
            price_changes: (n_regions,) 가격 변화율
            emotional_intensity: (n,) 감정 강도
            interest_rate: 현재 금리

        Returns:
            (n,) 구매 점수 (높을수록 매수 선호)
        """
        d = agents.data
        n = agents.n
        cfg = self.cfg

        # System1/System2 가중치 결정
        # 차분할 때: System2 비중↑
        # 불안/FOMO 높을 때: System1 지배
        s1_weight = np.where(
            emotional_intensity > 0.5,
            cfg.system1_weight_anxious,
            cfg.system1_weight_calm,
        )
        # 개인 성향 반영
        s1_weight = s1_weight * (1 - d.analytical_tendency) + (1 - s1_weight) * d.analytical_tendency
        s1_weight = np.clip(s1_weight, 0.1, 0.9)
        s2_weight = 1.0 - s1_weight

        # === System 1 (직감) ===
        s1_score = np.zeros(n, dtype=np.float32)

        # FOMO: "가격이 올라, 지금 사야 해"
        target_changes = price_changes[region_idx]
        fomo_signal = d.fomo_level * np.maximum(target_changes, 0) * 5.0
        s1_score += fomo_signal

        # 사회적 증거: "이웃들이 다 사고 있어"
        social_signal = d.social_conformity * d.neighbor_buying_ratio * 3.0
        s1_score += social_signal

        # 불안: "집이 없으면 안 돼"
        homeless_anxiety = np.where(d.owned_houses == 0, d.anxiety * 2.0, 0)
        s1_score += homeless_anxiety

        # 주거 필요: 무주택자 기본 주거 욕구 (시간 경과에 따라 강해짐)
        housing_need = np.where(
            d.owned_houses == 0,
            np.clip(d.homeless_months / 24.0, 0.3, 1.5),
            0
        )
        s1_score += housing_need

        # 현상유지 편향: "이사 안 하면 편해" (매수 억제)
        sqb = d.status_quo_bias * 0.5
        s1_score -= sqb

        # 앵커링: "매입가보다 싸면 좋은 거래"
        anchor_signal = np.where(
            (d.purchase_price > 0) & (known_prices < d.purchase_price * 0.9),
            0.3, 0
        )
        s1_score += anchor_signal

        s1_score = np.clip(s1_score, -2, 3)

        # === System 2 (분석) ===
        s2_score = np.zeros(n, dtype=np.float32)

        # DSR 계산
        available = agents.available_for_housing + d.parent_support * (d.owned_houses == 0).astype(float)
        required_loan = np.maximum(known_prices - available, 0)
        monthly_rate = interest_rate / 12.0
        n_payments = 360  # 30년

        monthly_payment = np.where(
            required_loan > 0,
            required_loan * monthly_rate * (1 + monthly_rate)**n_payments / ((1 + monthly_rate)**n_payments - 1),
            0
        )
        annual_payment = monthly_payment * 12
        annual_income = np.maximum(d.income * 12, 1.0)  # avoid divide by zero
        dsr = np.where(d.income > 0, annual_payment / annual_income, 999.0)

        # DSR 여유 → 긍정 신호 (가중치 강화)
        dsr_score = np.clip(1 - dsr / 0.4, -1.5, 1) * 0.8
        s2_score += dsr_score

        # 금리 직접 효과: 높은 금리 = 부담 증가 = 매수 억제
        # 기준: 4% → 중립, 그 이상 패널티, 그 이하 보너스
        rate_effect = np.clip((0.04 - interest_rate) * 10, -0.8, 0.8)
        s2_score += rate_effect

        # ROI 분석 (쌍곡선 할인 적용)
        monthly_appreciation = target_changes
        # 금리 고려: 높은 금리 → 기회비용 증가 → ROI 감소
        opportunity_cost = interest_rate / 12.0
        rental_yield = 0.003
        horizon = 60
        total_return = 0.0
        for t in range(1, min(horizon + 1, 61)):
            discount = d.discount_beta * d.discount_delta ** t
            total_return += discount * (monthly_appreciation + rental_yield - opportunity_cost)
        roi_score = np.clip(total_return * d.risk_tolerance * 0.5, -0.5, 1.0)
        s2_score += roi_score

        # 구매력 점수: 자산 대비 가격 비율
        price_to_asset = known_prices / np.maximum(available, 1.0)
        affordability_score = np.clip(2.0 - price_to_asset, -1.0, 1.0) * 0.3
        s2_score += affordability_score

        # 심리적 회계: 비상금은 안 씀
        can_afford_without_emergency = available >= known_prices * 0.3
        s2_score[~can_afford_without_emergency] -= 1.5

        s2_score = np.clip(s2_score, -3, 3)

        # === 블렌딩 ===
        total_score = s1_weight * s1_score + s2_weight * s2_score

        return total_score

    def evaluate_sale(
        self,
        agents,
        current_values: np.ndarray,
        emotional_intensity: np.ndarray,
    ) -> np.ndarray:
        """매도 평가 (전망이론 + 손실 회피)"""
        d = agents.data
        cfg = self.cfg

        sell_score = np.zeros(agents.n, dtype=np.float32)

        has_house = d.owned_houses >= 1
        safe_purchase = np.maximum(d.purchase_price, 1.0)
        gain_loss = np.where(
            d.purchase_price > 0,
            (current_values - safe_purchase) / safe_purchase,
            0.0
        )

        # 전망이론 가치 함수
        gain_mask = gain_loss >= 0
        loss_mask = gain_loss < 0

        # 이득: v(x) = x^0.88
        sell_score[has_house & gain_mask] += np.power(
            np.abs(gain_loss[has_house & gain_mask]) + 1e-6, 0.88
        ) * 0.3

        # 손실: v(x) = -λ * |x|^0.88 (매도 억제)
        sell_score[has_house & loss_mask] -= (
            d.loss_aversion[has_house & loss_mask] *
            np.power(np.abs(gain_loss[has_house & loss_mask]) + 1e-6, 0.88) * 0.3
        )

        # 앵커링: 매입가 근처면 매도 꺼림
        near_anchor = has_house & (np.abs(gain_loss) < 0.1)
        sell_score[near_anchor] -= cfg.anchoring_strength * 0.3

        return sell_score
