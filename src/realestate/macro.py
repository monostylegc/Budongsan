"""거시경제 모듈 - Taylor Rule 금리 및 GDP 연동"""

import numpy as np
from dataclasses import dataclass
from .config import Config, MacroConfig


@dataclass
class MacroState:
    """거시경제 상태"""
    policy_rate: float = 0.035      # 기준금리
    mortgage_rate: float = 0.05     # 주담대 금리
    inflation: float = 0.02         # 인플레이션율
    gdp_growth: float = 0.025       # GDP 성장률 (연율)
    output_gap: float = 0.0         # 산출갭
    credit_spread: float = 0.015    # 신용 스프레드

    # 통화량 관련
    m2_growth: float = 0.08         # M2 통화량 증가율 (연율, 기본 8%)
    m2_level: float = 1.0           # M2 수준 (기준=1.0)
    liquidity_index: float = 1.0    # 유동성 지수 (부동산 시장 영향)


class MacroModel:
    """거시경제 모델

    학술적 근거:
    - Taylor (1993): Taylor Rule for monetary policy
    - AR(1) GDP 성장 프로세스

    Taylor Rule:
        i = r* + π* + α_π(π - π*) + α_y(y - y*)

        - r*: 중립 실질금리 (2%)
        - π*: 인플레이션 목표 (2%)
        - α_π: 인플레이션 반응 계수 (1.5)
        - α_y: 산출갭 반응 계수 (0.5)

    GDP 성장 (AR(1)):
        g_t = μ + ρ(g_{t-1} - μ) + ε_t

        - μ: 장기 평균 성장률 (2.5%)
        - ρ: 지속성 (0.8)
        - σ: 충격 표준편차 (1%)
    """

    def __init__(self, config: Config):
        self.config = config
        self.macro_cfg = config.macro

        # 통화량 설정 (기본값)
        self.m2_growth_target = 0.08  # 목표 M2 증가율 (연 8%)
        self.liquidity_asset_beta = 0.3  # 유동성 → 자산가격 탄력성

        # 현재 상태
        self.state = MacroState(
            policy_rate=config.policy.interest_rate,
            mortgage_rate=config.policy.interest_rate + config.policy.mortgage_spread,
            inflation=self.macro_cfg.initial_inflation,
            gdp_growth=self.macro_cfg.initial_gdp_growth,
            output_gap=0.0,
            credit_spread=self.macro_cfg.credit_spread,
            m2_growth=self.m2_growth_target,
            m2_level=1.0,
            liquidity_index=1.0
        )

        # 기록
        self.history = {
            'policy_rate': [self.state.policy_rate],
            'mortgage_rate': [self.state.mortgage_rate],
            'inflation': [self.state.inflation],
            'gdp_growth': [self.state.gdp_growth],
            'output_gap': [self.state.output_gap],
            'm2_growth': [self.state.m2_growth],
            'm2_level': [self.state.m2_level],
            'liquidity_index': [self.state.liquidity_index]
        }

    def taylor_rule(self, inflation: float, output_gap: float) -> float:
        """Taylor Rule 금리 계산

        Args:
            inflation: 현재 인플레이션율
            output_gap: 현재 산출갭 (실제GDP - 잠재GDP) / 잠재GDP

        Returns:
            목표 기준금리
        """
        cfg = self.macro_cfg

        # Taylor Rule
        # i = r* + π* + α_π(π - π*) + α_y(y - y*)
        target_rate = (
            cfg.neutral_real_rate +
            cfg.inflation_target +
            cfg.alpha_inflation * (inflation - cfg.inflation_target) +
            cfg.alpha_output * output_gap
        )

        # 금리 하한 (ZLB)
        target_rate = max(target_rate, 0.0)

        # 금리 상한
        target_rate = min(target_rate, 0.15)

        return target_rate

    def update_gdp_growth(self, rng: np.random.Generator) -> float:
        """AR(1) GDP 성장률 업데이트

        g_t = μ + ρ(g_{t-1} - μ) + ε_t

        Args:
            rng: 난수 생성기

        Returns:
            새로운 GDP 성장률
        """
        cfg = self.macro_cfg

        # AR(1) 프로세스
        mean_reversion = cfg.gdp_growth_mean + cfg.gdp_growth_persistence * (
            self.state.gdp_growth - cfg.gdp_growth_mean
        )

        # 랜덤 충격
        shock = rng.normal(0, cfg.gdp_growth_volatility)

        new_growth = mean_reversion + shock

        # 성장률 제한 (-5% ~ 10%)
        new_growth = np.clip(new_growth, -0.05, 0.10)

        return new_growth

    def update_inflation(self, house_price_change: float, rng: np.random.Generator) -> float:
        """인플레이션 업데이트

        주택 가격 변화가 인플레이션에 영향

        Args:
            house_price_change: 전국 평균 주택가격 변화율
            rng: 난수 생성기

        Returns:
            새로운 인플레이션율
        """
        # 기본 인플레이션 (목표 + 노이즈)
        base_inflation = self.macro_cfg.inflation_target + rng.normal(0, 0.005)

        # 주택 가격 영향 (Pass-through)
        housing_effect = house_price_change * 0.15  # 15% pass-through

        # AR(1) 지속성
        persistence = 0.7
        new_inflation = (
            persistence * self.state.inflation +
            (1 - persistence) * (base_inflation + housing_effect)
        )

        # 인플레이션 제한 (-0.02 ~ 0.10)
        new_inflation = np.clip(new_inflation, -0.02, 0.10)

        return new_inflation

    def update_output_gap(self, rng: np.random.Generator) -> float:
        """산출갭 업데이트

        Returns:
            새로운 산출갭
        """
        # GDP 성장률과 잠재성장률 차이
        potential_growth = self.macro_cfg.gdp_growth_mean
        gap_change = self.state.gdp_growth - potential_growth

        # 누적 효과
        persistence = 0.9
        new_gap = persistence * self.state.output_gap + gap_change

        # 산출갭 제한 (-0.1 ~ 0.1)
        new_gap = np.clip(new_gap, -0.1, 0.1)

        return new_gap

    def update_m2(self, rng: np.random.Generator) -> tuple:
        """M2 통화량 업데이트

        통화량 증가율은 목표 주변에서 AR(1) 프로세스를 따름
        유동성 지수는 통화량 증가율과 GDP 대비 통화량 비율에 영향받음

        Args:
            rng: 난수 생성기

        Returns:
            (m2_growth, m2_level, liquidity_index)
        """
        # M2 증가율 AR(1) 프로세스
        persistence = 0.85
        shock = rng.normal(0, 0.01)  # 월간 1% 표준편차

        new_m2_growth = (
            persistence * self.state.m2_growth +
            (1 - persistence) * self.m2_growth_target +
            shock
        )
        new_m2_growth = np.clip(new_m2_growth, -0.05, 0.20)  # 연 -5% ~ 20%

        # M2 수준 업데이트 (월간 증가)
        monthly_growth = new_m2_growth / 12.0
        new_m2_level = self.state.m2_level * (1 + monthly_growth)

        # 유동성 지수 계산
        # 유동성 = f(M2 증가율, M2/GDP 비율)
        # 높은 통화량 증가 → 높은 유동성 → 자산가격 상승 압력
        gdp_ratio_effect = new_m2_level / (1 + self.state.gdp_growth)
        growth_effect = 1 + (new_m2_growth - 0.05) * 2  # 5% 기준, 초과분 2배 반영

        new_liquidity = gdp_ratio_effect * growth_effect
        new_liquidity = np.clip(new_liquidity, 0.5, 2.0)

        return new_m2_growth, new_m2_level, new_liquidity

    def set_m2_growth_target(self, target: float):
        """M2 증가율 목표 설정

        Args:
            target: 목표 M2 증가율 (연율, 예: 0.08 = 8%)
        """
        self.m2_growth_target = np.clip(target, -0.05, 0.25)

    def get_liquidity_effect(self) -> float:
        """유동성이 자산가격에 미치는 효과 반환

        Returns:
            자산가격 상승 압력 계수 (1.0 = 중립)
        """
        # 유동성 지수가 1보다 크면 가격 상승 압력
        excess_liquidity = self.state.liquidity_index - 1.0
        price_effect = 1.0 + excess_liquidity * self.liquidity_asset_beta
        return np.clip(price_effect, 0.8, 1.5)

    def step(self, house_price_change: float, rng: np.random.Generator):
        """월간 거시경제 업데이트

        Args:
            house_price_change: 전국 평균 주택가격 변화율 (월간)
            rng: 난수 생성기
        """
        # 1. GDP 성장률 업데이트 (AR(1))
        self.state.gdp_growth = self.update_gdp_growth(rng)

        # 2. M2 통화량 업데이트
        m2_growth, m2_level, liquidity = self.update_m2(rng)
        self.state.m2_growth = m2_growth
        self.state.m2_level = m2_level
        self.state.liquidity_index = liquidity

        # 3. 인플레이션 업데이트 (통화량 영향 추가)
        m2_inflation_effect = (self.state.m2_growth - 0.05) * 0.3  # M2 초과 증가분의 30%
        base_inflation = self.update_inflation(house_price_change, rng)
        self.state.inflation = np.clip(base_inflation + m2_inflation_effect, -0.02, 0.15)

        # 4. 산출갭 업데이트
        self.state.output_gap = self.update_output_gap(rng)

        # 5. Taylor Rule 금리 결정
        target_rate = self.taylor_rule(self.state.inflation, self.state.output_gap)

        # 금리 조정 (점진적)
        adjustment_speed = 0.1  # 월 10%씩 조정
        self.state.policy_rate = (
            (1 - adjustment_speed) * self.state.policy_rate +
            adjustment_speed * target_rate
        )

        # 6. 주담대 금리 업데이트
        self.state.mortgage_rate = self.state.policy_rate + self.state.credit_spread

        # 기록
        self.history['policy_rate'].append(self.state.policy_rate)
        self.history['mortgage_rate'].append(self.state.mortgage_rate)
        self.history['inflation'].append(self.state.inflation)
        self.history['gdp_growth'].append(self.state.gdp_growth)
        self.history['output_gap'].append(self.state.output_gap)
        self.history['m2_growth'].append(self.state.m2_growth)
        self.history['m2_level'].append(self.state.m2_level)
        self.history['liquidity_index'].append(self.state.liquidity_index)

    def get_income_growth(self) -> float:
        """소득 성장률 반환 (GDP 연동)

        Returns:
            월간 소득 성장률
        """
        # 소득은 GDP의 beta 배로 성장
        annual_income_growth = self.macro_cfg.income_gdp_beta * self.state.gdp_growth
        monthly_income_growth = annual_income_growth / 12.0
        return monthly_income_growth

    def get_mortgage_rate(self) -> float:
        """주담대 금리 반환

        Returns:
            연간 주담대 금리
        """
        return self.state.mortgage_rate

    def get_jeonse_conversion_rate(self) -> float:
        """전월세 전환율 반환

        주택금융공사 기준: 기준금리 + 리스크 프리미엄

        Returns:
            연간 전환율
        """
        base = self.state.policy_rate
        risk_premium = 0.02  # 2% 리스크 프리미엄
        return base + risk_premium

    def update_credit_spread(self, market_stress: float):
        """신용 스프레드 업데이트

        시장 스트레스에 따라 스프레드 확대

        Args:
            market_stress: 시장 스트레스 지표 (0~1)
        """
        base_spread = self.macro_cfg.credit_spread
        stress_spread = market_stress * 0.02  # 최대 2% 추가

        self.state.credit_spread = base_spread + stress_spread
        self.state.mortgage_rate = self.state.policy_rate + self.state.credit_spread

    def get_state_dict(self) -> dict:
        """현재 상태 딕셔너리 반환"""
        return {
            'policy_rate': self.state.policy_rate,
            'mortgage_rate': self.state.mortgage_rate,
            'inflation': self.state.inflation,
            'gdp_growth': self.state.gdp_growth,
            'output_gap': self.state.output_gap,
            'credit_spread': self.state.credit_spread,
            'm2_growth': self.state.m2_growth,
            'm2_level': self.state.m2_level,
            'liquidity_index': self.state.liquidity_index
        }

    def reset(self):
        """상태 초기화"""
        self.state = MacroState(
            policy_rate=self.config.policy.interest_rate,
            mortgage_rate=self.config.policy.interest_rate + self.config.policy.mortgage_spread,
            inflation=self.macro_cfg.initial_inflation,
            gdp_growth=self.macro_cfg.initial_gdp_growth,
            output_gap=0.0,
            credit_spread=self.macro_cfg.credit_spread,
            m2_growth=self.m2_growth_target,
            m2_level=1.0,
            liquidity_index=1.0
        )
        self.history = {
            'policy_rate': [self.state.policy_rate],
            'mortgage_rate': [self.state.mortgage_rate],
            'inflation': [self.state.inflation],
            'gdp_growth': [self.state.gdp_growth],
            'output_gap': [self.state.output_gap],
            'm2_growth': [self.state.m2_growth],
            'm2_level': [self.state.m2_level],
            'liquidity_index': [self.state.liquidity_index]
        }
