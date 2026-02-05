"""Pydantic 스키마 정의 - API 요청/응답 모델 (전체 파라미터 지원)"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class PolicyParams(BaseModel):
    """정책 파라미터 - 대출/세금 규제"""
    # 대출 규제
    ltv_1house: float = Field(default=0.50, ge=0.0, le=1.0)
    ltv_2house: float = Field(default=0.30, ge=0.0, le=1.0)
    ltv_3house: float = Field(default=0.00, ge=0.0, le=1.0)
    dti_limit: float = Field(default=0.40, ge=0.1, le=0.8)
    dsr_limit: float = Field(default=0.40, ge=0.1, le=0.8)

    # 취득세
    acq_tax_1house: float = Field(default=0.01, ge=0.0, le=0.2)
    acq_tax_2house: float = Field(default=0.08, ge=0.0, le=0.2)
    acq_tax_3house: float = Field(default=0.12, ge=0.0, le=0.3)

    # 양도세
    transfer_tax_short: float = Field(default=0.70, ge=0.1, le=0.9)
    transfer_tax_long: float = Field(default=0.40, ge=0.1, le=0.7)
    transfer_tax_multi_short: float = Field(default=0.75, ge=0.1, le=0.9)
    transfer_tax_multi_long: float = Field(default=0.60, ge=0.1, le=0.9)

    # 종부세
    jongbu_threshold_1house: float = Field(default=110000, ge=30000, le=300000)
    jongbu_threshold_multi: float = Field(default=60000, ge=10000, le=150000)
    jongbu_rate: float = Field(default=0.02, ge=0.0, le=0.1)

    # 금리
    interest_rate: float = Field(default=0.035, ge=0.005, le=0.15)
    mortgage_spread: float = Field(default=0.015, ge=0.005, le=0.05)

    # 전월세
    jeonse_loan_limit: float = Field(default=50000, ge=10000, le=150000)
    rent_increase_cap: float = Field(default=0.05, ge=0.0, le=0.2)


class BehavioralParams(BaseModel):
    """행동경제학 파라미터"""
    # FOMO
    fomo_trigger_threshold: float = Field(default=0.05, ge=0.01, le=0.3)
    fomo_intensity: float = Field(default=50.0, ge=1.0, le=500.0)

    # 손실 회피
    loss_aversion_mean: float = Field(default=2.5, ge=1.0, le=5.0)
    loss_aversion_std: float = Field(default=0.35, ge=0.1, le=1.5)
    loss_aversion_decay: float = Field(default=5.0, ge=1.0, le=30.0)

    # 앵커링
    anchoring_threshold: float = Field(default=0.1, ge=0.0, le=0.5)
    anchoring_penalty: float = Field(default=0.5, ge=0.0, le=3.0)

    # 군집 행동
    herding_trigger: float = Field(default=0.03, ge=0.01, le=0.2)
    herding_intensity: float = Field(default=10.0, ge=1.0, le=100.0)

    # 사회적 학습
    social_learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    news_impact: float = Field(default=0.2, ge=0.0, le=1.0)


class AgentCompositionParams(BaseModel):
    """에이전트 구성 파라미터"""
    investor_ratio: float = Field(default=0.15, ge=0.0, le=0.5)
    speculator_ratio: float = Field(default=0.05, ge=0.0, le=0.4)
    speculator_risk_multiplier: float = Field(default=1.5, ge=1.0, le=5.0)
    speculator_fomo_multiplier: float = Field(default=1.3, ge=1.0, le=5.0)
    speculator_horizon_min: int = Field(default=6, ge=1, le=24)
    speculator_horizon_max: int = Field(default=24, ge=6, le=60)
    initial_homeless_rate: float = Field(default=0.45, ge=0.1, le=0.8)
    initial_one_house_rate: float = Field(default=0.40, ge=0.1, le=0.7)
    initial_multi_house_rate: float = Field(default=0.15, ge=0.0, le=0.4)

    # 소득 분포 (로그정규: 중위값 + 로그 표준편차)
    income_median: float = Field(default=300, ge=100, le=1000, description="월소득 중위값 (만원)")
    income_sigma: float = Field(default=0.6, ge=0.2, le=1.5, description="로그 표준편차 (분산도)")

    # 자산 분포 (파레토: 중위값 + 알파)
    asset_median: float = Field(default=5000, ge=1000, le=50000, description="순자산 중위값 (만원)")
    asset_alpha: float = Field(default=1.5, ge=1.1, le=3.0, description="파레토 알파 (낮을수록 불평등)")

    # 연령 분포
    age_young_ratio: float = Field(default=0.45, ge=0.1, le=0.7, description="청년층 (25-34세) 비율")
    age_middle_ratio: float = Field(default=0.43, ge=0.1, le=0.7, description="중년층 (35-54세) 비율")
    age_senior_ratio: float = Field(default=0.12, ge=0.05, le=0.5, description="장년층 (55세+) 비율")


class LifeCycleParams(BaseModel):
    """생애주기 파라미터"""
    marriage_urgency_age_start: int = Field(default=28, ge=20, le=40)
    marriage_urgency_age_end: int = Field(default=35, ge=25, le=50)
    newlywed_housing_pressure: float = Field(default=1.5, ge=1.0, le=5.0)
    parenting_housing_pressure: float = Field(default=1.3, ge=1.0, le=3.0)
    school_transition_age_start: int = Field(default=10, ge=5, le=15)
    school_transition_age_end: int = Field(default=15, ge=10, le=20)
    school_district_premium: float = Field(default=1.2, ge=1.0, le=3.0)
    retirement_start_age: int = Field(default=55, ge=45, le=70)
    downsizing_probability: float = Field(default=0.1, ge=0.0, le=0.5)


class NetworkParams(BaseModel):
    """네트워크 파라미터"""
    avg_neighbors: int = Field(default=10, ge=2, le=50)
    rewiring_prob: float = Field(default=0.1, ge=0.0, le=0.5)
    cascade_threshold: float = Field(default=0.3, ge=0.1, le=0.8)
    cascade_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    self_weight: float = Field(default=0.6, ge=0.2, le=0.9)


class MacroParams(BaseModel):
    """거시경제 파라미터"""
    m2_growth: float = Field(default=0.08, ge=0.0, le=0.3)
    gdp_growth_mean: float = Field(default=0.025, ge=-0.1, le=0.15)
    gdp_growth_volatility: float = Field(default=0.01, ge=0.0, le=0.1)
    inflation_target: float = Field(default=0.02, ge=0.0, le=0.1)
    income_gdp_beta: float = Field(default=0.8, ge=0.3, le=2.0)


class SupplyParams(BaseModel):
    """공급 파라미터"""
    base_supply_rate: float = Field(default=0.001, ge=0.0001, le=0.01)
    elasticity_gangnam: float = Field(default=0.3, ge=0.05, le=2.0)
    elasticity_seoul: float = Field(default=0.5, ge=0.1, le=2.0)
    elasticity_gyeonggi: float = Field(default=1.5, ge=0.3, le=5.0)
    elasticity_local: float = Field(default=2.0, ge=0.5, le=10.0)
    redevelopment_base_prob: float = Field(default=0.001, ge=0.0, le=0.01)
    redevelopment_age_threshold: int = Field(default=30, ge=15, le=60)
    construction_period: int = Field(default=24, ge=6, le=60)


class DepreciationParams(BaseModel):
    """노후화/멸실 파라미터"""
    depreciation_rate: float = Field(default=0.002, ge=0.0005, le=0.01)
    natural_demolition_threshold: float = Field(default=0.1, ge=0.05, le=0.5)
    disaster_rate: float = Field(default=0.0001, ge=0.0, le=0.001)


class MarketParams(BaseModel):
    """시장 파라미터"""
    price_sensitivity: float = Field(default=0.001, ge=0.0001, le=0.01)
    expectation_weight: float = Field(default=0.015, ge=0.001, le=0.1)
    base_appreciation: float = Field(default=0.002, ge=0.0, le=0.02)
    buy_threshold: float = Field(default=0.25, ge=0.05, le=0.7)
    sell_threshold: float = Field(default=0.30, ge=0.05, le=0.8)
    spillover_rate: float = Field(default=0.005, ge=0.001, le=0.05)


class SimulationParams(BaseModel):
    """시뮬레이션 파라미터 전체"""
    # 규모
    num_households: int = Field(default=100_000, ge=1_000, le=2_000_000)
    num_houses: int = Field(default=60_000, ge=600, le=1_200_000)
    num_steps: int = Field(default=120, ge=12, le=600)
    seed: int = Field(default=42)

    # 하위 파라미터
    policy: PolicyParams = Field(default_factory=PolicyParams)
    behavioral: BehavioralParams = Field(default_factory=BehavioralParams)
    agent_composition: AgentCompositionParams = Field(default_factory=AgentCompositionParams)
    lifecycle: LifeCycleParams = Field(default_factory=LifeCycleParams)
    network: NetworkParams = Field(default_factory=NetworkParams)
    macro: MacroParams = Field(default_factory=MacroParams)
    supply: SupplyParams = Field(default_factory=SupplyParams)
    depreciation: DepreciationParams = Field(default_factory=DepreciationParams)
    market: MarketParams = Field(default_factory=MarketParams)

    scenario: str = Field(default="default")


class RegionStats(BaseModel):
    """지역별 통계"""
    region_id: int
    name: str
    price: float
    price_change: float
    transactions: int
    demand: int
    supply: int
    jeonse_ratio: float
    homeless_count: int
    one_house_count: int
    multi_house_count: int


class MonthlyState(BaseModel):
    """월별 시뮬레이션 상태"""
    month: int
    year: int
    avg_price: float
    total_transactions: int
    homeowner_rate: float
    multi_owner_rate: float
    demand_supply_ratio: float
    listing_rate: float
    interest_rate: float
    inflation: float
    gdp_growth: float
    m2_growth: float
    mean_building_age: float
    mean_building_condition: float
    demolished_count: int
    regions: List[RegionStats]
    recent_transactions: List[Dict[str, Any]]


class SimulationStartResponse(BaseModel):
    """시뮬레이션 시작 응답"""
    session_id: str
    status: str
    message: str
    params: SimulationParams


class SimulationStatusResponse(BaseModel):
    """시뮬레이션 상태 응답"""
    session_id: str
    status: str
    current_step: int
    total_steps: int
    progress: float


class ScenarioPreset(BaseModel):
    """시나리오 프리셋"""
    name: str
    description: str
    params: SimulationParams


class DefaultParamsResponse(BaseModel):
    """기본 파라미터 응답"""
    params: SimulationParams
    scenarios: Dict[str, ScenarioPreset]


class ScenarioType(BaseModel):
    """시나리오 유형"""
    id: str
    name: str
    description: str
