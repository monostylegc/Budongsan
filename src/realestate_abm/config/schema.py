"""Pydantic 기반 설정 스키마 - JSON 검증 및 기본값"""

from pydantic import BaseModel, Field
from typing import Optional


class AffordabilityConfig(BaseModel):
    """구매력 설정"""
    dsr_limit_end_user: float = 0.35
    dsr_limit_investor: float = 0.40
    dsr_limit_speculator: float = 0.45
    wealthy_asset_utilization: float = 0.7
    normal_asset_utilization: float = 0.5
    homeless_asset_utilization: float = 0.85
    wealthy_percentile: float = 90.0
    parent_support_rate: float = 0.6
    parent_support_mean: float = 25000.0
    parent_support_std: float = 12000.0
    parent_support_age_max: int = 40
    loan_term_years: int = 30


class TaxRule(BaseModel):
    """세금 규칙 (하나의 조건)"""
    house_count_min: int = 1
    house_count_max: int = 999
    rate: float = 0.01
    regions: Optional[list[str]] = None  # None = 전국 적용


class TaxConfig(BaseModel):
    """세금 시스템"""
    acquisition_tax: list[TaxRule] = Field(default_factory=lambda: [
        TaxRule(house_count_min=1, house_count_max=1, rate=0.01),
        TaxRule(house_count_min=2, house_count_max=2, rate=0.08),
        TaxRule(house_count_min=3, house_count_max=999, rate=0.12),
    ])
    transfer_tax_short: float = 0.70   # 2년 미만 보유
    transfer_tax_long: float = 0.40    # 2년 이상
    transfer_tax_multi_short: float = 0.75
    transfer_tax_multi_long: float = 0.60
    jongbu_threshold_1house: float = 110000
    jongbu_threshold_multi: float = 60000
    jongbu_rate: float = 0.02


class LendingRule(BaseModel):
    """대출 규제"""
    house_count: int
    ltv: float
    region_overrides: Optional[dict[str, float]] = None


class LendingConfig(BaseModel):
    """대출 규제 시스템"""
    ltv_rules: list[LendingRule] = Field(default_factory=lambda: [
        LendingRule(house_count=0, ltv=0.70),
        LendingRule(house_count=1, ltv=0.50),
        LendingRule(house_count=2, ltv=0.00),
        LendingRule(house_count=3, ltv=0.00),
    ])
    dsr_limit: float = 0.40
    dti_limit: float = 0.40


class MonetaryConfig(BaseModel):
    """통화정책"""
    interest_rate: float = 0.035
    mortgage_spread: float = 0.015
    neutral_real_rate: float = 0.02
    inflation_target: float = 0.02
    alpha_inflation: float = 1.5
    alpha_output: float = 0.5


class RentalConfig(BaseModel):
    """임대시장"""
    jeonse_loan_limit: float = 50000
    rent_increase_cap: float = 0.05
    jeonse_ratio: float = 0.55
    conversion_rate: float = 0.05


class PolicyEvent(BaseModel):
    """시간에 따른 정책 변경"""
    month: int
    type: str   # "set_ltv", "set_interest_rate", "set_tax", etc.
    params: dict


class InstitutionsConfig(BaseModel):
    """제도 환경 전체"""
    tax: TaxConfig = Field(default_factory=TaxConfig)
    lending: LendingConfig = Field(default_factory=LendingConfig)
    monetary: MonetaryConfig = Field(default_factory=MonetaryConfig)
    rental: RentalConfig = Field(default_factory=RentalConfig)
    affordability: AffordabilityConfig = Field(default_factory=AffordabilityConfig)
    policy_timeline: list[PolicyEvent] = Field(default_factory=list)


class LaborConfig(BaseModel):
    """노동시장"""
    unemployment_insurance_rate: float = 0.60
    unemployment_insurance_months: int = 6
    base_job_creation_rate: float = 0.02
    base_job_destruction_rate: float = 0.015
    reemployment_base_prob: float = 0.15
    reemployment_age_penalty: float = 0.005
    forced_sale_months: int = 12
    min_living_cost: float = 200.0
    income_growth_employed: float = 0.003


class MacroEconomyConfig(BaseModel):
    """거시경제"""
    gdp_growth_mean: float = 0.025
    gdp_growth_persistence: float = 0.8
    gdp_growth_volatility: float = 0.01
    income_gdp_beta: float = 0.8
    initial_inflation: float = 0.02
    initial_gdp_growth: float = 0.025


class BehavioralConfig(BaseModel):
    """행동경제학 기본 파라미터"""
    fomo_trigger_threshold: float = 0.05
    fomo_intensity: float = 50.0
    loss_aversion_mean: float = 2.5
    loss_aversion_std: float = 0.35
    herding_trigger: float = 0.03
    herding_intensity: float = 10.0
    social_learning_rate: float = 0.1
    news_impact: float = 0.2


class ProspectTheoryConfig(BaseModel):
    """전망이론"""
    alpha: float = 0.88
    beta: float = 0.88
    lambda_general: float = 2.25
    lambda_realestate: float = 2.5
    gamma_gain: float = 0.61
    gamma_loss: float = 0.69
    reference_point_decay: float = 0.008


class DiscountingConfig(BaseModel):
    """시간 할인"""
    beta_mean: float = 0.7
    beta_std: float = 0.1
    delta_mean: float = 0.99
    delta_std: float = 0.005
    investment_horizon: int = 60


class PerceptionConfig(BaseModel):
    """인지 제한 파라미터"""
    own_region_noise: float = 0.02       # 자기 동네 가격 노이즈 ±2%
    own_region_delay: int = 1            # 1개월 지연
    search_regions_max: int = 5          # 적극 탐색 최대 지역 수
    search_fatigue_rate: float = 0.1     # 탐색 피로 누적률
    neighbor_noise_add: float = 0.05     # 이웃 정보 추가 노이즈
    media_noise: float = 0.10            # 미디어 노이즈
    info_decay_months: int = 6           # 정보 유효 기간
    info_decay_rate: float = 0.15        # 월간 정보 품질 감쇠


class EmotionConfig(BaseModel):
    """감정 역학 파라미터"""
    anxiety_base_homeless: float = 0.02   # 무주택자 월간 불안 증가
    anxiety_price_sensitivity: float = 0.5
    anxiety_decay: float = 0.05           # 주택 보유 시 불안 감쇠
    fomo_accumulation_rate: float = 0.03  # FOMO 누적률
    fomo_decay_rate: float = 0.02         # FOMO 자연 감쇠
    regret_learning_rate: float = 0.1     # 후회 학습률
    satisfaction_base_owner: float = 0.6  # 주택 보유 기본 만족
    emotion_decision_threshold: float = 0.7  # 감정 의사결정 임계값


class ThinkingConfig(BaseModel):
    """이중처리 사고 파라미터"""
    system1_weight_calm: float = 0.3     # 차분할 때 System1 비중
    system1_weight_anxious: float = 0.8  # 불안할 때 System1 비중
    anchoring_strength: float = 0.5
    status_quo_bias_mean: float = 0.5
    status_quo_bias_std: float = 0.15
    mental_accounting_emergency_ratio: float = 0.20  # 비상금 비율
    mental_accounting_housing_ratio_homeless: float = 0.50
    mental_accounting_housing_ratio_owner: float = 0.20


class DecisionConfig(BaseModel):
    """의사결정 파라미터"""
    trigger_price_change: float = 0.05       # 가격 변동 트리거
    trigger_social_ratio: float = 0.30       # 사회적 압력 트리거
    trigger_anxiety_threshold: float = 0.7   # 불안 임계값
    trigger_fomo_threshold: float = 0.8      # FOMO 임계값
    trigger_homeless_months: int = 24        # 무주택 기간 트리거
    trigger_random_search_prob: float = 0.05 # 랜덤 탐색 확률
    satisficing_base_threshold: float = 0.6  # 만족화 기준
    delay_patience_weight: float = 0.5       # 인내심 가중치


class MemoryConfig(BaseModel):
    """기억과 학습"""
    max_decisions: int = 10              # 최근 기억 수
    experience_learning_rate: float = 0.05
    info_quality_growth: float = 0.01    # 경험에 따른 정보 품질 증가


class CognitiveConfig(BaseModel):
    """인지 아키텍처 전체"""
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class PersonalityDistribution(BaseModel):
    """성격 특성 분포"""
    risk_tolerance_alpha: float = 2.0
    risk_tolerance_beta: float = 5.0
    patience_mean: float = 0.5
    patience_std: float = 0.15
    social_conformity_alpha: float = 2.0
    social_conformity_beta: float = 3.0
    analytical_tendency_mean: float = 0.5
    analytical_tendency_std: float = 0.2
    fomo_sensitivity_alpha: float = 3.0
    fomo_sensitivity_beta: float = 3.0


class AgentCompositionConfig(BaseModel):
    """에이전트 구성"""
    investor_ratio: float = 0.15
    speculator_ratio: float = 0.05
    initial_homeless_rate: float = 0.43
    initial_one_house_rate: float = 0.42
    income_median: float = 400.0
    income_sigma: float = 0.65
    asset_median: float = 30000.0
    asset_alpha: float = 1.16
    age_young_ratio: float = 0.45
    age_middle_ratio: float = 0.43
    age_senior_ratio: float = 0.12
    personality: PersonalityDistribution = Field(default_factory=PersonalityDistribution)


class LifeCycleConfig(BaseModel):
    """생애주기"""
    marriage_urgency_age_start: int = 28
    marriage_urgency_age_end: int = 35
    newlywed_housing_pressure: float = 1.5
    parenting_housing_pressure: float = 1.3
    school_transition_age_start: int = 10
    school_transition_age_end: int = 15
    school_district_premium: float = 1.2
    retirement_start_age: int = 55
    downsizing_probability: float = 0.1


class NetworkConfig(BaseModel):
    """소셜 네트워크"""
    avg_neighbors: int = 10
    rewiring_prob: float = 0.1
    max_neighbors: int = 20
    self_weight: float = 0.6
    neighbor_weight: float = 0.4
    cascade_threshold: float = 0.3
    cascade_multiplier: float = 2.0


class SupplyConfig(BaseModel):
    """주택 공급"""
    base_supply_rate: float = 0.001
    price_threshold: float = 0.05
    redevelopment_age_threshold: int = 30
    redevelopment_price_threshold: float = 0.30
    construction_period: int = 24
    max_construction_ratio: float = 0.02
    depreciation_rate: float = 0.003
    min_condition: float = 0.3
    demolition_threshold: float = 0.35
    disaster_rate: float = 0.0001


class MarketConfig(BaseModel):
    """시장 파라미터"""
    price_sensitivity: float = 0.008
    expectation_weight: float = 0.010
    base_appreciation: float = 0.0015
    spillover_rate: float = 0.005
    buy_threshold: float = 0.08
    sell_threshold: float = 0.12
    use_double_auction: bool = True
    max_orders: int = 100000


class SimulationConfig(BaseModel):
    """시뮬레이션 마스터 설정"""
    name: str = "default"
    num_households: int = 100000
    num_steps: int = 120
    seed: int = 42
    world_file: str = "world.json"
    institutions_file: str = "institutions.json"
    agents_file: str = "agents.json"


class AgentsPresetConfig(BaseModel):
    """에이전트 프리셋"""
    composition: AgentCompositionConfig = Field(default_factory=AgentCompositionConfig)
    lifecycle: LifeCycleConfig = Field(default_factory=LifeCycleConfig)
    behavioral: BehavioralConfig = Field(default_factory=BehavioralConfig)
    prospect_theory: ProspectTheoryConfig = Field(default_factory=ProspectTheoryConfig)
    discounting: DiscountingConfig = Field(default_factory=DiscountingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)


class ScenarioConfig(BaseModel):
    """최상위 시나리오 설정 (모든 것을 통합)"""
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    institutions: InstitutionsConfig = Field(default_factory=InstitutionsConfig)
    agents: AgentsPresetConfig = Field(default_factory=AgentsPresetConfig)
    labor: LaborConfig = Field(default_factory=LaborConfig)
    macro: MacroEconomyConfig = Field(default_factory=MacroEconomyConfig)
    supply: SupplyConfig = Field(default_factory=SupplyConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
