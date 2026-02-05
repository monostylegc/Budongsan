/**
 * 시뮬레이션 관련 타입 정의
 */

// 분포 파라미터 (평균 + 표준편차)
export interface DistParam {
  mean: number;
  std: number;
}

// 정책 파라미터
export interface PolicyParams {
  ltv_1house: number;
  ltv_2house: number;
  ltv_3house: number;
  dti_limit: number;
  dsr_limit: number;
  acq_tax_1house: number;
  acq_tax_2house: number;
  acq_tax_3house: number;
  transfer_tax_short: number;
  transfer_tax_long: number;
  transfer_tax_multi_short: number;
  transfer_tax_multi_long: number;
  jongbu_threshold_1house: number;
  jongbu_threshold_multi: number;
  jongbu_rate: number;
  interest_rate: number;
  mortgage_spread: number;
  jeonse_loan_limit: number;
  rent_increase_cap: number;
}

// 행동경제학 파라미터 (분포 지원)
export interface BehavioralParams {
  // FOMO (분포)
  fomo_sensitivity: DistParam;     // FOMO 민감도
  fomo_trigger_threshold: number;  // 발동 임계값 (환경)

  // 손실 회피 (분포)
  loss_aversion: DistParam;

  // 앵커링 (분포)
  anchoring_strength: DistParam;
  anchoring_threshold: number;     // 발동 이익률 (환경)

  // 군집 행동 (분포)
  herding_tendency: DistParam;
  herding_trigger: number;         // 발동 비율 (환경)

  // 위험 허용도 (분포)
  risk_tolerance: DistParam;

  // 시간 할인 (분포) - 현재 편향
  present_bias: DistParam;         // beta (0.5~1.0)

  // 사회적 학습
  social_learning_rate: number;
  news_impact: number;
}

// 에이전트 구성 파라미터
export interface AgentCompositionParams {
  // 유형 비율
  investor_ratio: number;
  speculator_ratio: number;

  // 투기자 특성
  speculator_risk_multiplier: number;
  speculator_fomo_multiplier: number;
  speculator_horizon_min: number;
  speculator_horizon_max: number;

  // 초기 주택 보유
  initial_homeless_rate: number;
  initial_one_house_rate: number;
  initial_multi_house_rate: number;

  // 소득 분포 (로그정규)
  income_median: number;      // 중위값 (만원/월)
  income_sigma: number;       // 로그 표준편차 (분산도)

  // 자산 분포 (파레토)
  asset_median: number;       // 중위값 (만원)
  asset_alpha: number;        // 파레토 알파 (낮을수록 불평등)

  // 연령 분포
  age_young_ratio: number;    // 25-34세 비율
  age_middle_ratio: number;   // 35-54세 비율
  age_senior_ratio: number;   // 55세+ 비율
}

// 생애주기 파라미터
export interface LifeCycleParams {
  marriage_urgency_age_start: number;
  marriage_urgency_age_end: number;
  newlywed_housing_pressure: number;
  parenting_housing_pressure: number;
  school_transition_age_start: number;
  school_transition_age_end: number;
  school_district_premium: number;
  retirement_start_age: number;
  downsizing_probability: number;
}

// 네트워크 파라미터
export interface NetworkParams {
  avg_neighbors: number;
  rewiring_prob: number;
  cascade_threshold: number;
  cascade_multiplier: number;
  self_weight: number;
}

// 거시경제 파라미터
export interface MacroParams {
  m2_growth: number;
  gdp_growth_mean: number;
  gdp_growth_volatility: number;
  inflation_target: number;
  income_gdp_beta: number;
}

// 공급 파라미터
export interface SupplyParams {
  base_supply_rate: number;
  elasticity_gangnam: number;
  elasticity_seoul: number;
  elasticity_gyeonggi: number;
  elasticity_local: number;
  redevelopment_base_prob: number;
  redevelopment_age_threshold: number;
  construction_period: number;
}

// 노후화 파라미터
export interface DepreciationParams {
  depreciation_rate: number;
  natural_demolition_threshold: number;
  disaster_rate: number;
}

// 시장 파라미터
export interface MarketParams {
  price_sensitivity: number;
  expectation_weight: number;
  base_appreciation: number;
  buy_threshold: number;
  sell_threshold: number;
  spillover_rate: number;
}

// 시뮬레이션 파라미터 전체
export interface SimulationParams {
  num_households: number;
  num_houses: number;
  num_steps: number;
  seed: number;
  policy: PolicyParams;
  behavioral: BehavioralParams;
  agent_composition: AgentCompositionParams;
  lifecycle: LifeCycleParams;
  network: NetworkParams;
  macro: MacroParams;
  supply: SupplyParams;
  depreciation: DepreciationParams;
  market: MarketParams;
  scenario: string;
}

// 지역별 통계
export interface RegionStats {
  region_id: number;
  name: string;
  price: number;
  price_change: number;
  transactions: number;
  demand: number;
  supply: number;
  jeonse_ratio: number;
  homeless_count: number;
  one_house_count: number;
  multi_house_count: number;
}

// 거래 정보
export interface Transaction {
  region_id: number;
  region_name: string;
  count: number;
  avg_price: number;
  lat: number;
  lng: number;
}

// 월별 상태
export interface MonthlyState {
  month: number;
  year: number;
  avg_price: number;
  total_transactions: number;
  homeowner_rate: number;
  multi_owner_rate: number;
  demand_supply_ratio: number;
  listing_rate: number;
  interest_rate: number;
  inflation: number;
  gdp_growth: number;
  m2_growth: number;
  mean_building_age: number;
  mean_building_condition: number;
  demolished_count: number;
  regions: RegionStats[];
  recent_transactions: Transaction[];
}

// WebSocket 메시지
export type WSMessageType = 'status' | 'state' | 'completed' | 'error' | 'stopped';

export interface WSMessage {
  type: WSMessageType;
  message?: string;
  data?: MonthlyState;
  summary?: SimulationSummary;
  month?: number;
}

export interface SimulationSummary {
  duration_months: number;
  price_change: { gangnam: number; seoul_avg: number; national_avg: number };
  homeowner_rate: { initial: number; final: number };
  total_transactions: number;
}

export type SimulationStatus = 'idle' | 'connecting' | 'initializing' | 'running' | 'paused' | 'completed' | 'error' | 'stopped';

// 지역 좌표
export const REGION_COORDS: Record<number, { lat: number; lng: number }> = {
  0: { lat: 37.517, lng: 127.047 }, 1: { lat: 37.556, lng: 127.010 },
  2: { lat: 37.570, lng: 126.977 }, 3: { lat: 37.359, lng: 127.105 },
  4: { lat: 37.275, lng: 127.009 }, 5: { lat: 37.742, lng: 127.047 },
  6: { lat: 37.456, lng: 126.705 }, 7: { lat: 35.180, lng: 129.076 },
  8: { lat: 35.871, lng: 128.602 }, 9: { lat: 35.160, lng: 126.851 },
  10: { lat: 36.351, lng: 127.385 }, 11: { lat: 36.480, lng: 127.289 },
  12: { lat: 35.800, lng: 127.800 },
};

export const REGION_NAMES: Record<number, string> = {
  0: '강남3구', 1: '마용성', 2: '기타서울', 3: '분당판교',
  4: '경기남부', 5: '경기북부', 6: '인천', 7: '부산',
  8: '대구', 9: '광주', 10: '대전', 11: '세종', 12: '기타지방',
};
