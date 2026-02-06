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
  fomo_sensitivity: DistParam;
  fomo_trigger_threshold: number;
  loss_aversion: DistParam;
  anchoring_strength: DistParam;
  anchoring_threshold: number;
  herding_tendency: DistParam;
  herding_trigger: number;
  risk_tolerance: DistParam;
  present_bias: DistParam;
  social_learning_rate: number;
  news_impact: number;
}

// 에이전트 구성 파라미터
export interface AgentCompositionParams {
  investor_ratio: number;
  speculator_ratio: number;
  speculator_risk_multiplier: number;
  speculator_fomo_multiplier: number;
  speculator_horizon_min: number;
  speculator_horizon_max: number;
  initial_homeless_rate: number;
  initial_one_house_rate: number;
  initial_multi_house_rate: number;
  income_median: number;
  income_sigma: number;
  asset_median: number;
  asset_alpha: number;
  age_young_ratio: number;
  age_middle_ratio: number;
  age_senior_ratio: number;
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
  unemployment_rate: number;
  at_risk_households: number;
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

// 게임맵 타일 레이아웃 (한반도 근사 배치)
export interface TileInfo {
  id: number;
  label: string;
  col: number;
  row: number;
}

export const TILE_LAYOUT: TileInfo[] = [
  // 서울권 (상단 밀집)
  { id: 5, label: '경기북', col: 1, row: 0 },
  { id: 2, label: '기타서울', col: 2, row: 0 },
  { id: 6, label: '인천', col: 0, row: 1 },
  { id: 1, label: '마용성', col: 2, row: 1 },
  { id: 0, label: '강남3구', col: 3, row: 1 },
  { id: 4, label: '경기남', col: 1, row: 2 },
  { id: 3, label: '분당판교', col: 2, row: 2 },
  // 중부
  { id: 10, label: '대전', col: 2, row: 3 },
  { id: 11, label: '세종', col: 3, row: 3 },
  // 남부
  { id: 8, label: '대구', col: 3, row: 4 },
  { id: 9, label: '광주', col: 1, row: 5 },
  { id: 12, label: '기타지방', col: 2, row: 5 },
  { id: 7, label: '부산', col: 3, row: 5 },
];

// 지역 인접 관계 (풍선효과 연결선)
export const ADJACENCY: [number, number][] = [
  [0, 1], [0, 3], [1, 2], [2, 5], [2, 6],
  [3, 4], [4, 5], [4, 6], [4, 10],
  [7, 8], [8, 10], [9, 12], [10, 11], [10, 12],
];

export const REGION_NAMES: Record<number, string> = {
  0: '강남3구', 1: '마용성', 2: '기타서울', 3: '분당판교',
  4: '경기남부', 5: '경기북부', 6: '인천', 7: '부산',
  8: '대구', 9: '광주', 10: '대전', 11: '세종', 12: '기타지방',
};
