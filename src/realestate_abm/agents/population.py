"""에이전트 인구 - NumPy Structure of Arrays 컨테이너

기존 Taichi ti.field 기반 → 순수 NumPy 배열로 전환.
100K+ 에이전트를 벡터화 연산으로 처리.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class AgentArrays:
    """모든 에이전트 데이터를 NumPy 배열로 저장 (SoA 패턴)"""
    n: int  # 에이전트 수

    # === 기본 속성 ===
    age: np.ndarray = field(default=None)               # (n,) int32
    income: np.ndarray = field(default=None)             # (n,) float32 월소득(만원)
    region: np.ndarray = field(default=None)             # (n,) int32 현재 거주 지역
    agent_type: np.ndarray = field(default=None)         # (n,) int32 AgentType enum

    # === 심리적 회계 (3개 계좌) ===
    housing_fund: np.ndarray = field(default=None)       # (n,) float32 주거자금
    emergency_fund: np.ndarray = field(default=None)     # (n,) float32 비상자금
    investment_fund: np.ndarray = field(default=None)    # (n,) float32 투자자금

    # === 주택 보유 ===
    owned_houses: np.ndarray = field(default=None)       # (n,) int32
    primary_house_id: np.ndarray = field(default=None)   # (n,) int32 (-1=무주택)
    purchase_price: np.ndarray = field(default=None)     # (n,) float32
    purchase_month: np.ndarray = field(default=None)     # (n,) int32
    total_purchase_price: np.ndarray = field(default=None)  # (n,) float32

    # === 고용 ===
    employment_status: np.ndarray = field(default=None)  # (n,) int32
    industry: np.ndarray = field(default=None)           # (n,) int32
    unemployment_months: np.ndarray = field(default=None)  # (n,) int32
    previous_income: np.ndarray = field(default=None)    # (n,) float32

    # === 생애주기 ===
    is_married: np.ndarray = field(default=None)         # (n,) int32
    num_children: np.ndarray = field(default=None)       # (n,) int32
    eldest_child_age: np.ndarray = field(default=None)   # (n,) int32 (-1=없음)
    life_stage: np.ndarray = field(default=None)         # (n,) int32

    # === 성격 특성 (안정적, 초기화 시 설정) ===
    risk_tolerance: np.ndarray = field(default=None)     # (n,) float32
    patience: np.ndarray = field(default=None)           # (n,) float32
    social_conformity: np.ndarray = field(default=None)  # (n,) float32
    status_quo_bias: np.ndarray = field(default=None)    # (n,) float32
    loss_aversion: np.ndarray = field(default=None)      # (n,) float32 1.5~3.5
    fomo_sensitivity: np.ndarray = field(default=None)   # (n,) float32
    analytical_tendency: np.ndarray = field(default=None) # (n,) float32

    # === 인지 상태 (매 스텝 변동) ===
    # 인지 (perception)
    known_prices: np.ndarray = field(default=None)       # (n, n_regions) 에이전트가 아는 가격
    price_info_age: np.ndarray = field(default=None)     # (n, n_regions) 정보 노후도(개월)
    info_quality: np.ndarray = field(default=None)       # (n,) 정보 수준 0~1
    search_fatigue: np.ndarray = field(default=None)     # (n,) 탐색 피로도

    # 감정 (emotions)
    anxiety: np.ndarray = field(default=None)            # (n,) 불안감 0~1
    satisfaction: np.ndarray = field(default=None)        # (n,) 만족도 0~1
    fomo_level: np.ndarray = field(default=None)         # (n,) FOMO 누적 0~1
    regret: np.ndarray = field(default=None)             # (n,) 후회 0~1

    # 기억 (memory)
    past_decisions: np.ndarray = field(default=None)     # (n, 10) 최근 결정
    past_outcomes: np.ndarray = field(default=None)      # (n, 10) 결과
    decision_count: np.ndarray = field(default=None)     # (n,) 총 결정 횟수

    # === 의사결정 상태 ===
    wants_to_buy: np.ndarray = field(default=None)       # (n,) int32 0/1
    wants_to_sell: np.ndarray = field(default=None)      # (n,) int32 0/1
    target_region: np.ndarray = field(default=None)      # (n,) int32
    homeless_months: np.ndarray = field(default=None)    # (n,) int32
    is_triggered: np.ndarray = field(default=None)       # (n,) int32 트리거 여부

    # === 전망이론 ===
    reference_price: np.ndarray = field(default=None)    # (n,) float32
    discount_beta: np.ndarray = field(default=None)      # (n,) float32
    discount_delta: np.ndarray = field(default=None)     # (n,) float32

    # === 네트워크 ===
    neighbors: np.ndarray = field(default=None)          # (n, max_neighbors) int32
    num_neighbors: np.ndarray = field(default=None)      # (n,) int32
    neighbor_buying_ratio: np.ndarray = field(default=None)  # (n,) float32
    network_belief: np.ndarray = field(default=None)     # (n,) float32

    # === 대출 ===
    mortgage_balance: np.ndarray = field(default=None)   # (n,) float32
    jeonse_deposit: np.ndarray = field(default=None)     # (n,) float32

    # === 부모 지원 ===
    parent_support: np.ndarray = field(default=None)     # (n,) float32

    # === 강제매도 ===
    forced_sale_countdown: np.ndarray = field(default=None)  # (n,) int32
    housing_cost_unpaid: np.ndarray = field(default=None)    # (n,) int32

    # === 투기자 ===
    speculation_horizon: np.ndarray = field(default=None)  # (n,) int32

    # === 중간 점수 (의사결정 파이프라인) ===
    buy_score_affordability: np.ndarray = field(default=None)
    buy_score_lifecycle: np.ndarray = field(default=None)
    buy_score_behavioral: np.ndarray = field(default=None)
    buy_score_market: np.ndarray = field(default=None)
    buy_score_policy: np.ndarray = field(default=None)
    is_affordable_flag: np.ndarray = field(default=None)


class AgentPopulation:
    """에이전트 인구 관리자

    NumPy 배열 기반 Structure of Arrays 패턴.
    모든 연산은 벡터화되어 100K+ 에이전트를 효율적으로 처리.
    """

    def __init__(self, n: int, n_regions: int, max_neighbors: int = 20, max_decisions: int = 10):
        self.n = n
        self.n_regions = n_regions
        self.max_neighbors = max_neighbors
        self.max_decisions = max_decisions

        self.data = AgentArrays(n=n)
        self._allocate_arrays()

    def _allocate_arrays(self):
        """모든 배열 할당"""
        n = self.n
        nr = self.n_regions
        mn = self.max_neighbors
        md = self.max_decisions
        d = self.data

        # 기본 속성
        d.age = np.zeros(n, dtype=np.int32)
        d.income = np.zeros(n, dtype=np.float32)
        d.region = np.zeros(n, dtype=np.int32)
        d.agent_type = np.zeros(n, dtype=np.int32)

        # 심리적 회계
        d.housing_fund = np.zeros(n, dtype=np.float32)
        d.emergency_fund = np.zeros(n, dtype=np.float32)
        d.investment_fund = np.zeros(n, dtype=np.float32)

        # 주택 보유
        d.owned_houses = np.zeros(n, dtype=np.int32)
        d.primary_house_id = np.full(n, -1, dtype=np.int32)
        d.purchase_price = np.zeros(n, dtype=np.float32)
        d.purchase_month = np.zeros(n, dtype=np.int32)
        d.total_purchase_price = np.zeros(n, dtype=np.float32)

        # 고용
        d.employment_status = np.zeros(n, dtype=np.int32)
        d.industry = np.zeros(n, dtype=np.int32)
        d.unemployment_months = np.zeros(n, dtype=np.int32)
        d.previous_income = np.zeros(n, dtype=np.float32)

        # 생애주기
        d.is_married = np.zeros(n, dtype=np.int32)
        d.num_children = np.zeros(n, dtype=np.int32)
        d.eldest_child_age = np.full(n, -1, dtype=np.int32)
        d.life_stage = np.zeros(n, dtype=np.int32)

        # 성격 특성
        d.risk_tolerance = np.zeros(n, dtype=np.float32)
        d.patience = np.zeros(n, dtype=np.float32)
        d.social_conformity = np.zeros(n, dtype=np.float32)
        d.status_quo_bias = np.zeros(n, dtype=np.float32)
        d.loss_aversion = np.full(n, 2.5, dtype=np.float32)
        d.fomo_sensitivity = np.zeros(n, dtype=np.float32)
        d.analytical_tendency = np.full(n, 0.5, dtype=np.float32)

        # 인지
        d.known_prices = np.zeros((n, nr), dtype=np.float32)
        d.price_info_age = np.full((n, nr), 12, dtype=np.int32)  # 초기: 오래된 정보
        d.info_quality = np.zeros(n, dtype=np.float32)
        d.search_fatigue = np.zeros(n, dtype=np.float32)

        # 감정
        d.anxiety = np.zeros(n, dtype=np.float32)
        d.satisfaction = np.full(n, 0.5, dtype=np.float32)
        d.fomo_level = np.zeros(n, dtype=np.float32)
        d.regret = np.zeros(n, dtype=np.float32)

        # 기억
        d.past_decisions = np.zeros((n, md), dtype=np.int32)
        d.past_outcomes = np.zeros((n, md), dtype=np.float32)
        d.decision_count = np.zeros(n, dtype=np.int32)

        # 의사결정
        d.wants_to_buy = np.zeros(n, dtype=np.int32)
        d.wants_to_sell = np.zeros(n, dtype=np.int32)
        d.target_region = np.zeros(n, dtype=np.int32)
        d.homeless_months = np.zeros(n, dtype=np.int32)
        d.is_triggered = np.zeros(n, dtype=np.int32)

        # 전망이론
        d.reference_price = np.zeros(n, dtype=np.float32)
        d.discount_beta = np.full(n, 0.7, dtype=np.float32)
        d.discount_delta = np.full(n, 0.99, dtype=np.float32)

        # 네트워크
        d.neighbors = np.full((n, mn), -1, dtype=np.int32)
        d.num_neighbors = np.zeros(n, dtype=np.int32)
        d.neighbor_buying_ratio = np.zeros(n, dtype=np.float32)
        d.network_belief = np.zeros(n, dtype=np.float32)

        # 대출
        d.mortgage_balance = np.zeros(n, dtype=np.float32)
        d.jeonse_deposit = np.zeros(n, dtype=np.float32)

        # 부모 지원
        d.parent_support = np.zeros(n, dtype=np.float32)

        # 강제매도
        d.forced_sale_countdown = np.full(n, -1, dtype=np.int32)
        d.housing_cost_unpaid = np.zeros(n, dtype=np.int32)

        # 투기자
        d.speculation_horizon = np.zeros(n, dtype=np.int32)

        # 중간 점수
        d.buy_score_affordability = np.zeros(n, dtype=np.float32)
        d.buy_score_lifecycle = np.zeros(n, dtype=np.float32)
        d.buy_score_behavioral = np.zeros(n, dtype=np.float32)
        d.buy_score_market = np.zeros(n, dtype=np.float32)
        d.buy_score_policy = np.zeros(n, dtype=np.float32)
        d.is_affordable_flag = np.zeros(n, dtype=np.int32)

    @property
    def total_assets(self) -> np.ndarray:
        """총 자산 = 주거자금 + 비상자금 + 투자자금"""
        d = self.data
        return d.housing_fund + d.emergency_fund + d.investment_fund

    @property
    def available_for_housing(self) -> np.ndarray:
        """주택 구입 가용 자산 = 주거자금 + 투자자금*0.5"""
        d = self.data
        return d.housing_fund + d.investment_fund * 0.5

    def distribute_income(self, cfg_thinking):
        """월소득을 3개 계좌에 배분

        Args:
            cfg_thinking: ThinkingConfig (비상금/주거자금 비율)
        """
        d = self.data
        income = d.income
        employed = d.employment_status == 0

        # 비상금 목표: 6개월치 소득
        emergency_target = income * 6.0
        emergency_deficit = np.maximum(emergency_target - d.emergency_fund, 0)

        # 비상금 채우기 (소득의 20%까지)
        emergency_alloc = np.minimum(
            income * cfg_thinking.mental_accounting_emergency_ratio,
            emergency_deficit
        )
        emergency_alloc[~employed] = 0  # 실업자는 비상금 적립 안함

        remaining = income - emergency_alloc

        # 주거자금 배분
        housing_ratio = np.where(
            d.owned_houses == 0,
            cfg_thinking.mental_accounting_housing_ratio_homeless,
            cfg_thinking.mental_accounting_housing_ratio_owner,
        )
        housing_alloc = remaining * housing_ratio

        # 투자자금 = 나머지
        investment_alloc = remaining - housing_alloc

        # 실업자는 비상금에서 생활비 인출
        d.emergency_fund[employed] += emergency_alloc[employed]
        d.housing_fund[employed] += housing_alloc[employed]
        d.investment_fund[employed] += investment_alloc[employed]

    def get_region_mask(self, region_idx: int) -> np.ndarray:
        """특정 지역 에이전트 마스크"""
        return self.data.region == region_idx

    def get_homeless_mask(self) -> np.ndarray:
        return self.data.owned_houses == 0

    def get_employed_mask(self) -> np.ndarray:
        return self.data.employment_status == 0
