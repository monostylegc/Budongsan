"""가구 에이전트 정의 (Taichi fields) - 행동경제학 기반"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, REGION_HOUSEHOLD_RATIO, REGION_PRESTIGE, REGION_JOB_DENSITY, ADJACENCY


# =============================================================================
# Prospect Theory Functions (Kahneman & Tversky, 1992)
# =============================================================================

# =============================================================================
# DSR-based Affordability Functions (통일된 구매력 체계)
# =============================================================================

@ti.func
def calculate_dsr(
    price: ti.f32,
    asset: ti.f32,
    annual_income: ti.f32,
    interest_rate: ti.f32,
    asset_utilization: ti.f32,
    loan_term_years: ti.i32
) -> ti.f32:
    """DSR(부채상환비율) 계산

    DSR = 연간 원리금 상환액 / 연간 소득

    Args:
        price: 주택 가격 (만원)
        asset: 순자산 (만원)
        annual_income: 연간 소득 (만원)
        interest_rate: 연이율
        asset_utilization: 자산 활용 비율 (0.5~0.7)
        loan_term_years: 대출 기간 (년)

    Returns:
        DSR 비율 (0~1+)
    """
    dsr = 0.0

    # 필요 대출액 = 주택가격 - 활용 가능 자산
    available_asset = asset * asset_utilization
    required_loan = ti.max(price - available_asset, 0.0)

    # 대출 필요하고 소득 있는 경우에만 계산
    if required_loan > 0 and annual_income > 0:
        # 월간 이자율
        monthly_rate = interest_rate / 12.0
        n_payments = loan_term_years * 12

        # 원리금균등상환 월 상환액 계산
        # PMT = P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_payment = 0.0
        if monthly_rate > 0:
            factor = ti.pow(1.0 + monthly_rate, ti.cast(n_payments, ti.f32))
            monthly_payment = required_loan * monthly_rate * factor / (factor - 1.0)
        else:
            monthly_payment = required_loan / ti.cast(n_payments, ti.f32)

        # 연간 상환액
        annual_payment = monthly_payment * 12.0

        # DSR = 연간 상환액 / 연간 소득
        dsr = annual_payment / annual_income

    return dsr


@ti.func
def check_dsr_affordability(
    dsr: ti.f32,
    agent_type: ti.i32,
    is_wealthy: ti.i32,
    job_density: ti.f32,
    dsr_limit_end_user: ti.f32,
    dsr_limit_investor: ti.f32,
    dsr_limit_speculator: ti.f32,
    allow_stretched: ti.i32,
    stretched_multiplier: ti.f32
) -> ti.i32:
    """DSR 기반 구매 가능 여부 판단

    에이전트 유형별, 자산 수준별, 지역별 차등 적용

    Args:
        dsr: 계산된 DSR
        agent_type: 0=실수요자, 1=투자자, 2=투기자
        is_wealthy: 고자산가 여부
        job_density: 지역 일자리 밀도
        dsr_limit_*: 유형별 DSR 한도
        allow_stretched: 영끌 허용 여부
        stretched_multiplier: 영끌 시 한도 배율

    Returns:
        1=구매 가능, 0=구매 불가
    """
    # 에이전트 유형별 기본 DSR 한도
    dsr_limit = dsr_limit_end_user  # 기본: 실수요자
    if agent_type == 1:
        dsr_limit = dsr_limit_investor
    elif agent_type == 2:
        dsr_limit = dsr_limit_speculator

    # 고자산가는 DSR 부담이 낮음 (자산 활용 비율이 높아서)
    # 이미 calculate_dsr에서 asset_utilization으로 반영됨

    # 영끌 허용: 일자리 밀도 높은 지역에서 DSR 한도 상향
    if allow_stretched == 1 and job_density >= 0.5:
        dsr_limit = dsr_limit * stretched_multiplier

    # 판정: DSR이 한도 이내면 1(구매 가능), 아니면 0(구매 불가)
    result = 0
    if dsr <= dsr_limit:
        result = 1

    return result


# =============================================================================
# Policy Cost Functions (명시적 비용 계산)
# =============================================================================

@ti.func
def calculate_acquisition_tax(price: ti.f32, owned_houses: ti.i32) -> ti.f32:
    """취득세 계산 (주택 보유 수에 따른 누진세)

    한국 취득세율:
    - 1주택: 1-3% (평균 1%)
    - 2주택: 8%
    - 3주택+: 12%

    Args:
        price: 주택 가격 (만원)
        owned_houses: 현재 보유 주택 수 (매수 후 기준)

    Returns:
        취득세 금액 (만원)
    """
    tax_rate = 0.01  # 1주택 기본
    if owned_houses == 2:
        tax_rate = 0.08  # 2주택
    elif owned_houses >= 3:
        tax_rate = 0.12  # 3주택 이상
    return price * tax_rate


@ti.func
def calculate_annual_jongbu_tax(
    total_house_value: ti.f32,
    owned_houses: ti.i32,
    jongbu_threshold_1house: ti.f32,
    jongbu_threshold_multi: ti.f32,
    jongbu_rate: ti.f32
) -> ti.f32:
    """종합부동산세 연간 비용 계산

    한국 종부세:
    - 1주택: 11억 초과분에 대해 과세
    - 다주택: 6억 초과분에 대해 합산 과세, 세율 가중

    Args:
        total_house_value: 보유 주택 총 가치 (만원)
        owned_houses: 보유 주택 수
        jongbu_threshold_1house: 1주택 종부세 기준 (만원)
        jongbu_threshold_multi: 다주택 종부세 기준 (만원)
        jongbu_rate: 종부세율

    Returns:
        연간 종부세 금액 (만원)
    """
    tax = 0.0
    if owned_houses == 1:
        if total_house_value > jongbu_threshold_1house:
            tax = (total_house_value - jongbu_threshold_1house) * jongbu_rate
    elif owned_houses >= 2:
        if total_house_value > jongbu_threshold_multi:
            # 다주택자는 세율 가중 (1.5배)
            tax = (total_house_value - jongbu_threshold_multi) * jongbu_rate * 1.5
    return tax


@ti.func
def calculate_expected_profit(
    price: ti.f32,
    expected_appreciation: ti.f32,
    holding_period: ti.i32,
    rental_yield: ti.f32
) -> ti.f32:
    """예상 투자 수익 계산 (취득 후 보유 기간 동안)

    Args:
        price: 주택 가격 (만원)
        expected_appreciation: 월간 예상 가격 상승률
        holding_period: 예상 보유 기간 (월)
        rental_yield: 월간 임대 수익률

    Returns:
        예상 총 수익 (만원)
    """
    # 시세차익 (복리)
    final_price = price * ti.pow(1.0 + expected_appreciation, ti.cast(holding_period, ti.f32))
    capital_gain = final_price - price

    # 임대수익 (단리)
    rental_income = price * rental_yield * ti.cast(holding_period, ti.f32)

    return capital_gain + rental_income


@ti.func
def calculate_investment_profitability(
    price: ti.f32,
    owned_after_purchase: ti.i32,
    expected_appreciation: ti.f32,
    holding_period: ti.i32,
    rental_yield: ti.f32,
    jongbu_threshold_1house: ti.f32,
    jongbu_threshold_multi: ti.f32,
    jongbu_rate: ti.f32,
    transfer_tax_rate: ti.f32,
    current_total_value: ti.f32
) -> ti.f32:
    """투자 수익성 계산 (예상 수익 - 세금 비용)

    경제적 합리성 기반 의사결정:
    - 수익성 > 0: 투자 가치 있음
    - 수익성 <= 0: 투자 가치 없음 (세금이 수익보다 큼)

    Args:
        price: 매수 가격 (만원)
        owned_after_purchase: 매수 후 보유 주택 수
        expected_appreciation: 월간 예상 가격 상승률
        holding_period: 예상 보유 기간 (월)
        rental_yield: 월간 임대 수익률
        jongbu_threshold_*: 종부세 기준
        jongbu_rate: 종부세율
        transfer_tax_rate: 양도세율
        current_total_value: 현재 보유 주택 총 가치

    Returns:
        투자 수익성 (양수면 수익, 음수면 손실)
    """
    # 1. 취득세 (매수 시 즉시 비용)
    acq_tax = calculate_acquisition_tax(price, owned_after_purchase)

    # 2. 종부세 (보유 기간 동안 연간 비용)
    new_total_value = current_total_value + price
    annual_jongbu = calculate_annual_jongbu_tax(
        new_total_value, owned_after_purchase,
        jongbu_threshold_1house, jongbu_threshold_multi, jongbu_rate
    )
    holding_years = ti.cast(holding_period, ti.f32) / 12.0
    total_jongbu = annual_jongbu * holding_years

    # 3. 예상 수익 (시세차익 + 임대수익)
    expected_profit = calculate_expected_profit(
        price, expected_appreciation, holding_period, rental_yield
    )

    # 4. 예상 양도세 (매도 시 시세차익에 대해)
    final_price = price * ti.pow(1.0 + expected_appreciation, ti.cast(holding_period, ti.f32))
    capital_gain = final_price - price
    transfer_tax = capital_gain * transfer_tax_rate if capital_gain > 0 else 0.0

    # 5. 순수익 = 예상 수익 - (취득세 + 종부세 + 양도세)
    net_profit = expected_profit - acq_tax - total_jongbu - transfer_tax

    return net_profit


@ti.func
def prospect_value(x: ti.f32, alpha: ti.f32, beta: ti.f32, lambda_: ti.f32) -> ti.f32:
    """전망이론 가치 함수 (Tversky & Kahneman, 1992)

    Args:
        x: 이득/손실 금액 (참조점 대비)
        alpha: 이득 곡률 파라미터 (0.88)
        beta: 손실 곡률 파라미터 (0.88)
        lambda_: 손실 회피 계수 (2.25 일반, 2.5 부동산)

    Returns:
        주관적 가치 (이득 영역은 오목, 손실 영역은 볼록)
    """
    value = 0.0
    if x >= 0:
        # 이득: v(x) = x^α (diminishing sensitivity)
        value = ti.pow(x + 1e-10, alpha)
    else:
        # 손실: v(x) = -λ * |x|^β (손실 회피 + diminishing sensitivity)
        value = -lambda_ * ti.pow(-x + 1e-10, beta)
    return value


@ti.func
def probability_weight(p: ti.f32, gamma: ti.f32) -> ti.f32:
    """확률 가중 함수 (Prelec, 1998)

    Args:
        p: 객관적 확률 (0~1)
        gamma: 가중 파라미터 (이득 0.61, 손실 0.69)

    Returns:
        주관적 확률 가중치 (작은 확률 과대평가, 큰 확률 과소평가)
    """
    # w(p) = exp(-(-ln(p))^γ)
    # 0과 1 근처에서 안정성을 위해 클리핑
    p_clipped = ti.math.clamp(p, 1e-6, 1.0 - 1e-6)
    neg_log_p = -ti.log(p_clipped)
    return ti.exp(-ti.pow(neg_log_p, gamma))


@ti.func
def prospect_utility(
    gain_loss: ti.f32,
    prob: ti.f32,
    alpha: ti.f32,
    beta: ti.f32,
    lambda_: ti.f32,
    gamma: ti.f32
) -> ti.f32:
    """전망이론 기대 효용 계산

    Args:
        gain_loss: 예상 이득/손실
        prob: 발생 확률
        alpha, beta, lambda_, gamma: 전망이론 파라미터

    Returns:
        전망이론 기반 기대 효용
    """
    v = prospect_value(gain_loss, alpha, beta, lambda_)
    w = probability_weight(prob, gamma)
    return w * v


# =============================================================================
# Hyperbolic Discounting Functions (Laibson, 1997)
# =============================================================================

@ti.func
def hyperbolic_discount(t: ti.i32, beta: ti.f32, delta: ti.f32) -> ti.f32:
    """준쌍곡선 할인 함수 (β-δ 모델, Laibson, 1997)

    Args:
        t: 미래 시점 (월)
        beta: 현재 편향 파라미터 (0.7)
        delta: 기하 할인율 (0.99)

    Returns:
        할인 계수 (0~1)
    """
    discount = 0.0
    if t == 0:
        discount = 1.0
    else:
        # D(t) = β * δ^t for t > 0
        discount = beta * ti.pow(delta, ti.cast(t, ti.f32))
    return discount


@ti.func
def discounted_future_value(
    monthly_values: ti.template(),
    horizon: ti.i32,
    beta: ti.f32,
    delta: ti.f32
) -> ti.f32:
    """미래 가치의 현재가치 합산 (준쌍곡선 할인)

    Args:
        monthly_values: 월별 예상 가치 배열
        horizon: 투자 기간 (월)
        beta: 현재 편향 파라미터
        delta: 기하 할인율

    Returns:
        할인된 미래 가치의 합
    """
    total = 0.0
    for t in range(horizon):
        discount = hyperbolic_discount(t, beta, delta)
        total += discount * monthly_values[t]
    return total


@ti.func
def expected_investment_return(
    expected_appreciation: ti.f32,
    expected_rental_yield: ti.f32,
    horizon: ti.i32,
    beta: ti.f32,
    delta: ti.f32
) -> ti.f32:
    """투자 기대 수익률 계산 (준쌍곡선 할인 적용)

    Args:
        expected_appreciation: 월간 예상 가격상승률
        expected_rental_yield: 월간 예상 임대수익률
        horizon: 투자 기간 (월)
        beta: 현재 편향 파라미터
        delta: 기하 할인율

    Returns:
        할인된 총 기대 수익률
    """
    total_return = 0.0
    for t in range(1, horizon + 1):
        # 월별 수익 = 가격상승 + 임대수익
        monthly_return = expected_appreciation + expected_rental_yield
        discount = hyperbolic_discount(t, beta, delta)
        total_return += discount * monthly_return
    return total_return


@ti.data_oriented
class Households:
    """가구 에이전트들을 관리하는 클래스 (Structure of Arrays)

    행동경제학 요소:
    - 손실 회피 (Loss Aversion): 손실 확정 회피
    - FOMO (Fear Of Missing Out): 가격 상승 시 매수 욕구 급증
    - 앵커링 (Anchoring): 매입가에 집착
    - 군집 행동 (Herding): 주변 사람들 행동 모방
    - 생애주기: 결혼/출산/학군/은퇴에 따른 주거 수요 변화
    """

    def __init__(self, config: Config):
        self.n = config.num_households
        self.config = config

        # 기본 속성
        self.age = ti.field(dtype=ti.i32, shape=self.n)
        self.income = ti.field(dtype=ti.f32, shape=self.n)  # 월소득 (만원)
        self.asset = ti.field(dtype=ti.f32, shape=self.n)   # 순자산 (만원)
        self.region = ti.field(dtype=ti.i32, shape=self.n)  # 현재 거주 지역

        # 주택 보유
        self.owned_houses = ti.field(dtype=ti.i32, shape=self.n)  # 보유 주택 수
        self.primary_house_id = ti.field(dtype=ti.i32, shape=self.n)  # 거주 주택 ID (-1이면 무주택)

        # 에이전트 유형 (JASSS 2020 한국 ABM 참고)
        # 0: 실수요자 (owner-occupier) - 거주 목적
        # 1: 투자자 (investor/buy-to-let) - 임대 수익 목적
        # 2: 투기자 (speculator) - 시세차익 목적, 단기 보유
        self.agent_type = ti.field(dtype=ti.i32, shape=self.n)

        # 심리/행동
        self.price_expectation = ti.field(dtype=ti.f32, shape=self.n)  # 가격 기대 (-1~1)
        self.risk_tolerance = ti.field(dtype=ti.f32, shape=self.n)     # 위험 허용도 (0~1)
        self.fomo_sensitivity = ti.field(dtype=ti.f32, shape=self.n)   # FOMO 민감도 (0~1)
        self.loss_aversion = ti.field(dtype=ti.f32, shape=self.n)      # 손실 회피 계수 (1.5~3.5, 평균 2.5 - Genesove & Mayer 2001)
        self.herding_tendency = ti.field(dtype=ti.f32, shape=self.n)   # 군집 성향 (0~1)
        self.speculation_horizon = ti.field(dtype=ti.i32, shape=self.n)  # 투기자의 목표 보유 기간 (개월)

        # 생애주기 속성
        self.is_married = ti.field(dtype=ti.i32, shape=self.n)        # 결혼 여부
        self.num_children = ti.field(dtype=ti.i32, shape=self.n)      # 자녀 수
        self.eldest_child_age = ti.field(dtype=ti.i32, shape=self.n)  # 장자녀 나이 (없으면 -1)
        self.life_stage = ti.field(dtype=ti.i32, shape=self.n)        # 생애주기 단계 (0:미혼, 1:신혼, 2:육아, 3:학령기, 4:빈둥지, 5:은퇴)

        # 주택 관련 기록
        self.purchase_price = ti.field(dtype=ti.f32, shape=self.n)    # 주 주택 매입가
        self.purchase_month = ti.field(dtype=ti.i32, shape=self.n)    # 매입 시점
        self.total_purchase_price = ti.field(dtype=ti.f32, shape=self.n)  # 전체 주택 매입가 합계

        # 상태
        self.homeless_months = ti.field(dtype=ti.i32, shape=self.n)  # 무주택 기간
        self.wants_to_buy = ti.field(dtype=ti.i32, shape=self.n)     # 매수 희망 (0/1)
        self.wants_to_sell = ti.field(dtype=ti.i32, shape=self.n)    # 매도 희망 (0/1)
        self.target_region = ti.field(dtype=ti.i32, shape=self.n)    # 매수 희망 지역

        # 사회적 영향 (지역 내 거래 동향)
        self.observed_buying = ti.field(dtype=ti.f32, shape=self.n)   # 관측된 매수 비율
        self.observed_price_trend = ti.field(dtype=ti.f32, shape=self.n)  # 관측된 가격 추세

        # 대출
        self.mortgage_balance = ti.field(dtype=ti.f32, shape=self.n)  # 주담대 잔액
        self.jeonse_deposit = ti.field(dtype=ti.f32, shape=self.n)    # 전세 보증금 (세입자인 경우)

        # 부모 지원 (무주택 청년층 구매력 보강)
        self.parent_support = ti.field(dtype=ti.f32, shape=self.n)    # 부모 지원 가능 금액

        # 랜덤 시드
        self.rand_seed = ti.field(dtype=ti.i32, shape=self.n)

        # 지역별 집계 (사회적 영향 계산용)
        self.region_buy_rate = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        self.region_price_trend_6m = ti.field(dtype=ti.f32, shape=NUM_REGIONS)

        # === 가격 적정성 지표 (구조적 개선) ===
        self.region_pir = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # PIR (소득대비가격비)
        self.region_price_to_hist = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # 역사적 평균 대비
        self.region_attractiveness = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # 투자 매력도
        self.region_prestige = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # 심리적 프리미엄
        self.region_job_density = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # 일자리 밀도

        # === 고용 상태 (JobMarket 연동) ===
        self.employment_status = ti.field(dtype=ti.i32, shape=self.n)
        # 0: 취업, 1: 실업, 2: 실업급여 수령중
        self.industry = ti.field(dtype=ti.i32, shape=self.n)
        # 산업 분류 (0: IT/금융, 1: 전문서비스, 2: 제조업, 3: 서비스업, 4: 공공/교육)
        self.unemployment_months = ti.field(dtype=ti.i32, shape=self.n)
        # 연속 실업 개월수
        self.previous_income = ti.field(dtype=ti.f32, shape=self.n)
        # 실업 전 소득 (실업급여 계산용)
        self.forced_sale_countdown = ti.field(dtype=ti.i32, shape=self.n)
        # 강제매도 카운트다운 (-1이면 정상)
        self.housing_cost_unpaid = ti.field(dtype=ti.i32, shape=self.n)
        # 주거비 미납 개월수

        # === 매수/매도 의사결정 중간 결과 (독립 모듈 분리용) ===
        self.buy_score_affordability = ti.field(dtype=ti.f32, shape=self.n)
        self.buy_score_lifecycle = ti.field(dtype=ti.f32, shape=self.n)
        self.buy_score_behavioral = ti.field(dtype=ti.f32, shape=self.n)
        self.buy_score_market = ti.field(dtype=ti.f32, shape=self.n)
        self.buy_score_policy = ti.field(dtype=ti.f32, shape=self.n)
        self.is_affordable_flag = ti.field(dtype=ti.i32, shape=self.n)

        # === 전망이론 파라미터 (Prospect Theory) ===
        # 개인별 손실 회피 계수 (λ) - 기존 loss_aversion 필드 사용
        # 추가: 참조점 (reference point)
        self.reference_price = ti.field(dtype=ti.f32, shape=self.n)  # 심리적 참조 가격

        # === 시간 할인 파라미터 (Hyperbolic Discounting) ===
        self.discount_beta = ti.field(dtype=ti.f32, shape=self.n)  # 현재 편향 (β)
        self.discount_delta = ti.field(dtype=ti.f32, shape=self.n)  # 기하 할인율 (δ)

        # === 네트워크 구조 (Small-World Network) ===
        max_neighbors = config.network.max_neighbors
        self.neighbors = ti.field(dtype=ti.i32, shape=(self.n, max_neighbors))
        self.num_neighbors = ti.field(dtype=ti.i32, shape=self.n)
        self.neighbor_buying_ratio = ti.field(dtype=ti.f32, shape=self.n)  # 이웃 매수 비율
        self.network_belief = ti.field(dtype=ti.f32, shape=self.n)  # 네트워크 기반 신념

    def initialize(self, rng: np.random.Generator):
        """초기 상태 설정 (행동경제학 요소 포함)"""
        # 에이전트 구성 설정 읽기
        ac = self.config.agent_composition

        # 연령 분포 (25-80) - config에서 파라미터 읽기
        # 청년(25-34), 중년(35-54), 장년(55+)
        young_ratio = ac.age_young_ratio
        middle_ratio = ac.age_middle_ratio
        senior_ratio = ac.age_senior_ratio

        # 세부 연령대로 분할: 청년(25-34), 중년전반(35-44), 중년후반(45-54), 장년전반(55-64), 장년후반(65-80)
        age_probs = np.array([
            young_ratio,                    # 25-34
            middle_ratio * 0.5,             # 35-44
            middle_ratio * 0.5,             # 45-54
            senior_ratio * 0.6,             # 55-64
            senior_ratio * 0.4              # 65-80
        ])
        age_bins = [(25, 34), (35, 44), (45, 54), (55, 64), (65, 80)]

        ages = np.zeros(self.n, dtype=np.int32)
        idx = 0
        for prob, (low, high) in zip(age_probs, age_bins):
            count = int(self.n * prob)
            ages[idx:idx+count] = rng.integers(low, high+1, size=count)
            idx += count
        ages[idx:] = rng.integers(25, 80, size=self.n - idx)
        rng.shuffle(ages)

        # 소득 분포 (로그정규) - config에서 파라미터 읽기
        income_median = ac.income_median
        income_sigma = ac.income_sigma
        incomes = rng.lognormal(mean=np.log(income_median), sigma=income_sigma, size=self.n).astype(np.float32)
        incomes = np.clip(incomes, 100, 10000)

        # 자산 분포 (파레토) - config에서 파라미터 읽기
        # 파레토 분포에서 중위값 = x_m * 2^(1/alpha)
        # 따라서 x_m = median / 2^(1/alpha)
        asset_median = ac.asset_median
        asset_alpha = ac.asset_alpha
        x_m = asset_median / (2 ** (1 / asset_alpha))
        assets = (rng.pareto(a=asset_alpha, size=self.n) + 1) * x_m
        assets = assets.astype(np.float32)

        # 지역 배치
        regions = rng.choice(NUM_REGIONS, size=self.n, p=REGION_HOUSEHOLD_RATIO).astype(np.int32)

        # 주택 보유 분포 - config에서 파라미터 읽기
        homeless_rate = ac.initial_homeless_rate
        one_house_threshold = homeless_rate
        multi_house_threshold = homeless_rate + ac.initial_one_house_rate
        ownership_roll = rng.random(self.n)
        owned = np.zeros(self.n, dtype=np.int32)
        owned[ownership_roll >= one_house_threshold] = 1
        owned[ownership_roll >= multi_house_threshold] = rng.integers(2, 6, size=np.sum(ownership_roll >= multi_house_threshold))

        # 자산과 주택 보유 상관관계 조정 (자산 많은 사람이 다주택자일 확률 높음)
        asset_rank = np.argsort(assets)[::-1]
        multi_owner_indices = np.where(owned >= 2)[0]
        top_asset_indices = asset_rank[:len(multi_owner_indices) * 2]
        # 다주택자를 상위 자산가 중에서 선택
        new_multi = rng.choice(top_asset_indices, size=len(multi_owner_indices), replace=False)
        owned_new = np.zeros(self.n, dtype=np.int32)
        owned_new[ownership_roll >= one_house_threshold] = 1
        for i in new_multi:
            owned_new[i] = rng.integers(2, 6)

        # 가격 기대 (-1 ~ 1) - 중립 분포 (편향 제거)
        expectations = rng.normal(0.0, 0.3, size=self.n).astype(np.float32)
        expectations = np.clip(expectations, -1, 1)

        # 위험 허용도 (나이가 많을수록 낮음)
        base_risk = rng.beta(2, 5, size=self.n).astype(np.float32)
        age_factor = np.clip(1.0 - (ages - 25) / 55 * 0.5, 0.5, 1.0)
        risk_tolerance = (base_risk * age_factor).astype(np.float32)

        # === 행동경제학 속성 초기화 ===

        # FOMO 민감도 (젊은 층이 더 민감)
        fomo_base = rng.beta(3, 3, size=self.n).astype(np.float32)
        fomo_age_factor = np.clip(1.0 - (ages - 25) / 40 * 0.4, 0.6, 1.0)
        fomo_sensitivity = (fomo_base * fomo_age_factor).astype(np.float32)

        # 손실 회피 계수 (Loss Aversion Coefficient)
        # 학술적 근거:
        # - Tversky & Kahneman (1992): 기본 전망이론 계수 lambda = 2.25
        # - Genesove & Mayer (2001, QJE): 부동산 시장 실증 연구
        #   - 보스턴 콘도 데이터: 손실 상황 매도자는 호가를 25-35% 높게 책정
        #   - 덴마크 후속 연구: 손실이 이득보다 약 2.5배 더 크게 작용
        # - 범위 1.5-3.5: 개인차 반영 (연구에서 약 30%가 3.0 이상 보고)
        # 참고: docs/references.md
        loss_aversion = rng.normal(2.5, 0.35, size=self.n).astype(np.float32)
        loss_aversion = np.clip(loss_aversion, 1.5, 3.5)

        # 군집 성향 (개인차 존재)
        herding_tendency = rng.beta(2, 3, size=self.n).astype(np.float32)

        # === 에이전트 유형 초기화 (JASSS 2020 한국 ABM 참고) ===
        # 유형 분포 - config에서 읽기
        # - 실수요자 (owner-occupier): 거주 목적 구매
        # - 투자자 (investor): 임대 수익 목적
        # - 투기자 (speculator): 시세차익 목적, 단기 보유
        investor_ratio = ac.investor_ratio
        speculator_ratio = ac.speculator_ratio
        agent_type = np.zeros(self.n, dtype=np.int32)  # 기본: 실수요자

        # 자산 상위 30% 중에서 투자자/투기자 선정
        asset_percentile_70 = np.percentile(assets, 70)
        high_asset_mask = assets >= asset_percentile_70

        # 투자자: 자산 상위 30% 중에서 선정 (전체 대비 investor_ratio 비율)
        investor_candidates = np.where(high_asset_mask)[0]
        n_investors = int(self.n * investor_ratio)
        n_investors = min(n_investors, len(investor_candidates))
        investors = rng.choice(investor_candidates, size=n_investors, replace=False)
        agent_type[investors] = 1

        # 투기자: 자산 상위 30% & 나이 25-50세 중 선정 (전체 대비 speculator_ratio 비율)
        speculator_candidates = np.where(high_asset_mask & (ages >= 25) & (ages <= 50))[0]
        speculator_candidates = np.setdiff1d(speculator_candidates, investors)  # 투자자 제외
        n_speculators = int(self.n * speculator_ratio)
        n_speculators = min(n_speculators, len(speculator_candidates))
        if n_speculators > 0:
            speculators = rng.choice(speculator_candidates, size=n_speculators, replace=False)
            agent_type[speculators] = 2

        # 투기자 특성 조정 - config에서 배율 읽기
        speculator_risk_mult = ac.speculator_risk_multiplier
        speculator_fomo_mult = ac.speculator_fomo_multiplier
        speculator_mask = agent_type == 2
        risk_tolerance[speculator_mask] = np.clip(
            risk_tolerance[speculator_mask] * speculator_risk_mult + 0.2, 0, 1
        )
        fomo_sensitivity[speculator_mask] = np.clip(
            fomo_sensitivity[speculator_mask] * speculator_fomo_mult + 0.2, 0, 1
        )
        loss_aversion[speculator_mask] = np.clip(
            loss_aversion[speculator_mask] * 0.7, 1.5, 2.5  # 낮은 손실 회피
        )
        herding_tendency[speculator_mask] = np.clip(
            herding_tendency[speculator_mask] * 1.4 + 0.1, 0, 1
        )

        # 투기자의 목표 보유 기간 - config에서 읽기
        horizon_min = ac.speculator_horizon_min
        horizon_max = ac.speculator_horizon_max
        speculation_horizon = np.zeros(self.n, dtype=np.int32)
        speculation_horizon[speculator_mask] = rng.integers(horizon_min, horizon_max + 1, size=np.sum(speculator_mask))

        # === 생애주기 속성 초기화 ===

        # 결혼 여부 (나이별 기혼율)
        marriage_prob = np.zeros(self.n, dtype=np.float32)
        marriage_prob[ages < 30] = 0.2
        marriage_prob[(ages >= 30) & (ages < 35)] = 0.5
        marriage_prob[(ages >= 35) & (ages < 45)] = 0.75
        marriage_prob[(ages >= 45) & (ages < 60)] = 0.8
        marriage_prob[ages >= 60] = 0.7  # 사별/이혼 고려
        is_married = (rng.random(self.n) < marriage_prob).astype(np.int32)

        # 자녀 수 (기혼자 중)
        num_children = np.zeros(self.n, dtype=np.int32)
        married_mask = is_married == 1
        num_children[married_mask] = rng.choice(
            [0, 1, 2, 3],
            size=np.sum(married_mask),
            p=[0.15, 0.35, 0.40, 0.10]
        )

        # 장자녀 나이 (부모 나이 - 출산 나이 추정)
        eldest_child_age = np.full(self.n, -1, dtype=np.int32)
        has_children = (num_children > 0) & married_mask
        birth_age = rng.integers(25, 35, size=np.sum(has_children))
        eldest_child_age[has_children] = np.maximum(0, ages[has_children] - birth_age)
        eldest_child_age[has_children] = np.minimum(eldest_child_age[has_children], 30)

        # 생애주기 단계 결정
        life_stage = self._determine_life_stage(ages, is_married, num_children, eldest_child_age)

        # 무주택 기간
        homeless = np.zeros(self.n, dtype=np.int32)
        homeless[owned_new == 0] = rng.integers(0, 120, size=np.sum(owned_new == 0))

        # 매입가 기록 초기화 (보유자의 경우 현재 지역가 기준)
        from .config import REGIONS
        purchase_price = np.zeros(self.n, dtype=np.float32)
        total_purchase_price = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if owned_new[i] > 0:
                region_price = REGIONS[regions[i]]["base_price"]
                # 과거 매입가는 현재가의 80-120% 범위로 설정
                price_factor = rng.uniform(0.8, 1.2)
                purchase_price[i] = region_price * price_factor
                total_purchase_price[i] = purchase_price[i] * owned_new[i]

        purchase_month = np.zeros(self.n, dtype=np.int32)
        purchase_month[owned_new > 0] = rng.integers(-120, 0, size=np.sum(owned_new > 0))

        # Taichi 필드에 복사 (기본)
        self.age.from_numpy(ages)
        self.income.from_numpy(incomes)
        self.asset.from_numpy(assets)
        self.region.from_numpy(regions)
        self.owned_houses.from_numpy(owned_new)
        self.primary_house_id.from_numpy(np.full(self.n, -1, dtype=np.int32))
        self.price_expectation.from_numpy(expectations)
        self.risk_tolerance.from_numpy(risk_tolerance)
        self.homeless_months.from_numpy(homeless)
        self.wants_to_buy.from_numpy(np.zeros(self.n, dtype=np.int32))
        self.wants_to_sell.from_numpy(np.zeros(self.n, dtype=np.int32))
        self.target_region.from_numpy(regions)  # 초기에는 현재 지역
        self.mortgage_balance.from_numpy(np.zeros(self.n, dtype=np.float32))
        self.jeonse_deposit.from_numpy(np.zeros(self.n, dtype=np.float32))
        self.rand_seed.from_numpy(rng.integers(0, 2**30, size=self.n, dtype=np.int32))

        # Taichi 필드에 복사 (행동경제학)
        self.fomo_sensitivity.from_numpy(fomo_sensitivity)
        self.loss_aversion.from_numpy(loss_aversion)
        self.herding_tendency.from_numpy(herding_tendency)

        # Taichi 필드에 복사 (에이전트 유형)
        self.agent_type.from_numpy(agent_type)
        self.speculation_horizon.from_numpy(speculation_horizon)

        # Taichi 필드에 복사 (생애주기)
        self.is_married.from_numpy(is_married)
        self.num_children.from_numpy(num_children)
        self.eldest_child_age.from_numpy(eldest_child_age)
        self.life_stage.from_numpy(life_stage)

        # Taichi 필드에 복사 (주택 기록)
        self.purchase_price.from_numpy(purchase_price)
        self.purchase_month.from_numpy(purchase_month)
        self.total_purchase_price.from_numpy(total_purchase_price)

        # 사회적 영향 초기화
        self.observed_buying.from_numpy(np.zeros(self.n, dtype=np.float32))
        self.observed_price_trend.from_numpy(np.zeros(self.n, dtype=np.float32))
        self.region_buy_rate.from_numpy(np.zeros(NUM_REGIONS, dtype=np.float32))
        self.region_price_trend_6m.from_numpy(np.zeros(NUM_REGIONS, dtype=np.float32))

        # === 전망이론 파라미터 초기화 ===
        # 참조 가격 (매입가 또는 현재가)
        reference_price = purchase_price.copy()
        reference_price[owned_new == 0] = 0.0  # 무주택자는 참조점 없음
        self.reference_price.from_numpy(reference_price)

        # === 시간 할인 파라미터 초기화 (Laibson, 1997) ===
        # β-δ 모델: 개인별 이질성 반영
        discount_cfg = self.config.discounting
        discount_beta = rng.normal(
            discount_cfg.beta_mean,
            discount_cfg.beta_std,
            size=self.n
        ).astype(np.float32)
        discount_beta = np.clip(discount_beta, 0.5, 1.0)

        discount_delta = rng.normal(
            discount_cfg.delta_mean,
            discount_cfg.delta_std,
            size=self.n
        ).astype(np.float32)
        discount_delta = np.clip(discount_delta, 0.95, 1.0)

        # 투기자는 더 높은 현재 편향 (낮은 β)
        discount_beta[speculator_mask] = np.clip(
            discount_beta[speculator_mask] * 0.85, 0.5, 0.8
        )

        self.discount_beta.from_numpy(discount_beta)
        self.discount_delta.from_numpy(discount_delta)

        # === 부모 지원 초기화 (현실 반영) ===
        # 한국에서 30대 주택 구입의 약 40%가 부모 지원을 받음 (KB부동산 조사)
        # 무주택 청년층에게 평균 1.5억원 수준의 지원
        aff_cfg = self.config.policy.affordability
        parent_support = np.zeros(self.n, dtype=np.float32)

        # 지원 대상: 무주택자 & 40세 이하
        eligible_mask = (owned_new == 0) & (ages <= aff_cfg.parent_support_age_max)
        n_eligible = np.sum(eligible_mask)

        if n_eligible > 0:
            # 지원 받는 사람 선정 (확률적)
            receives_support = rng.random(n_eligible) < aff_cfg.parent_support_rate

            # 지원액 결정 (정규분포)
            support_amounts = rng.normal(
                aff_cfg.parent_support_mean,
                aff_cfg.parent_support_std,
                size=n_eligible
            ).astype(np.float32)
            support_amounts = np.maximum(support_amounts, 0)  # 음수 방지
            support_amounts[~receives_support] = 0  # 지원 안 받는 사람은 0

            parent_support[eligible_mask] = support_amounts

        self.parent_support.from_numpy(parent_support)

        # === 네트워크 초기화 (Small-World, Watts & Strogatz, 1998) ===
        self._initialize_network(rng)

    def _determine_life_stage(self, ages, is_married, num_children, eldest_child_age):
        """생애주기 단계 결정

        0: 미혼 (single)
        1: 신혼 (newly_married, 결혼 후 자녀 없음 or 영아)
        2: 육아기 (parenting, 자녀 0-6세)
        3: 학령기 (school_age, 자녀 7-18세, 학군 중요)
        4: 빈둥지 (empty_nest, 자녀 독립)
        5: 은퇴기 (retired, 55세 이상)
        """
        n = len(ages)
        life_stage = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if is_married[i] == 0:
                life_stage[i] = 0  # 미혼
            elif ages[i] >= 60:
                life_stage[i] = 5  # 은퇴기
            elif num_children[i] == 0:
                life_stage[i] = 1  # 신혼
            elif eldest_child_age[i] <= 6:
                life_stage[i] = 2  # 육아기
            elif eldest_child_age[i] <= 18:
                life_stage[i] = 3  # 학령기
            elif eldest_child_age[i] > 18:
                life_stage[i] = 4  # 빈둥지
            else:
                life_stage[i] = 1  # 기본값

        return life_stage

    def _initialize_network(self, rng: np.random.Generator):
        """Small-World 네트워크 초기화 (Watts & Strogatz, 1998)

        지역 기반 초기 연결 후 일부 재연결하여 Small-World 속성 부여
        """
        net_cfg = self.config.network
        max_neighbors = net_cfg.max_neighbors
        avg_neighbors = net_cfg.avg_neighbors
        rewiring_prob = net_cfg.rewiring_prob

        # 지역별 에이전트 인덱스
        regions_np = self.region.to_numpy()
        region_indices = [np.where(regions_np == r)[0] for r in range(NUM_REGIONS)]

        neighbors = np.full((self.n, max_neighbors), -1, dtype=np.int32)
        num_neighbors = np.zeros(self.n, dtype=np.int32)

        for i in range(self.n):
            my_region = regions_np[i]
            same_region = region_indices[my_region]

            # 같은 지역 내에서 이웃 선택 (자신 제외)
            candidates = same_region[same_region != i]
            n_local = min(avg_neighbors, len(candidates))

            if n_local > 0:
                local_neighbors = rng.choice(candidates, size=n_local, replace=False)

                # 일부를 다른 지역으로 재연결 (long-range connections)
                for j, neighbor in enumerate(local_neighbors):
                    if rng.random() < rewiring_prob:
                        # 다른 지역에서 랜덤 선택
                        other_region = rng.integers(0, NUM_REGIONS)
                        if len(region_indices[other_region]) > 0:
                            local_neighbors[j] = rng.choice(region_indices[other_region])

                neighbors[i, :n_local] = local_neighbors
                num_neighbors[i] = n_local

        self.neighbors.from_numpy(neighbors)
        self.num_neighbors.from_numpy(num_neighbors)
        self.neighbor_buying_ratio.from_numpy(np.zeros(self.n, dtype=np.float32))
        self.network_belief.from_numpy(np.zeros(self.n, dtype=np.float32))

    def update_network_beliefs(self):
        """DeGroot Learning 기반 신념 업데이트 (DeGroot, 1974)

        이웃의 매수 행동과 기대를 관찰하여 자신의 신념 업데이트
        """
        net_cfg = self.config.network
        self_weight = net_cfg.self_weight
        cascade_threshold = net_cfg.cascade_threshold

        # NumPy에서 계산 후 Taichi 필드에 복사
        wants_to_buy_np = self.wants_to_buy.to_numpy()
        expectations_np = self.price_expectation.to_numpy()
        neighbors_np = self.neighbors.to_numpy()
        num_neighbors_np = self.num_neighbors.to_numpy()

        neighbor_buying = np.zeros(self.n, dtype=np.float32)
        neighbor_expectation = np.zeros(self.n, dtype=np.float32)

        for i in range(self.n):
            n_neighbors = num_neighbors_np[i]
            if n_neighbors > 0:
                neighbor_ids = neighbors_np[i, :n_neighbors]
                valid_neighbors = neighbor_ids[neighbor_ids >= 0]

                if len(valid_neighbors) > 0:
                    # 이웃 매수 비율
                    neighbor_buying[i] = wants_to_buy_np[valid_neighbors].mean()
                    # 이웃 기대 평균
                    neighbor_expectation[i] = expectations_np[valid_neighbors].mean()

        # DeGroot 신념 업데이트
        # new_belief = self_weight * own_expectation + (1 - self_weight) * neighbor_avg
        network_belief = (
            self_weight * expectations_np +
            (1 - self_weight) * neighbor_expectation
        ).astype(np.float32)

        self.neighbor_buying_ratio.from_numpy(neighbor_buying)
        self.network_belief.from_numpy(network_belief)

    @ti.kernel
    def apply_information_cascade(self, cascade_threshold: ti.f32, cascade_multiplier: ti.f32):
        """정보 캐스케이드 적용 (이웃 매수 비율이 임계값 초과 시)

        수정 (2024): 다주택자(2주택 이상)는 캐스케이드에서 제외
        - 정책 규제(취득세 8-12%)가 심리적 효과보다 우선
        - 다주택자는 이웃이 매수해도 정책 비용으로 인해 따라 매수하지 않음
        """
        for i in range(self.n):
            owned = self.owned_houses[i]

            # 다주택자(2주택 이상)는 캐스케이드에서 제외
            # 8-12% 취득세 부담으로 심리적 효과가 무력화됨
            if owned >= 2:
                continue

            if self.neighbor_buying_ratio[i] > cascade_threshold:
                # 캐스케이드: 매수 확률 증가
                if self.wants_to_buy[i] == 0:
                    # 랜덤하게 매수 전환
                    seed = self.rand_seed[i]
                    roll = ti.cast(seed % 1000, ti.f32) / 1000.0
                    self.rand_seed[i] = (seed * 1103515245 + 12345) % 2147483647

                    # 이웃 매수 비율에 비례한 전환 확률
                    cascade_prob = (self.neighbor_buying_ratio[i] - cascade_threshold) * cascade_multiplier
                    if roll < cascade_prob:
                        self.wants_to_buy[i] = 1

    def update_social_signals(self, market, recent_transactions: np.ndarray):
        """사회적 신호 업데이트 (지역별 거래 동향, 6개월 가격 추세)

        Args:
            market: Market 인스턴스
            recent_transactions: 최근 월별 거래량 배열
        """
        # 지역별 6개월 가격 추세 계산
        if len(market.price_history) >= 6:
            prices_6m_ago = market.price_history[-6]
            prices_now = market.region_prices.to_numpy()
            price_trend = (prices_now - prices_6m_ago) / (prices_6m_ago + 1e-6)
            price_trend = np.clip(price_trend, -0.5, 0.5).astype(np.float32)
        else:
            price_trend = np.zeros(NUM_REGIONS, dtype=np.float32)

        # 지역별 매수 비율 (최근 거래량 / 가구수)
        regions_np = self.region.to_numpy()
        region_counts = np.bincount(regions_np, minlength=NUM_REGIONS).astype(np.float32)
        region_counts = np.maximum(region_counts, 1.0)

        demand_np = market.demand.to_numpy().astype(np.float32)
        buy_rate = demand_np / region_counts
        buy_rate = np.clip(buy_rate, 0, 0.2).astype(np.float32)

        self.region_price_trend_6m.from_numpy(price_trend)
        self.region_buy_rate.from_numpy(buy_rate)

    def _calculate_dsr_numpy(self, price: float, asset: float, annual_income: float,
                              interest_rate: float, asset_utilization: float,
                              loan_term_years: int = 30) -> float:
        """DSR 계산 (NumPy 버전, select_target_regions에서 사용)"""
        available_asset = asset * asset_utilization
        required_loan = max(price - available_asset, 0.0)

        if required_loan <= 0 or annual_income <= 0:
            return 0.0

        monthly_rate = interest_rate / 12.0
        n_payments = loan_term_years * 12

        if monthly_rate > 0:
            factor = (1.0 + monthly_rate) ** n_payments
            monthly_payment = required_loan * monthly_rate * factor / (factor - 1.0)
        else:
            monthly_payment = required_loan / n_payments

        annual_payment = monthly_payment * 12.0
        return annual_payment / annual_income

    def _check_dsr_affordable_numpy(self, dsr: float, agent_type: int,
                                    is_wealthy: bool, job_density: float = 0.0) -> bool:
        """DSR 기반 구매 가능 여부 (NumPy 버전)

        [수정] 영끌(DSR 한도 확대) 로직 제거
        소득이 지역×산업 기반이므로 고소득 지역은 자연스럽게 DSR이 낮아짐
        """
        aff_cfg = self.config.policy.affordability

        # 에이전트 유형별 DSR 한도
        if agent_type == 0:  # 실수요자
            dsr_limit = aff_cfg.dsr_limit_end_user
        elif agent_type == 1:  # 투자자
            dsr_limit = aff_cfg.dsr_limit_investor
        else:  # 투기자
            dsr_limit = aff_cfg.dsr_limit_speculator

        return dsr <= dsr_limit

    def select_target_regions(self, market, rng: np.random.Generator, job_density=None):
        """에이전트 유형별 목표 지역 선택 (DSR 기반 통일 체계)

        실수요자: 거주 지역 위주, 인근 지역도 고려 (DSR 기반)
        투자자: 투자매력도 높은 지역 탐색 (기대수익률 기반)
        투기자: 가격 상승률 높은 지역 탐색 (모멘텀 기반)

        Args:
            job_density: 동적 일자리 밀도 (None이면 정적 REGION_JOB_DENSITY 사용)
        """
        agent_types = self.agent_type.to_numpy()
        home_regions = self.region.to_numpy()
        assets = self.asset.to_numpy()
        incomes = self.income.to_numpy()
        owned_houses = self.owned_houses.to_numpy()  # 주택 보유 수
        parent_supports = self.parent_support.to_numpy()  # 부모 지원금

        # 시장 지표
        prices = market.region_prices.to_numpy()
        attractiveness = market.investment_attractiveness.to_numpy()
        price_trends = self.region_price_trend_6m.to_numpy()
        pir = market.region_pir.to_numpy()
        price_to_hist = market.price_to_historical.to_numpy()

        # 적정성 지표를 필드에 복사 (decide_buy_sell에서 사용)
        self.region_pir.from_numpy(pir.astype(np.float32))
        self.region_price_to_hist.from_numpy(price_to_hist.astype(np.float32))
        self.region_attractiveness.from_numpy(attractiveness.astype(np.float32))
        # 동적 프리미엄 사용 (market에서 매 스텝 업데이트)
        dynamic_prestige = getattr(market, 'dynamic_prestige', REGION_PRESTIGE)
        self.region_prestige.from_numpy(dynamic_prestige.astype(np.float32))
        # 동적 일자리 밀도 사용 (JobMarket 제공, 없으면 정적 값)
        effective_job_density = job_density if job_density is not None else REGION_JOB_DENSITY
        self.region_job_density.from_numpy(effective_job_density.astype(np.float32))

        target_regions = np.copy(home_regions)  # 기본값: 거주 지역

        # 자산 상위 10% 기준값 (고자산가 판별용)
        # 데이터: 상위 10% 순자산 10.5억+ (전체 자산의 44.4% 점유)
        asset_threshold_80 = np.percentile(assets, 90)

        # DSR 계산을 위한 설정값
        aff_cfg = self.config.policy.affordability
        interest_rate = self.config.policy.interest_rate + self.config.policy.mortgage_spread
        loan_term = aff_cfg.loan_term_years

        for i in range(self.n):
            agent_type = agent_types[i]
            home = home_regions[i]
            asset = assets[i]
            income = incomes[i]
            annual_income = income * 12
            owned = owned_houses[i]  # 주택 보유 수

            # 고자산가 여부 (상위 20%)
            is_wealthy = asset >= asset_threshold_80

            # ================================================================
            # 자산 활용 비율 (2026-02-06 수정: decide_buy_sell과 일관성)
            # ================================================================
            # 무주택자: 첫 집 마련에 85% 동원 + 부모 지원
            # 유주택자: 50% (고자산가 70%)
            # ================================================================
            parent_support = parent_supports[i] if owned == 0 else 0.0

            if owned == 0:
                asset_util = aff_cfg.homeless_asset_utilization  # 무주택자: 85%
                # 부모 지원 포함 (100% 사용 가능)
                effective_asset = asset + parent_support / asset_util
            elif is_wealthy:
                asset_util = aff_cfg.wealthy_asset_utilization  # 고자산가: 70%
                effective_asset = asset
            else:
                asset_util = aff_cfg.normal_asset_utilization  # 일반: 50%
                effective_asset = asset

            if agent_type == 0:  # 실수요자
                # 후보 지역 선정: 인접 지역 + 고자산가만 원거리 이동 허용
                candidates = self._get_adjacent_regions(home)
                candidates.append(home)

                # 고자산가만 원거리(수도권↔지방) 이동 허용
                if is_wealthy:
                    candidates = list(set(candidates + [0, 1, 3]))
                    # 고소득 일자리 지역도 추가
                    high_job_regions = [r for r in range(NUM_REGIONS) if effective_job_density[r] >= 0.5]
                    candidates = list(set(candidates + high_job_regions))
                else:
                    # 일반 가구: 인접 + 가까운 고용 밀집 지역만
                    for r in range(NUM_REGIONS):
                        if effective_job_density[r] >= 0.5 and ADJACENCY[home][r] >= 0.3:
                            candidates.append(r)
                    candidates = list(set(candidates))

                best_region = home
                best_score = -999.0

                for r in candidates:
                    if prices[r] <= 0:
                        continue

                    # DSR 계산 (통일된 체계, 부모 지원 포함)
                    dsr = self._calculate_dsr_numpy(
                        prices[r], effective_asset, annual_income,
                        interest_rate, asset_util, loan_term
                    )

                    # DSR 기반 구매 가능 여부 판단
                    jd = effective_job_density[r]
                    is_affordable = self._check_dsr_affordable_numpy(
                        dsr, agent_type, is_wealthy, jd
                    )

                    if not is_affordable:
                        continue  # DSR 초과 → 구매 불가

                    # 점수 계산 (가중치 재조정: prestige 비중 확대)
                    dsr_score = max(0, 1 - dsr) * 0.25  # 25% (축소)

                    # 일자리 밀도 점수 (축소)
                    job_score = effective_job_density[r] * 0.35  # 35% (축소)

                    # PIR 기반 점수 (축소)
                    pir_score = max(0, 1 - pir[r] / 30) * 0.10  # 10% (축소)

                    # 동적 프리미엄 보너스 (대폭 확대: 15-20%)
                    prestige = dynamic_prestige[r]
                    prestige_bonus = prestige * (0.20 if is_wealthy else 0.15)

                    # 이동 마찰 보너스 (가까울수록 유리, 10%)
                    adjacency_score = ADJACENCY[home][r]
                    mobility_bonus = adjacency_score * 0.10

                    score = dsr_score + job_score + pir_score + prestige_bonus + mobility_bonus

                    if score > best_score:
                        best_score = score
                        best_region = r

                target_regions[i] = best_region

            elif agent_type == 1:  # 투자자
                # DSR 기반 구매 가능 지역 마스크
                affordable_mask = np.zeros(NUM_REGIONS, dtype=bool)
                for r in range(NUM_REGIONS):
                    if prices[r] <= 0:
                        continue
                    dsr = self._calculate_dsr_numpy(
                        prices[r], effective_asset, annual_income,
                        interest_rate, asset_util, loan_term
                    )
                    affordable_mask[r] = self._check_dsr_affordable_numpy(
                        dsr, agent_type, is_wealthy, effective_job_density[r]
                    )

                if np.sum(affordable_mask) > 0:
                    # 투자매력도 60% + 동적 프리미엄 40%
                    combined_score = np.where(
                        affordable_mask,
                        attractiveness * 0.6 + dynamic_prestige * 0.4,
                        -999
                    )
                    top_regions = np.argsort(combined_score)[-5:][::-1]
                    valid_top = [r for r in top_regions if affordable_mask[r]]

                    if valid_top:
                        probs = np.array([max(0, combined_score[r] + 0.3) for r in valid_top])
                        if np.sum(probs) > 0:
                            probs = probs / np.sum(probs)
                            target_regions[i] = rng.choice(valid_top, p=probs)

            elif agent_type == 2:  # 투기자
                # DSR 기반 구매 가능 지역 마스크
                affordable_mask = np.zeros(NUM_REGIONS, dtype=bool)
                for r in range(NUM_REGIONS):
                    if prices[r] <= 0:
                        continue
                    dsr = self._calculate_dsr_numpy(
                        prices[r], effective_asset, annual_income,
                        interest_rate, asset_util, loan_term
                    )
                    affordable_mask[r] = self._check_dsr_affordable_numpy(
                        dsr, agent_type, is_wealthy, effective_job_density[r]
                    )

                if np.sum(affordable_mask) > 0:
                    # 모멘텀 70% + 동적 프리미엄 30%
                    combined_score = np.where(
                        affordable_mask,
                        price_trends * 0.7 + dynamic_prestige * 0.3,
                        -999
                    )
                    top_regions = np.argsort(combined_score)[-5:][::-1]
                    valid_top = [r for r in top_regions if affordable_mask[r]]

                    if valid_top:
                        probs = np.array([max(0, combined_score[r] + 0.15) for r in valid_top])
                        if np.sum(probs) > 0:
                            probs = probs / np.sum(probs)
                            target_regions[i] = rng.choice(valid_top, p=probs)

        self.target_region.from_numpy(target_regions.astype(np.int32))

    def _get_adjacent_regions(self, region: int) -> list:
        """인접 지역 반환"""
        # 지역 인접 관계 정의
        adjacency_map = {
            0: [1, 2, 3],      # 강남 → 마용성, 기타서울, 분당
            1: [0, 2],         # 마용성 → 강남, 기타서울
            2: [0, 1, 4, 5, 6],  # 기타서울 → 강남, 마용성, 경기남부, 경기북부, 인천
            3: [0, 4],         # 분당 → 강남, 경기남부
            4: [2, 3, 5, 6],   # 경기남부 → 기타서울, 분당, 경기북부, 인천
            5: [2, 4, 6],      # 경기북부 → 기타서울, 경기남부, 인천
            6: [2, 4, 5],      # 인천 → 기타서울, 경기남부, 경기북부
            7: [8, 12],        # 부산 → 대구, 기타지방
            8: [7, 10, 12],    # 대구 → 부산, 대전, 기타지방
            9: [12],           # 광주 → 기타지방
            10: [8, 11, 12],   # 대전 → 대구, 세종, 기타지방
            11: [10, 12],      # 세종 → 대전, 기타지방
            12: [7, 8, 9, 10, 11],  # 기타지방 → 광역시들
        }
        return adjacency_map.get(region, [])

    @ti.kernel
    def update_expectations(self, price_changes: ti.template(), social_weight: ti.f32):
        """가격 기대 업데이트 (적응적 기대 + 사회적 학습 + FOMO)

        수정 (2024): 기대 증폭 계수를 10.0에서 4.0으로 낮춤
        - 기존 *10.0은 월간 가격변화율(0.01 수준)을 기대(-1~1)로 변환하기 위한 것이었으나 과도했음
        - *4.0으로 변경하여 더 점진적인 기대 형성
        """
        for i in range(self.n):
            region = self.region[i]
            observed_change = price_changes[region]

            # 적응적 기대: 과거 변화를 반영 (비대칭: 상승은 빠르게, 하락은 느리게)
            adaptation_rate = 0.15  # 기본값 (하락장) - 축소
            if observed_change > 0:
                # 상승장: 빠르게 기대 조정 (FOMO 효과)
                adaptation_rate = 0.3 + self.fomo_sensitivity[i] * 0.15  # 축소

            # 기대 증폭 계수 축소: 10.0 → 4.0
            adaptive = self.price_expectation[i] * (1.0 - adaptation_rate) + observed_change * 4.0 * adaptation_rate

            # 군집 효과: 지역 내 다른 사람들의 매수 행동 참조
            region_buy_rate = self.region_buy_rate[region]
            herding_effect = self.herding_tendency[i] * (region_buy_rate - 0.05) * 1.5  # 축소

            # 사회적 학습: 랜덤 노이즈 (개인차)
            seed = self.rand_seed[i]
            noise = ti.cast((seed % 1000) - 500, ti.f32) / 5000.0
            self.rand_seed[i] = (seed * 1103515245 + 12345) % 2147483647

            new_expectation = adaptive + herding_effect + noise * social_weight
            self.price_expectation[i] = ti.math.clamp(new_expectation, -1.0, 1.0)

            # 관측 데이터 업데이트
            self.observed_buying[i] = region_buy_rate
            self.observed_price_trend[i] = self.region_price_trend_6m[region]

    # ================================================================
    # 매수/매도 의사결정 - 6개 독립 커널으로 분리
    # ================================================================
    # 기존 decide_buy_sell (600줄 단일 커널) →
    # compute_affordability → compute_lifecycle_urgency →
    # compute_behavioral_signals → compute_market_signals →
    # compute_policy_penalty → finalize_buy_sell_decision
    # ================================================================

    @ti.kernel
    def compute_affordability(
        self,
        region_prices: ti.template(),
        ltv_limits: ti.template(),
        dti_limit: ti.f32,
        interest_rate: ti.f32,
        dsr_limit_end_user: ti.f32,
        dsr_limit_investor: ti.f32,
        dsr_limit_speculator: ti.f32,
        asset_util_normal: ti.f32,
        asset_util_wealthy: ti.f32,
        asset_util_homeless: ti.f32,
        loan_term_years: ti.i32,
        wealthy_threshold: ti.f32
    ):
        """[환경 모듈] DSR/LTV 기반 구매력 계산

        결과: is_affordable_flag, buy_score_affordability
        실업자(employment_status != 0)는 매수 불가
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            income = self.income[i]
            asset = self.asset[i]
            target = self.target_region[i]
            agent_type = self.agent_type[i]

            price = region_prices[target]

            # 실업자는 매수 불가
            if self.employment_status[i] != 0:
                self.is_affordable_flag[i] = 0
                self.buy_score_affordability[i] = 0.0
                continue

            # 고자산가 여부
            is_wealthy = 1 if asset >= wealthy_threshold else 0

            # 자산 활용 비율
            asset_util = asset_util_normal
            if owned == 0:
                asset_util = asset_util_homeless
            elif is_wealthy == 1:
                asset_util = asset_util_wealthy

            # 부모 지원금 (무주택자만)
            parent_support_amount = self.parent_support[i] if owned == 0 else 0.0

            # DSR 계산
            effective_asset = asset + parent_support_amount / asset_util if asset_util > 0 else asset
            annual_income = income * 12.0
            dsr = calculate_dsr(price, effective_asset, annual_income, interest_rate, asset_util, loan_term_years)

            # 에이전트 유형별 DSR 한도
            dsr_limit = dsr_limit_end_user
            if agent_type == 1:
                dsr_limit = dsr_limit_investor
            elif agent_type == 2:
                dsr_limit = dsr_limit_speculator

            # DSR 기반 구매 가능 여부
            is_affordable = 1 if dsr <= dsr_limit else 0

            # LTV 체크
            ltv = ltv_limits[ti.min(owned, 3)]
            parent_support_val = self.parent_support[i] if owned == 0 else 0.0
            available_asset = asset * asset_util + parent_support_val
            required_loan = ti.max(price - available_asset, 0.0)
            max_loan_by_ltv = price * ltv

            if required_loan > max_loan_by_ltv:
                is_affordable = 0

            self.is_affordable_flag[i] = is_affordable

            # DSR 여유 보너스
            dsr_bonus = 0.0
            if is_affordable == 1:
                dsr_bonus = ti.max(0.0, (dsr_limit - dsr) * 0.3)
                dsr_bonus = ti.min(dsr_bonus, 0.15)
            self.buy_score_affordability[i] = dsr_bonus

    @ti.kernel
    def compute_lifecycle_urgency(self):
        """[에이전트 모듈] 생애주기 기반 주거 긴급도

        결과: buy_score_lifecycle
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            age = self.age[i]
            homeless = self.homeless_months[i]
            life_stage = self.life_stage[i]
            eldest_child = self.eldest_child_age[i]

            urgency = 0.0
            life_stage_bonus = 0.0

            if owned == 0:  # 무주택자
                urgency = 0.15

                if life_stage == 0:  # 미혼
                    if 28 <= age <= 35:
                        urgency += 0.10
                elif life_stage == 1:  # 신혼
                    urgency += 0.20
                    life_stage_bonus = 0.10
                elif life_stage == 2:  # 육아기
                    urgency += 0.15
                    life_stage_bonus = 0.08
                elif life_stage == 3:  # 학령기
                    urgency += 0.12
                    if 10 <= eldest_child <= 15:
                        urgency += 0.08
                elif life_stage == 5:  # 은퇴기
                    urgency += 0.05

                if homeless > 24:
                    urgency += ti.min(homeless / 300.0, 0.15)

            elif owned == 1:  # 1주택자 갈아타기
                urgency = 0.10
                if life_stage == 2:
                    urgency += 0.12
                    life_stage_bonus = 0.05
                elif life_stage == 3:
                    urgency += 0.15
                    if 10 <= eldest_child <= 15:
                        urgency += 0.08
                    life_stage_bonus = 0.05
                elif life_stage == 5:
                    urgency += 0.10
                    life_stage_bonus = 0.03

            self.buy_score_lifecycle[i] = urgency + life_stage_bonus

    @ti.kernel
    def compute_behavioral_signals(
        self,
        pt_alpha: ti.f32,
        pt_beta: ti.f32,
        pt_gamma_gain: ti.f32,
    ):
        """[에이전트 모듈] FOMO/군집행동/전망이론/기대수익

        결과: buy_score_behavioral
        """
        for i in range(self.n):
            target = self.target_region[i]
            region = self.region[i]
            owned = self.owned_houses[i]
            expectation = self.price_expectation[i]
            risk = self.risk_tolerance[i]
            fomo_sens = self.fomo_sensitivity[i]
            herding = self.herding_tendency[i]
            loss_aversion_coef = self.loss_aversion[i]
            beta = self.discount_beta[i]
            delta = self.discount_delta[i]

            price_trend = self.region_price_trend_6m[target]

            # FOMO
            fomo_bonus = 0.0
            if price_trend > 0.05:
                excess_rise = price_trend - 0.05
                fomo_bonus = fomo_sens * excess_rise * 3.0
                fomo_bonus = ti.min(fomo_bonus, 0.3)
            elif price_trend > 0.02:
                fomo_bonus = fomo_sens * (price_trend - 0.02) * 1.5
                fomo_bonus = ti.min(fomo_bonus, 0.15)

            # 군집 행동
            herding_bonus = 0.0
            region_buying = self.region_buy_rate[region]
            if region_buying > 0.03:
                herding_bonus = herding * (region_buying - 0.03) * 3.0
                herding_bonus = ti.min(herding_bonus, 0.2)

            # 기대 수익 (쌍곡선 할인)
            expected_return_bonus = 0.0
            if owned >= 1:
                monthly_appreciation = expectation * 0.01
                rental_yield = 0.003
                horizon = 60
                discounted_return = expected_investment_return(
                    monthly_appreciation, rental_yield, horizon, beta, delta
                )
                expected_return_bonus = discounted_return * risk * 0.05

            # Prospect Theory 이득 기대
            pt_bonus = 0.0
            pt_value = prospect_value(price_trend, pt_alpha, pt_beta, loss_aversion_coef)
            rise_prob = 0.5 + expectation * 0.3
            rise_prob = ti.math.clamp(rise_prob, 0.1, 0.9)
            weighted_prob = probability_weight(rise_prob, pt_gamma_gain)
            if pt_value > 0:
                pt_bonus = weighted_prob * pt_value * 0.05
                pt_bonus = ti.min(pt_bonus, 0.15)

            self.buy_score_behavioral[i] = fomo_bonus + herding_bonus + expected_return_bonus + pt_bonus

    @ti.kernel
    def compute_market_signals(self, wealthy_threshold: ti.f32):
        """[일자리/환경 모듈] 일자리 밀도 + 프리미엄 + 가격 적정성

        결과: buy_score_market
        region_job_density는 JobMarket에서 동적으로 갱신됨
        """
        for i in range(self.n):
            target = self.target_region[i]
            agent_type = self.agent_type[i]
            asset = self.asset[i]

            pir = self.region_pir[target]
            price_to_hist = self.region_price_to_hist[target]
            prestige = self.region_prestige[target]
            job_density = self.region_job_density[target]  # 동적 값

            is_wealthy = 1 if asset >= wealthy_threshold else 0

            # 가격 적정성 보너스/페널티
            valuation_bonus = 0.0
            if pir < 10.0:
                valuation_bonus += (10.0 - pir) * 0.01
            elif pir > 15.0:
                pir_penalty = (pir - 15.0) * 0.02
                if is_wealthy == 1:
                    pir_penalty *= 0.5
                valuation_bonus -= pir_penalty

            if price_to_hist < 1.0:
                valuation_bonus += (1.0 - price_to_hist) * 0.1
            elif price_to_hist > 1.2:
                hist_penalty = (price_to_hist - 1.2) * 0.15
                if is_wealthy == 1:
                    hist_penalty *= 0.5
                valuation_bonus -= hist_penalty
            valuation_bonus = ti.math.clamp(valuation_bonus, -0.15, 0.15)

            # 일자리 밀도 보너스 (동적)
            job_bonus = job_density * 0.22
            if agent_type == 0:
                job_bonus *= 1.3

            # 심리적 프리미엄
            prestige_bonus = prestige * 0.04
            if agent_type == 1:
                prestige_bonus = prestige * 0.06
            elif agent_type == 2:
                prestige_bonus = prestige * 0.08
            if is_wealthy == 1:
                prestige_bonus *= 1.3

            self.buy_score_market[i] = valuation_bonus + job_bonus + prestige_bonus

    @ti.kernel
    def compute_policy_penalty(self):
        """[환경 모듈] 다주택 규제 정책 페널티

        결과: buy_score_policy
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            agent_type = self.agent_type[i]
            policy_penalty = 0.0

            if owned >= 2:
                policy_penalty = 1.5
            elif owned == 1:
                if self.wants_to_sell[i] == 0:
                    policy_penalty = 0.6
                    if agent_type == 1:
                        policy_penalty = 0.4
                    elif agent_type == 2:
                        policy_penalty = 0.35
                else:
                    policy_penalty = 0.05

            self.buy_score_policy[i] = policy_penalty

    @ti.kernel
    def finalize_buy_sell_decision(
        self,
        region_prices: ti.template(),
        buy_threshold: ti.f32,
        sell_threshold: ti.f32,
        transfer_tax_multi: ti.f32,
        jongbu_rate: ti.f32,
        jongbu_threshold: ti.f32,
        pt_alpha: ti.f32,
        pt_beta: ti.f32,
        pt_gamma_gain: ti.f32,
    ):
        """최종 매수/매도 결정

        모든 중간 점수를 합산하고, 다주택 투자 수익성 + 매도 결정
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            income = self.income[i]
            asset = self.asset[i]
            target = self.target_region[i]
            agent_type = self.agent_type[i]
            expectation = self.price_expectation[i]
            life_stage = self.life_stage[i]
            age = self.age[i]
            eldest_child = self.eldest_child_age[i]
            loss_aversion_coef = self.loss_aversion[i]
            purchase_price_val = self.purchase_price[i]

            price = region_prices[target]
            price_trend = self.region_price_trend_6m[target]

            # === 매수 점수 합산 ===
            buy_score = 0.0
            if self.is_affordable_flag[i] == 1:
                buy_score = (self.buy_score_affordability[i] +
                            self.buy_score_lifecycle[i] +
                            self.buy_score_behavioral[i] +
                            self.buy_score_market[i] -
                            self.buy_score_policy[i])

            # 노이즈
            seed = self.rand_seed[i]
            noise = ti.cast((seed % 1000) - 500, ti.f32) / 10000.0
            self.rand_seed[i] = (seed * 1103515245 + 12345) % 2147483647

            # === 다주택 투자 수익성 계산 ===
            monthly_appreciation = expectation * 0.01 + 0.002
            monthly_appreciation = ti.math.clamp(monthly_appreciation, -0.01, 0.03)

            holding_period = 60
            if agent_type == 2:
                holding_period = self.speculation_horizon[i]
            elif agent_type == 1:
                holding_period = 84

            rental_yield = 0.003
            current_total_value = self.total_purchase_price[i]

            transfer_tax_rate = transfer_tax_multi
            if owned == 0:
                transfer_tax_rate = 0.40
            elif holding_period < 24:
                transfer_tax_rate = 0.70

            final_buy_decision = 0

            if owned == 0:
                final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0

            elif owned == 1:
                has_lifecycle_reason = (life_stage == 2) or (life_stage == 3) or (life_stage == 5)
                if has_lifecycle_reason:
                    final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0
                    if final_buy_decision == 1:
                        self.wants_to_sell[i] = 1
                else:
                    profitability = calculate_investment_profitability(
                        price, 2, monthly_appreciation, holding_period, rental_yield,
                        jongbu_threshold, jongbu_threshold * 0.55, jongbu_rate,
                        transfer_tax_rate, current_total_value
                    )
                    if profitability > 0 and (buy_score + noise) > buy_threshold:
                        final_buy_decision = 1

            elif owned == 2:
                profitability = calculate_investment_profitability(
                    price, 3, monthly_appreciation, holding_period, rental_yield,
                    jongbu_threshold, jongbu_threshold * 0.55, jongbu_rate,
                    transfer_tax_rate, current_total_value
                )
                if profitability > 0 and (buy_score + noise) > buy_threshold:
                    final_buy_decision = 1

            else:
                profitability = calculate_investment_profitability(
                    price, owned + 1, monthly_appreciation, holding_period, rental_yield,
                    jongbu_threshold, jongbu_threshold * 0.55, jongbu_rate,
                    transfer_tax_rate, current_total_value
                )
                if profitability > 0 and (buy_score + noise) > buy_threshold * 1.5:
                    final_buy_decision = 1

            self.wants_to_buy[i] = final_buy_decision

            # === 매도 의사결정 ===
            sell_score = 0.0

            if owned >= 1:
                current_value = price
                gain_loss = 0.0
                gain_loss_ratio = 0.0
                if purchase_price_val > 0:
                    gain_loss = current_value - purchase_price_val
                    gain_loss_ratio = gain_loss / purchase_price_val

                # Prospect Theory 손실 회피
                pt_sell_value = prospect_value(
                    gain_loss_ratio, pt_alpha, pt_beta, loss_aversion_coef
                )
                loss_penalty = 0.0
                if pt_sell_value < 0:
                    loss_penalty = ti.min(-pt_sell_value * 0.3, 0.4)

                # 앵커링
                anchoring_penalty = 0.0
                if gain_loss_ratio < 0.1:
                    anchoring_penalty = (0.1 - gain_loss_ratio) * 0.3

                if owned >= 2:  # 다주택자
                    total_value = price * ti.cast(owned, ti.f32)
                    holding_cost = 0.0
                    if total_value > jongbu_threshold:
                        holding_cost = (total_value - jongbu_threshold) * jongbu_rate / 12.0
                    holding_burden = holding_cost / income if income > 0 else 0.0
                    holding_bonus = 0.0
                    if holding_burden > 0.05:
                        holding_bonus = ti.min((holding_burden - 0.05) * 0.2, 0.10)

                    tax_penalty = 0.0
                    if transfer_tax_multi > 0.5:
                        tax_penalty = (transfer_tax_multi - 0.5) * 0.3

                    expectation_bonus = 0.0
                    if expectation < -0.3:
                        expectation_bonus = 0.10

                    profit_bonus = 0.0
                    if gain_loss_ratio > 0.3:
                        certainty_bonus = probability_weight(0.9, pt_gamma_gain) * 0.10
                        profit_bonus = 0.08 + certainty_bonus

                    sell_score = (holding_bonus + expectation_bonus + profit_bonus
                                 - tax_penalty - loss_penalty - anchoring_penalty)

                elif owned == 1:
                    base_sell = 0.0
                    if life_stage == 3 and eldest_child >= 10:
                        base_sell = 0.1
                    elif life_stage == 5:
                        if age >= 60:
                            base_sell = 0.08
                            if gain_loss_ratio > 0.5:
                                base_sell += 0.1
                    sell_score = base_sell - loss_penalty * 1.5 - anchoring_penalty

            if price_trend < -0.03:
                sell_score -= 0.15
            sell_score = ti.max(sell_score, 0.0)

            self.wants_to_sell[i] = 1 if sell_score > sell_threshold else 0

    @ti.kernel
    def update_reference_price(self, region_prices: ti.template(), decay_rate: ti.f32):
        """참조 가격 업데이트 (Prospect Theory)

        참조점은 시간이 지남에 따라 현재 시장가격에 적응
        """
        for i in range(self.n):
            if self.owned_houses[i] > 0:
                region = self.region[i]
                current_price = region_prices[region]

                # 참조점 적응: 현재가격 방향으로 서서히 이동
                old_ref = self.reference_price[i]
                if old_ref > 0:
                    # 지수적 적응
                    self.reference_price[i] = old_ref * (1.0 - decay_rate) + current_price * decay_rate
                else:
                    self.reference_price[i] = current_price

    @ti.kernel
    def update_homeless_months(self):
        """무주택 기간 업데이트"""
        for i in range(self.n):
            if self.owned_houses[i] == 0:
                self.homeless_months[i] += 1
            else:
                self.homeless_months[i] = 0

    @ti.kernel
    def update_assets(self, savings_rate: ti.f32, min_living_cost: ti.f32,
                      mortgage_rate_monthly: ti.f32):
        """자산 업데이트 (저축/소진)

        취업자: 소득의 savings_rate만큼 저축 (대출이자 차감 후)
        실업자: 생활비 + 대출이자로 자산 소진
        소득 성장은 JobMarket에서 처리 (기존 uniform 성장 제거)
        """
        for i in range(self.n):
            # 대출이자 (취업/실업 무관하게 납부)
            mortgage_payment = self.mortgage_balance[i] * mortgage_rate_monthly

            if self.employment_status[i] == 0:  # 취업
                net_income = self.income[i] - mortgage_payment - min_living_cost
                if net_income > 0.0:
                    self.asset[i] += net_income * savings_rate
                else:
                    self.asset[i] += net_income  # 적자분 자산에서 차감
            else:  # 실업 (급여 수령 또는 무급)
                # 총 비용 = 생활비 + 대출이자
                total_cost = min_living_cost + mortgage_payment
                shortfall = ti.max(total_cost - self.income[i], 0.0)
                self.asset[i] -= shortfall

            # 자산 하한 (완전 파산 방지 - 최소 0)
            self.asset[i] = ti.max(self.asset[i], 0.0)

    @ti.kernel
    def update_yearly_aging(self):
        """연간 나이 및 생애주기 업데이트 (1월에 호출)"""
        for i in range(self.n):
            # 나이 증가
            self.age[i] += 1

            # 자녀 나이 증가
            if self.eldest_child_age[i] >= 0:
                self.eldest_child_age[i] += 1

            # 생애주기 재계산
            age = self.age[i]
            is_married = self.is_married[i]
            num_children = self.num_children[i]
            eldest_child = self.eldest_child_age[i]

            if is_married == 0:
                self.life_stage[i] = 0  # 미혼
            elif age >= 60:
                self.life_stage[i] = 5  # 은퇴기
            elif num_children == 0:
                self.life_stage[i] = 1  # 신혼
            elif eldest_child <= 6:
                self.life_stage[i] = 2  # 육아기
            elif eldest_child <= 18:
                self.life_stage[i] = 3  # 학령기
            elif eldest_child > 18:
                self.life_stage[i] = 4  # 빈둥지

    def update_life_events(self, rng: np.random.Generator, current_month: int):
        """생애 이벤트 처리 (결혼, 출산 등) - NumPy 기반"""
        ages = self.age.to_numpy()
        is_married = self.is_married.to_numpy()
        num_children = self.num_children.to_numpy()
        eldest_child_age = self.eldest_child_age.to_numpy()
        life_stage = self.life_stage.to_numpy()

        # 결혼 이벤트 (미혼자 대상)
        unmarried_mask = (is_married == 0)
        marriage_candidates = unmarried_mask & (ages >= 25) & (ages <= 45)

        # 연령별 결혼 확률 (연간 기준을 월간으로 변환)
        marriage_prob = np.zeros(self.n, dtype=np.float32)
        marriage_prob[(ages >= 25) & (ages < 30)] = 0.15 / 12  # 연 15%
        marriage_prob[(ages >= 30) & (ages < 35)] = 0.20 / 12  # 연 20%
        marriage_prob[(ages >= 35) & (ages < 40)] = 0.10 / 12  # 연 10%
        marriage_prob[(ages >= 40) & (ages <= 45)] = 0.05 / 12  # 연 5%

        new_marriages = marriage_candidates & (rng.random(self.n) < marriage_prob)
        is_married[new_marriages] = 1
        life_stage[new_marriages] = 1  # 신혼

        # 출산 이벤트 (기혼자, 자녀 2명 이하)
        birth_candidates = (is_married == 1) & (num_children < 3) & (ages >= 25) & (ages <= 42)

        # 출산 확률 (연간 기준을 월간으로 변환)
        birth_prob = np.zeros(self.n, dtype=np.float32)
        birth_prob[(ages >= 25) & (ages < 30) & (num_children == 0)] = 0.15 / 12
        birth_prob[(ages >= 30) & (ages < 35) & (num_children == 0)] = 0.20 / 12
        birth_prob[(ages >= 35) & (ages < 40) & (num_children == 0)] = 0.10 / 12
        birth_prob[(ages >= 25) & (ages < 35) & (num_children == 1)] = 0.12 / 12
        birth_prob[(ages >= 35) & (ages < 40) & (num_children == 1)] = 0.08 / 12
        birth_prob[(ages >= 25) & (ages < 38) & (num_children == 2)] = 0.03 / 12

        new_births = birth_candidates & (rng.random(self.n) < birth_prob)
        first_births = new_births & (num_children == 0)
        eldest_child_age[first_births] = 0  # 첫째 출생
        num_children[new_births] += 1
        life_stage[new_births] = 2  # 육아기

        # 필드 업데이트
        self.is_married.from_numpy(is_married)
        self.num_children.from_numpy(num_children)
        self.eldest_child_age.from_numpy(eldest_child_age)
        self.life_stage.from_numpy(life_stage)

    def record_purchase(self, buyer_id: int, house_price: float, current_month: int):
        """주택 매수 기록 (거래 시 호출)"""
        # NumPy 배열로 가져와서 수정
        purchase_prices = self.purchase_price.to_numpy()
        purchase_months = self.purchase_month.to_numpy()
        total_prices = self.total_purchase_price.to_numpy()
        owned = self.owned_houses.to_numpy()

        # 첫 주택 또는 주 주택 업데이트
        if owned[buyer_id] == 1:  # 방금 1주택자가 됨
            purchase_prices[buyer_id] = house_price
        total_prices[buyer_id] += house_price
        purchase_months[buyer_id] = current_month

        self.purchase_price.from_numpy(purchase_prices)
        self.purchase_month.from_numpy(purchase_months)
        self.total_purchase_price.from_numpy(total_prices)

    def record_sale(self, seller_id: int, house_price: float):
        """주택 매도 기록 (거래 시 호출)"""
        total_prices = self.total_purchase_price.to_numpy()
        purchase_prices = self.purchase_price.to_numpy()
        owned = self.owned_houses.to_numpy()

        # 총 매입가에서 차감 (평균값 기준)
        if owned[seller_id] > 0:
            avg_purchase = total_prices[seller_id] / (owned[seller_id] + 1)
            total_prices[seller_id] -= avg_purchase

        # 무주택자가 되면 매입가 초기화
        if owned[seller_id] == 0:
            purchase_prices[seller_id] = 0.0
            total_prices[seller_id] = 0.0

        self.purchase_price.from_numpy(purchase_prices)
        self.total_purchase_price.from_numpy(total_prices)
