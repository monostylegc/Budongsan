"""가구 에이전트 정의 (Taichi fields) - 행동경제학 기반"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, REGION_HOUSEHOLD_RATIO, REGION_PRESTIGE, REGION_JOB_DENSITY


# =============================================================================
# Prospect Theory Functions (Kahneman & Tversky, 1992)
# =============================================================================

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

    def select_target_regions(self, market, rng: np.random.Generator):
        """에이전트 유형별 목표 지역 선택 (구조적 개선)

        실수요자: 거주 지역 위주, 인근 지역도 고려 (affordability 기반)
        투자자: 투자매력도 높은 지역 탐색 (기대수익률 기반)
        투기자: 가격 상승률 높은 지역 탐색 (모멘텀 기반)
        """
        agent_types = self.agent_type.to_numpy()
        home_regions = self.region.to_numpy()
        assets = self.asset.to_numpy()
        incomes = self.income.to_numpy()

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
        self.region_prestige.from_numpy(REGION_PRESTIGE.astype(np.float32))
        self.region_job_density.from_numpy(REGION_JOB_DENSITY.astype(np.float32))

        target_regions = np.copy(home_regions)  # 기본값: 거주 지역

        # 자산 상위 20% 기준값 (고자산가 판별용)
        asset_threshold_80 = np.percentile(assets, 80)

        for i in range(self.n):
            agent_type = agent_types[i]
            home = home_regions[i]
            asset = assets[i]
            income = incomes[i]

            # 고자산가 여부 (상위 20%)
            is_wealthy = asset >= asset_threshold_80

            if agent_type == 0:  # 실수요자
                # 일자리가 있는 곳에 살아야 함 (출퇴근 필수)
                # 일자리 밀도가 높은 지역 + 인근 지역 후보
                candidates = self._get_adjacent_regions(home)
                candidates.append(home)

                # 일자리 밀도 높은 지역도 후보에 추가 (출퇴근 가능 범위)
                high_job_regions = [r for r in range(NUM_REGIONS) if REGION_JOB_DENSITY[r] >= 0.5]
                candidates = list(set(candidates + high_job_regions))

                # 고자산가는 프리미엄 지역도 고려
                if is_wealthy:
                    candidates = list(set(candidates + [0, 1, 3]))  # 강남, 마용성, 분당

                best_region = home
                best_score = -999.0

                for r in candidates:
                    if prices[r] <= 0:
                        continue
                    buying_power = asset * 0.5 + income * 12 * 5

                    # 고자산가는 더 높은 구매력 (상속/증여, 기존 주택 매도금)
                    if is_wealthy:
                        buying_power *= 1.5

                    afford = buying_power / prices[r]

                    # 일자리 밀도 점수 (핵심 요인!)
                    # 일자리 없는 곳은 출퇴근 불가 → 실수요 없음
                    job_score = REGION_JOB_DENSITY[r] * 0.5  # 일자리 밀도 가중치 높임

                    # PIR 기반 점수 (가중치 낮춤 - 일자리가 더 중요)
                    pir_score = max(0, 1 - pir[r] / 30) * 0.15

                    # 프리미엄 지역 심리적 보너스
                    prestige = REGION_PRESTIGE[r]
                    prestige_bonus = prestige * (0.2 if is_wealthy else 0.1)

                    score = afford * 0.3 + job_score + pir_score + prestige_bonus

                    # affordability 임계값: 일자리 밀도가 높으면 낮은 afford도 허용 (영끌)
                    afford_threshold = 0.15 if REGION_JOB_DENSITY[r] >= 0.5 else 0.2
                    if is_wealthy:
                        afford_threshold = 0.10  # 고자산가는 더 낮은 임계값

                    if score > best_score and afford > afford_threshold:
                        best_score = score
                        best_region = r

                target_regions[i] = best_region

            elif agent_type == 1:  # 투자자
                # 투자매력도 + 프리미엄 지역 선호
                buying_power = asset * 0.5 + income * 12 * 5
                if is_wealthy:
                    buying_power *= 1.5
                affordable_mask = (prices > 0) & (buying_power / prices > 0.12)

                if np.sum(affordable_mask) > 0:
                    # 투자매력도 + 프리미엄 가중 점수
                    combined_score = np.where(
                        affordable_mask,
                        attractiveness + REGION_PRESTIGE * 0.2,  # 프리미엄 지역 가중
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
                # 가격 상승률 + 프리미엄 지역 모멘텀 효과
                buying_power = asset * 0.5 + income * 12 * 5
                if is_wealthy:
                    buying_power *= 1.5
                affordable_mask = (prices > 0) & (buying_power / prices > 0.12)

                if np.sum(affordable_mask) > 0:
                    # 모멘텀 + 프리미엄 가중 점수
                    combined_score = np.where(
                        affordable_mask,
                        price_trends + REGION_PRESTIGE * 0.15,  # 프리미엄 지역 가중
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

    @ti.kernel
    def decide_buy_sell(
        self,
        region_prices: ti.template(),
        ltv_limits: ti.template(),
        acq_tax_rates: ti.template(),  # 취득세율 (주택 보유 수별)
        dti_limit: ti.f32,
        interest_rate: ti.f32,
        buy_threshold: ti.f32,
        sell_threshold: ti.f32,
        transfer_tax_multi: ti.f32,
        jongbu_rate: ti.f32,
        jongbu_threshold: ti.f32,
        # Prospect Theory 파라미터
        pt_alpha: ti.f32,
        pt_beta: ti.f32,
        pt_gamma_gain: ti.f32,
        pt_gamma_loss: ti.f32
    ):
        """매수/매도 의사결정 (행동경제학 기반, Prospect Theory + Hyperbolic Discounting)

        행동경제학 요소:
        - Prospect Theory (Kahneman & Tversky, 1992): S자형 가치 함수, 손실 회피
        - Hyperbolic Discounting (Laibson, 1997): 현재 편향
        - FOMO: 가격 상승 시 매수 욕구 증가 (덧셈 기반, 제한된 배율)
        - 앵커링: 매입가에 집착
        - 군집 행동: 주변 매수 증가 시 따라 매수
        - 생애주기: 결혼/육아/학군/은퇴에 따른 수요

        수정 (2024): 매수/매도 점수 계산을 덧셈 기반으로 통일하여 비대칭 문제 해결
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            income = self.income[i]
            asset = self.asset[i]
            region = self.region[i]
            target = self.target_region[i]  # 목표 지역 (select_target_regions에서 설정)
            expectation = self.price_expectation[i]
            risk = self.risk_tolerance[i]
            age = self.age[i]
            homeless = self.homeless_months[i]
            life_stage = self.life_stage[i]
            fomo_sens = self.fomo_sensitivity[i]
            herding = self.herding_tendency[i]
            loss_aversion_coef = self.loss_aversion[i]
            purchase_price_val = self.purchase_price[i]
            eldest_child = self.eldest_child_age[i]
            beta = self.discount_beta[i]
            delta = self.discount_delta[i]

            # 목표 지역의 가격 및 지표 사용
            price = region_prices[target]
            price_trend = self.region_price_trend_6m[target]  # 6개월 상승률

            # 가격 적정성 지표 (구조적 개선)
            pir = self.region_pir[target]  # 소득대비가격비
            price_to_hist = self.region_price_to_hist[target]  # 역사적 평균 대비
            attractiveness = self.region_attractiveness[target]  # 투자 매력도
            prestige = self.region_prestige[target]  # 심리적 프리미엄
            job_density = self.region_job_density[target]  # 일자리 밀도

            # === 매수 의사결정 ===
            buy_score = 0.0

            # 1. 구매력 계산 (기본)
            ltv = ltv_limits[ti.min(owned, 2)]
            max_loan = ti.min(income * 12.0 * dti_limit / interest_rate, price * ltv)
            buying_power = asset * 0.5 + max_loan

            affordability = buying_power / price if price > 0 else 0.0
            affordability = ti.min(affordability, 2.0)

            # 2. 생애주기 기반 주거 긴급도 (덧셈 기반)
            # 수정 (2024): 기본 urgency를 낮추어 매수 희망 비율 현실화
            urgency = 0.0
            life_stage_bonus = 0.0

            if owned == 0:  # 무주택자
                urgency = 0.15  # 기본 무주택 압박 (0.3 → 0.15로 하향)

                # 생애주기별 긴급도 (덧셈 방식, 값 축소)
                if life_stage == 0:  # 미혼
                    if 28 <= age <= 35:
                        urgency += 0.10  # 결혼 준비기
                elif life_stage == 1:  # 신혼
                    urgency += 0.20  # 신혼집 마련 압박 최대
                    life_stage_bonus = 0.10
                elif life_stage == 2:  # 육아기
                    urgency += 0.15  # 넓은 집 필요
                    life_stage_bonus = 0.08
                elif life_stage == 3:  # 학령기
                    urgency += 0.12  # 학군 이동 수요
                    if 10 <= eldest_child <= 15:
                        urgency += 0.08
                elif life_stage == 5:  # 은퇴기
                    urgency += 0.05  # 안정적 주거

                # 무주택 기간에 따른 초조함 (더 느리게)
                if homeless > 24:  # 2년 이상
                    urgency += ti.min(homeless / 300.0, 0.15)

            elif owned == 1:  # 1주택자 갈아타기
                urgency = 0.02
                if life_stage == 2:  # 육아기: 넓은 집
                    urgency += 0.08
                elif life_stage == 3:  # 학령기: 학군 이동
                    urgency += 0.10
                    if 10 <= eldest_child <= 15:
                        urgency += 0.06

            # 3. FOMO (Fear Of Missing Out) - 덧셈 기반으로 변경
            fomo_bonus = 0.0
            if price_trend > 0.05:  # 6개월간 5% 이상 상승
                # FOMO: 덧셈 방식, 최대값 제한
                excess_rise = price_trend - 0.05
                fomo_bonus = fomo_sens * excess_rise * 3.0  # 배율 축소
                fomo_bonus = ti.min(fomo_bonus, 0.3)  # 최대 0.3
            elif price_trend > 0.02:  # 2-5% 상승
                fomo_bonus = fomo_sens * (price_trend - 0.02) * 1.5
                fomo_bonus = ti.min(fomo_bonus, 0.15)

            # 4. 군집 행동 (Herding) - 덧셈 기반으로 변경
            herding_bonus = 0.0
            region_buying = self.region_buy_rate[region]
            if region_buying > 0.03:  # 지역 내 3% 이상이 매수 시도
                herding_bonus = herding * (region_buying - 0.03) * 3.0
                herding_bonus = ti.min(herding_bonus, 0.2)  # 최대 0.2

            # 5. 기대 수익 (Hyperbolic Discounting 적용)
            expected_return_bonus = 0.0
            if owned >= 1:
                # 월간 기대 수익률 (가격상승 + 임대수익)
                monthly_appreciation = expectation * 0.01
                rental_yield = 0.003

                horizon = 60
                discounted_return = expected_investment_return(
                    monthly_appreciation, rental_yield, horizon, beta, delta
                )
                expected_return_bonus = discounted_return * risk * 0.05  # 축소

            # 6. Prospect Theory 기반 이득 기대 평가 - 덧셈 기반
            pt_bonus = 0.0
            expected_gain = price_trend * price
            pt_value = prospect_value(expected_gain / price, pt_alpha, pt_beta, loss_aversion_coef)

            rise_prob = 0.5 + expectation * 0.3
            rise_prob = ti.math.clamp(rise_prob, 0.1, 0.9)
            weighted_prob = probability_weight(rise_prob, pt_gamma_gain)

            if pt_value > 0:
                pt_bonus = weighted_prob * pt_value * 0.05  # 덧셈 방식
                pt_bonus = ti.min(pt_bonus, 0.15)

            # 7. 가격 적정성 보너스/페널티 (구조적 개선)
            # PIR(소득대비가격비)과 역사적 평균 대비 현재가 기반
            # 고자산가(상위 20%)는 PIR 페널티 완화 - 프리미엄 지역 감당 가능
            valuation_bonus = 0.0

            # 고자산가 여부 판단 (자산 1억 이상 = 상위 ~20%)
            is_wealthy = asset >= 10000.0  # 10000만원 = 1억원

            # PIR 기반: 10 이하 적정(보너스), 15 이상 고평가(페널티)
            if pir < 10.0:
                valuation_bonus += (10.0 - pir) * 0.01  # 적정가 보너스
            elif pir > 15.0:
                # 고자산가는 PIR 페널티 50% 감소 (프리미엄 지역 감당 가능)
                pir_penalty = (pir - 15.0) * 0.02
                if is_wealthy:
                    pir_penalty *= 0.5  # 페널티 50% 감소
                valuation_bonus -= pir_penalty

            # 역사적 평균 대비: 1.0 이하 저평가(보너스), 1.2 이상 고평가(페널티)
            if price_to_hist < 1.0:
                valuation_bonus += (1.0 - price_to_hist) * 0.1  # 저평가 보너스
            elif price_to_hist > 1.2:
                hist_penalty = (price_to_hist - 1.2) * 0.15
                if is_wealthy:
                    hist_penalty *= 0.5  # 고자산가는 페널티 50% 감소
                valuation_bonus -= hist_penalty

            valuation_bonus = ti.math.clamp(valuation_bonus, -0.15, 0.15)

            # 8. 심리적 프리미엄 보너스 (Prestige Effect)
            # 강남, 마용성, 분당 등 프리미엄 지역에 대한 심리적 선호
            # 에이전트 타입별, 자산 수준별 차등
            agent_type = self.agent_type[i]
            prestige_bonus = prestige * 0.08  # 기본 심리적 프리미엄 (최대 0.08)

            if agent_type == 1:  # 투자자: 프리미엄 지역 임대수익 기대
                prestige_bonus = prestige * 0.12
            elif agent_type == 2:  # 투기자: 프리미엄 지역 시세차익 기대
                prestige_bonus = prestige * 0.15

            # 고자산가는 프리미엄 지역 선호도 더 강함 (1.5배)
            # "똘똘한 한채" 심리: 좋은 곳에 1채 집중
            if is_wealthy:
                prestige_bonus *= 1.5

            # 9. 일자리 밀도 보너스 (주거 수요의 핵심 요인)
            # 강남/판교에 고소득 일자리 집중 → 출퇴근 편의 → 주거 수요
            # 일자리 없는 지역은 수요 감소 → 가격 하락
            job_bonus = job_density * 0.15  # 일자리 밀도에 비례 (최대 0.15)

            # 실수요자(agent_type=0)는 일자리 중요성 더 높음 (출퇴근 필수)
            if agent_type == 0:
                job_bonus *= 1.3

            # 10. 다주택 규제 정책 (똘똘한 한채의 핵심 원인)
            # 한국의 다주택자 규제는 매우 강력함:
            # - 취득세: 2주택 8%, 3주택+ 12%
            # - 종부세: 다주택 합산 과세
            # - 양도세: 다주택 중과세 (최대 75%)
            # - 대출: 2주택 LTV 30%, 3주택+ 대출 불가
            # 이 정책들로 인해 "1채를 좋은 곳에" 전략이 합리적
            policy_penalty = 0.0

            if owned >= 2:  # 2주택 이상: 추가 매수 거의 불가능
                # 취득세 12% + 종부세 + 양도세 중과 + 대출 불가
                # → 투자 수익성 없음, 사실상 매수 포기
                policy_penalty = 1.5  # buy_threshold(0.4)보다 훨씬 큼 → 거의 매수 안함

            elif owned == 1:  # 1주택자: 추가 매수 vs 갈아타기 구분
                # "추가 매수"는 8% 취득세로 매우 억제됨
                # "갈아타기"는 기존 주택 매도 후 → wants_to_sell=1이어야 함
                if self.wants_to_sell[i] == 0:
                    # 갈아타기가 아닌 순수 추가 매수
                    # → 8% 취득세 + 종부세 + 양도세 중과 = 거의 안함
                    policy_penalty = 0.6  # 강한 억제

                    # 투자자/투기자만 약간의 여지 (하지만 여전히 큰 부담)
                    if agent_type == 1:  # 투자자
                        policy_penalty = 0.4
                    elif agent_type == 2:  # 투기자
                        policy_penalty = 0.35
                else:
                    # 갈아타기 (기존 주택 팔고 새로 사는 것)
                    # → 1% 취득세만 적용, 정상적인 결정
                    policy_penalty = 0.05

            # 11. 최종 매수 점수 계산 (모두 덧셈 방식)
            if affordability > 0.25:  # 25% 이상 감당 가능 (대출 활용 가정)
                affordability_bonus = ti.min((affordability - 0.25) * 0.15, 0.15)
                buy_score = (urgency + life_stage_bonus + fomo_bonus + herding_bonus +
                            expected_return_bonus + pt_bonus + affordability_bonus +
                            valuation_bonus + prestige_bonus + job_bonus -  # 일자리 보너스 추가
                            policy_penalty)  # 다주택 규제 정책 반영

            # 확률적 결정 (노이즈 추가)
            seed = self.rand_seed[i]
            noise = ti.cast((seed % 1000) - 500, ti.f32) / 10000.0
            self.rand_seed[i] = (seed * 1103515245 + 12345) % 2147483647

            # 다주택 규제 정책의 핵심: 보유 수에 따른 매수 결정 차등
            # 한국의 부동산 정책은 다주택 취득을 강력히 억제
            final_buy_decision = 0

            if owned == 0:  # 무주택자: 정상적인 매수 결정
                final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0

            elif owned == 1:  # 1주택자: 갈아타기 vs 추가 매수 구분
                # 1주택자가 매수하면 2주택자가 됨 → 8% 취득세
                # 따라서 "추가 매수"는 매우 드물어야 함
                # "갈아타기"(기존 집 팔고 새 집)는 1주택 유지이므로 정상

                # 생애주기상 갈아타기 이유: 학군 이동, 넓은 집, 다운사이징
                has_lifecycle_reason = (life_stage == 2) or (life_stage == 3) or (life_stage == 5)

                if has_lifecycle_reason:
                    # 갈아타기 목적: 정상적인 매수 결정 (1주택 → 1주택)
                    # 단, 매도도 함께 해야 진정한 갈아타기
                    final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0
                else:
                    # 생애주기 이유 없음 = 투자 목적 추가 매수 (1주택 → 2주택)
                    # 8% 취득세 고려 → 낮은 확률로만 허용 (5%)
                    seed2 = self.rand_seed[i]
                    add_buy_chance = ti.cast(seed2 % 1000, ti.f32) / 1000.0
                    self.rand_seed[i] = (seed2 * 1103515245 + 12345) % 2147483647
                    final_buy_decision = 1 if add_buy_chance < 0.05 else 0

            elif owned == 2:  # 2주택자: 8% 취득세 = 거의 매수 안함
                # buy_score와 무관하게 낮은 확률로만 매수 (3% 확률)
                # 8% 취득세 = 투자 수익 3-4년치 → 대부분 포기
                seed2 = self.rand_seed[i]
                rare_chance_2 = ti.cast(seed2 % 1000, ti.f32) / 1000.0
                self.rand_seed[i] = (seed2 * 1103515245 + 12345) % 2147483647
                # 3% 확률로만 매수 (특수 상황: 증여, 상속, 법인 매수 등)
                final_buy_decision = 1 if rare_chance_2 < 0.03 else 0

            else:  # 3주택 이상: 12% 취득세 + 대출 불가 = 매수 불가
                # buy_score와 무관하게 1% 확률로만 매수
                # 12% 취득세 + 종부세 + 양도세 중과 + 대출 불가 → 사실상 불가능
                seed2 = self.rand_seed[i]
                rare_chance_3 = ti.cast(seed2 % 1000, ti.f32) / 1000.0
                self.rand_seed[i] = (seed2 * 1103515245 + 12345) % 2147483647
                # 1% 확률로만 매수 (특수 상황: 증여, 상속 등)
                final_buy_decision = 1 if rare_chance_3 < 0.01 else 0

            self.wants_to_buy[i] = final_buy_decision
            # target_region은 select_target_regions()에서 별도 설정

            # === 매도 의사결정 (덧셈 기반 + 배율 보정) ===
            sell_score = 0.0

            if owned >= 1:
                # 현재 가치 대비 이익/손실 계산
                current_value = price
                gain_loss = 0.0
                gain_loss_ratio = 0.0
                if purchase_price_val > 0:
                    gain_loss = current_value - purchase_price_val
                    gain_loss_ratio = gain_loss / purchase_price_val

                # 1. Prospect Theory 가치 함수 적용 (손실 회피)
                normalized_gain_loss = gain_loss_ratio

                pt_sell_value = prospect_value(
                    normalized_gain_loss,
                    pt_alpha,
                    pt_beta,
                    loss_aversion_coef
                )

                # 손실 상태에서의 매도 확률 감소 (덧셈 기반 페널티)
                loss_penalty = 0.0
                if pt_sell_value < 0:
                    # 손실의 주관적 가치에 비례한 페널티
                    loss_penalty = ti.min(-pt_sell_value * 0.3, 0.4)

                # 2. 앵커링 (Anchoring) - 페널티 방식 유지
                anchoring_penalty = 0.0
                if gain_loss_ratio < 0.1:
                    anchoring_penalty = (0.1 - gain_loss_ratio) * 0.3

                if owned >= 2:  # 다주택자
                    # 보유 비용 압박
                    total_value = price * ti.cast(owned, ti.f32)
                    holding_cost = 0.0
                    if total_value > jongbu_threshold:
                        holding_cost = (total_value - jongbu_threshold) * jongbu_rate / 12.0

                    holding_burden = holding_cost / income if income > 0 else 0.0
                    holding_bonus = ti.min(holding_burden * 0.3, 0.2)

                    # 양도세 부담 낮을 때 매도 선호
                    tax_bonus = (1.0 - transfer_tax_multi) * 0.15

                    # 기대 수익 낮으면 매도 고려
                    expectation_bonus = 0.0
                    if expectation < -0.1:
                        expectation_bonus = 0.15

                    # 충분한 이익 실현 시 매도 고려
                    profit_bonus = 0.0
                    if gain_loss_ratio > 0.3:
                        certainty_bonus = probability_weight(0.9, pt_gamma_gain) * 0.15
                        profit_bonus = 0.12 + certainty_bonus

                    # 최종 매도 점수 (덧셈 - 페널티)
                    sell_score = (holding_bonus + tax_bonus + expectation_bonus +
                                 profit_bonus - loss_penalty - anchoring_penalty)

                elif owned == 1:  # 1주택자 (갈아타기 목적만)
                    base_sell = 0.0
                    if life_stage == 3 and eldest_child >= 10:  # 학군 이동
                        base_sell = 0.1
                    elif life_stage == 5:  # 은퇴 후 현금화
                        if age >= 60:
                            base_sell = 0.08
                            if gain_loss_ratio > 0.5:
                                base_sell += 0.1

                    # 1주택자는 손실 회피 더 강함
                    sell_score = base_sell - loss_penalty * 1.5 - anchoring_penalty

            # 하락장에서 매물 잠김 현상 (감소 방식)
            if price_trend < -0.03:
                sell_score -= 0.15  # 덧셈 기반 페널티

            # 음수 방지
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
    def update_assets(self, income_growth: ti.f32, savings_rate: ti.f32):
        """자산 업데이트 (저축)"""
        for i in range(self.n):
            monthly_saving = self.income[i] * savings_rate
            self.asset[i] += monthly_saving
            # 소득 성장 (연 단위를 월 단위로)
            self.income[i] *= (1.0 + income_growth / 12.0)

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
