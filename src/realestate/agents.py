"""가구 에이전트 정의 (Taichi fields) - 행동경제학 기반"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, REGION_HOUSEHOLD_RATIO


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

    def initialize(self, rng: np.random.Generator):
        """초기 상태 설정 (행동경제학 요소 포함)"""
        # 연령 분포 (25-80)
        age_probs = np.array([0.20, 0.25, 0.25, 0.18, 0.12])
        age_bins = [(25, 34), (35, 44), (45, 54), (55, 64), (65, 80)]

        ages = np.zeros(self.n, dtype=np.int32)
        idx = 0
        for prob, (low, high) in zip(age_probs, age_bins):
            count = int(self.n * prob)
            ages[idx:idx+count] = rng.integers(low, high+1, size=count)
            idx += count
        ages[idx:] = rng.integers(25, 80, size=self.n - idx)
        rng.shuffle(ages)

        # 소득 분포 (로그정규, 중위값 300만원)
        incomes = rng.lognormal(mean=np.log(300), sigma=0.6, size=self.n).astype(np.float32)
        incomes = np.clip(incomes, 100, 10000)

        # 자산 분포 (파레토, 중위값 5000만원)
        assets = (rng.pareto(a=1.5, size=self.n) + 1) * 3000
        assets = assets.astype(np.float32)

        # 지역 배치
        regions = rng.choice(NUM_REGIONS, size=self.n, p=REGION_HOUSEHOLD_RATIO).astype(np.int32)

        # 주택 보유 (45% 무주택, 40% 1주택, 15% 다주택)
        ownership_roll = rng.random(self.n)
        owned = np.zeros(self.n, dtype=np.int32)
        owned[ownership_roll >= 0.45] = 1
        owned[ownership_roll >= 0.85] = rng.integers(2, 6, size=np.sum(ownership_roll >= 0.85))

        # 자산과 주택 보유 상관관계 조정 (자산 많은 사람이 다주택자일 확률 높음)
        asset_rank = np.argsort(assets)[::-1]
        multi_owner_indices = np.where(owned >= 2)[0]
        top_asset_indices = asset_rank[:len(multi_owner_indices) * 2]
        # 다주택자를 상위 자산가 중에서 선택
        new_multi = rng.choice(top_asset_indices, size=len(multi_owner_indices), replace=False)
        owned_new = np.zeros(self.n, dtype=np.int32)
        owned_new[ownership_roll >= 0.45] = 1
        for i in new_multi:
            owned_new[i] = rng.integers(2, 6)

        # 가격 기대 (-1 ~ 1)
        expectations = rng.normal(0.1, 0.3, size=self.n).astype(np.float32)
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
        # 유형 분포:
        # - 실수요자 (owner-occupier): 80% - 거주 목적 구매
        # - 투자자 (investor): 15% - 임대 수익 목적
        # - 투기자 (speculator): 5% - 시세차익 목적, 단기 보유
        # 학술적 근거: JASSS 2020 "Housing Market ABM with LTV and DTI"
        # 투기자는 주로 자산 상위층, 젊은 층에 분포
        agent_type = np.zeros(self.n, dtype=np.int32)  # 기본: 실수요자

        # 자산 상위 30% 중에서 투자자/투기자 선정
        asset_percentile_70 = np.percentile(assets, 70)
        high_asset_mask = assets >= asset_percentile_70

        # 투자자: 자산 상위 30% 중 50% (전체의 약 15%)
        investor_candidates = np.where(high_asset_mask)[0]
        n_investors = int(len(investor_candidates) * 0.5)
        investors = rng.choice(investor_candidates, size=n_investors, replace=False)
        agent_type[investors] = 1

        # 투기자: 자산 상위 30% & 나이 25-50세 중 일부 (전체의 약 5%)
        speculator_candidates = np.where(high_asset_mask & (ages >= 25) & (ages <= 50))[0]
        speculator_candidates = np.setdiff1d(speculator_candidates, investors)  # 투자자 제외
        n_speculators = min(int(self.n * 0.05), len(speculator_candidates))
        if n_speculators > 0:
            speculators = rng.choice(speculator_candidates, size=n_speculators, replace=False)
            agent_type[speculators] = 2

        # 투기자 특성 조정
        # - 높은 위험 허용도
        # - 높은 FOMO 민감도
        # - 낮은 손실 회피 (과신 성향)
        # - 높은 군집 성향
        speculator_mask = agent_type == 2
        risk_tolerance[speculator_mask] = np.clip(
            risk_tolerance[speculator_mask] * 1.5 + 0.2, 0, 1
        )
        fomo_sensitivity[speculator_mask] = np.clip(
            fomo_sensitivity[speculator_mask] * 1.3 + 0.2, 0, 1
        )
        loss_aversion[speculator_mask] = np.clip(
            loss_aversion[speculator_mask] * 0.7, 1.5, 2.5  # 낮은 손실 회피
        )
        herding_tendency[speculator_mask] = np.clip(
            herding_tendency[speculator_mask] * 1.4 + 0.1, 0, 1
        )

        # 투기자의 목표 보유 기간 (6-24개월, 단기)
        speculation_horizon = np.zeros(self.n, dtype=np.int32)
        speculation_horizon[speculator_mask] = rng.integers(6, 25, size=np.sum(speculator_mask))

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

    @ti.kernel
    def update_expectations(self, price_changes: ti.template(), social_weight: ti.f32):
        """가격 기대 업데이트 (적응적 기대 + 사회적 학습 + FOMO)"""
        for i in range(self.n):
            region = self.region[i]
            observed_change = price_changes[region]

            # 적응적 기대: 과거 변화를 반영 (비대칭: 상승은 빠르게, 하락은 느리게)
            # Taichi에서는 if 문 전에 변수를 미리 초기화해야 함
            adaptation_rate = 0.2  # 기본값 (하락장)
            if observed_change > 0:
                # 상승장: 빠르게 기대 조정 (FOMO 효과)
                adaptation_rate = 0.4 + self.fomo_sensitivity[i] * 0.2

            adaptive = self.price_expectation[i] * (1.0 - adaptation_rate) + observed_change * 10.0 * adaptation_rate

            # 군집 효과: 지역 내 다른 사람들의 매수 행동 참조
            region_buy_rate = self.region_buy_rate[region]
            herding_effect = self.herding_tendency[i] * (region_buy_rate - 0.05) * 2.0

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
        dti_limit: ti.f32,
        interest_rate: ti.f32,
        buy_threshold: ti.f32,
        sell_threshold: ti.f32,
        transfer_tax_multi: ti.f32,
        jongbu_rate: ti.f32,
        jongbu_threshold: ti.f32
    ):
        """매수/매도 의사결정 (행동경제학 기반)

        행동경제학 요소:
        - FOMO: 가격 상승 시 매수 욕구 비선형 급증
        - 손실 회피: 손실 시 매도 확률 급감 (손해보고 못 팔음)
        - 앵커링: 매입가에 집착, 그 이하로 안 팔려함
        - 군집 행동: 주변 매수 증가 시 따라 매수
        - 생애주기: 결혼/육아/학군/은퇴에 따른 수요
        """
        for i in range(self.n):
            owned = self.owned_houses[i]
            income = self.income[i]
            asset = self.asset[i]
            region = self.region[i]
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

            price = region_prices[region]
            price_trend = self.region_price_trend_6m[region]  # 6개월 상승률

            # === 매수 의사결정 ===
            buy_score = 0.0

            # 1. 구매력 계산 (기본)
            ltv = ltv_limits[ti.min(owned, 2)]
            max_loan = ti.min(income * 12.0 * dti_limit / interest_rate, price * ltv)
            buying_power = asset * 0.5 + max_loan

            affordability = buying_power / price if price > 0 else 0.0
            affordability = ti.min(affordability, 2.0)

            # 2. 생애주기 기반 주거 긴급도
            urgency = 0.0
            life_stage_factor = 1.0

            if owned == 0:  # 무주택자
                urgency = 0.4  # 기본 무주택 압박

                # 생애주기별 긴급도
                if life_stage == 0:  # 미혼
                    if 28 <= age <= 35:
                        urgency += 0.2  # 결혼 준비기
                elif life_stage == 1:  # 신혼
                    urgency += 0.4  # 신혼집 마련 압박 최대
                    life_stage_factor = 1.5
                elif life_stage == 2:  # 육아기
                    urgency += 0.35  # 넓은 집 필요
                    life_stage_factor = 1.3
                elif life_stage == 3:  # 학령기
                    urgency += 0.3  # 학군 이동 수요
                    if 10 <= eldest_child <= 15:  # 중학교 입학 전후
                        urgency += 0.15
                elif life_stage == 5:  # 은퇴기
                    urgency += 0.1  # 안정적 주거

                # 무주택 기간에 따른 초조함
                if homeless > 12:
                    urgency += ti.min(homeless / 120.0, 0.3)  # 최대 0.3 추가

            elif owned == 1:  # 1주택자 갈아타기
                urgency = 0.05
                if life_stage == 2:  # 육아기: 넓은 집
                    urgency += 0.15
                elif life_stage == 3:  # 학령기: 학군 이동
                    urgency += 0.2
                    if 10 <= eldest_child <= 15:
                        urgency += 0.1

            # 3. FOMO (Fear Of Missing Out) - 비선형 급증
            fomo_factor = 1.0
            if price_trend > 0.05:  # 6개월간 5% 이상 상승
                # FOMO 급증: 상승률에 따라 기하급수적 증가
                excess_rise = price_trend - 0.05
                fomo_factor = 1.0 + fomo_sens * excess_rise * excess_rise * 50.0
                # 최대 3배까지 제한
                fomo_factor = ti.min(fomo_factor, 3.0)
            elif price_trend > 0.02:  # 2-5% 상승
                fomo_factor = 1.0 + fomo_sens * (price_trend - 0.02) * 5.0

            # 4. 군집 행동 (Herding)
            region_buying = self.region_buy_rate[region]
            herding_factor = 1.0
            if region_buying > 0.03:  # 지역 내 3% 이상이 매수 시도
                herding_factor = 1.0 + herding * (region_buying - 0.03) * 10.0
                herding_factor = ti.min(herding_factor, 2.0)

            # 5. 기대 수익 (투자 목적, 다주택자)
            expected_return = 0.0
            if owned >= 1:
                expected_return = (expectation + 0.3) * risk * 0.3

            # 6. 최종 매수 점수 계산
            if affordability > 0.4:  # 40% 이상 감당 가능
                base_score = urgency * 0.5 + expected_return * 0.2 + (affordability - 0.4) * 0.15
                buy_score = base_score * life_stage_factor * fomo_factor * herding_factor

            # 확률적 결정 (노이즈 추가)
            seed = self.rand_seed[i]
            noise = ti.cast((seed % 1000) - 500, ti.f32) / 10000.0
            self.rand_seed[i] = (seed * 1103515245 + 12345) % 2147483647

            self.wants_to_buy[i] = 1 if (buy_score + noise) > buy_threshold else 0
            self.target_region[i] = region

            # === 매도 의사결정 ===
            sell_score = 0.0

            if owned >= 1:
                # 현재 가치 대비 이익/손실 계산
                current_value = price
                gain_loss_ratio = 0.0
                if purchase_price_val > 0:
                    gain_loss_ratio = (current_value - purchase_price_val) / purchase_price_val

                # 1. 손실 회피 (Loss Aversion)
                # Kahneman의 전망이론: 손실의 고통 = 이익의 기쁨 * 2.25배
                loss_aversion_factor = 1.0
                if gain_loss_ratio < 0:
                    # 손실 상태: 매도 확률 급감
                    loss_ratio = -gain_loss_ratio
                    loss_aversion_factor = ti.exp(-loss_aversion_coef * loss_ratio * 5.0)
                    # 10% 손실 시 매도 확률 약 30%로 감소

                # 2. 앵커링 (Anchoring)
                # 매입가 이하로는 팔기 싫음
                anchoring_penalty = 0.0
                if gain_loss_ratio < 0.1:  # 10% 미만 수익
                    # 매입가 근처에서는 매도 꺼림
                    anchoring_penalty = (0.1 - gain_loss_ratio) * 0.5

                if owned >= 2:  # 다주택자
                    # 보유 비용
                    total_value = price * ti.cast(owned, ti.f32)
                    holding_cost = 0.0
                    if total_value > jongbu_threshold:
                        holding_cost = (total_value - jongbu_threshold) * jongbu_rate / 12.0

                    holding_burden = holding_cost / income if income > 0 else 0.0

                    # 양도세 부담
                    tax_burden = transfer_tax_multi

                    # 기대 수익 낮으면 매도 고려
                    if expectation < -0.1:
                        sell_score += 0.25

                    # 보유 비용 압박
                    sell_score += holding_burden * 0.4

                    # 양도세 부담 낮을 때 매도 선호
                    sell_score += (1.0 - tax_burden) * 0.25

                    # 충분한 이익 실현 시 매도 고려
                    if gain_loss_ratio > 0.3:  # 30% 이상 이익
                        sell_score += 0.2

                    # 손실 회피 및 앵커링 적용
                    sell_score = sell_score * loss_aversion_factor - anchoring_penalty

                elif owned == 1:  # 1주택자 (갈아타기 목적만)
                    # 1주택자는 갈아타기 시에만 매도
                    if life_stage == 3 and eldest_child >= 10:  # 학군 이동
                        sell_score = 0.15
                    elif life_stage == 5:  # 은퇴 후 현금화 또는 다운사이징
                        if age >= 60:
                            sell_score = 0.1
                            if gain_loss_ratio > 0.5:  # 50% 이상 이익 시
                                sell_score += 0.15

                    # 손실 회피 적용 (1주택자는 더 강함)
                    sell_score = sell_score * loss_aversion_factor * 0.5 - anchoring_penalty

            # 하락장에서 매물 잠김 현상
            if price_trend < -0.03:  # 3% 이상 하락 중
                # 시장 하락 시 매도 더욱 꺼림 (역의 FOMO)
                sell_score *= 0.5

            self.wants_to_sell[i] = 1 if sell_score > sell_threshold else 0

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
