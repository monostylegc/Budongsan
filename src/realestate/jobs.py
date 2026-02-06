"""일자리 시장 모듈 - 산업별 고용/실업/소득 모델 (numpy 벡터화)

핵심 역할:
- 지역×산업별 고용 수준 추적
- GDP 변화 → 산업별 일자리 생성/파괴
- 에이전트 고용 상태 업데이트 (취업/실업 전환)
- 소득 계산 (지역×산업 기반)
- 실업급여 지급
- 자산 소진 → 강제 매도 트리거

생존 메커니즘:
GDP 하락 → 산업별 일자리 파괴 → 에이전트 실직
→ 소득 0 (급여 만료 후) → 생활비로 자산 소진
→ 주거비 미납 12개월 → 강제매도(급매)

성능: 모든 에이전트 루프를 numpy 벡터 연산으로 처리 (230K+ 에이전트 지원)
"""

import numpy as np
from .config import (
    Config, NUM_REGIONS, NUM_INDUSTRIES,
    REGION_INDUSTRY_MIX, INDUSTRY_INCOME_MULTIPLIER,
    INDUSTRY_GDP_SENSITIVITY, INDUSTRY_BASE_UNEMPLOYMENT,
    REGION_INCOME_PREMIUM, REGION_JOB_DENSITY,
)


class JobMarket:
    """지역별 일자리 시장 모델 (벡터화)"""

    def __init__(self, config: Config):
        self.config = config
        self.job_cfg = config.job_market

        # 지역별 실업률 (동적, 매 스텝 갱신)
        self.regional_unemployment_rate = np.zeros(NUM_REGIONS, dtype=np.float32)
        # 지역별 평균 소득 (동적)
        self.regional_avg_income = np.zeros(NUM_REGIONS, dtype=np.float32)
        # 동적 일자리 밀도 (기존 정적 REGION_JOB_DENSITY 대체)
        self.dynamic_job_density = REGION_JOB_DENSITY.copy()
        # 산업별 현재 실업률 (GDP에 따라 변동)
        self.industry_unemployment_rate = INDUSTRY_BASE_UNEMPLOYMENT.copy()
        # 통계 기록
        self.history = []

    def initialize(self, households, rng: np.random.Generator):
        """초기화: 에이전트별 산업 배정 및 소득 재계산 (벡터화)"""
        regions = households.region.to_numpy()
        n = households.n

        # 1. 산업 배정 (지역별 산업 구성 비율에 따라)
        # 지역별 누적확률 테이블 → 균일 난수로 한번에 배정
        cum_probs = np.cumsum(REGION_INDUSTRY_MIX, axis=1)  # (13, 5)
        rolls = rng.random(n).astype(np.float32)
        region_cum = cum_probs[regions]  # (n, 5)
        industries = np.sum(rolls[:, None] >= region_cum, axis=1).astype(np.int32)
        industries = np.clip(industries, 0, NUM_INDUSTRIES - 1)

        # 2. 지역×산업 기반 소득 재배정
        ac = self.config.agent_composition
        base_incomes = rng.lognormal(
            mean=np.log(ac.income_median), sigma=ac.income_sigma, size=n
        ).astype(np.float32)

        industry_mult = INDUSTRY_INCOME_MULTIPLIER[industries]
        region_premium = REGION_INCOME_PREMIUM[regions]
        incomes = np.clip(base_incomes * industry_mult * region_premium, 100.0, 15000.0)

        # 3. 초기 고용 상태 설정 (산업별 기본 실업률 적용)
        employment_status = np.zeros(n, dtype=np.int32)
        unemployment_months = np.zeros(n, dtype=np.int32)
        previous_incomes = np.zeros(n, dtype=np.float32)

        base_unemp_per_agent = INDUSTRY_BASE_UNEMPLOYMENT[industries]
        is_unemployed = rng.random(n) < base_unemp_per_agent
        employment_status[is_unemployed] = 2  # 실업급여 수령중
        unemployment_months[is_unemployed] = rng.integers(0, 6, size=np.sum(is_unemployed))
        previous_incomes[is_unemployed] = incomes[is_unemployed]
        incomes[is_unemployed] *= self.job_cfg.unemployment_insurance_rate

        # 4. Taichi 필드에 기록
        households.income.from_numpy(incomes)
        households.industry.from_numpy(industries)
        households.employment_status.from_numpy(employment_status)
        households.unemployment_months.from_numpy(unemployment_months)
        households.previous_income.from_numpy(previous_incomes)
        households.forced_sale_countdown.from_numpy(np.full(n, -1, dtype=np.int32))
        households.housing_cost_unpaid.from_numpy(np.zeros(n, dtype=np.int32))

        # 5. 지역별 지표 초기 계산
        self._recalculate_regional_metrics(households)

    def step(self, households, gdp_growth: float, rng: np.random.Generator):
        """월간 일자리 시장 업데이트"""
        self._update_industry_unemployment(gdp_growth)
        self._update_employment_status(households, rng)
        self._update_incomes(households, gdp_growth)
        self._recalculate_regional_metrics(households)
        self.history.append(self.get_state_dict())

    def _update_industry_unemployment(self, gdp_growth: float):
        """GDP → 산업별 실업률 조정 (벡터화, 5개 산업이라 이미 빠름)"""
        gdp_deviation = gdp_growth - 0.025
        unemployment_change = -INDUSTRY_GDP_SENSITIVITY * gdp_deviation * 0.5
        mean_reversion = (INDUSTRY_BASE_UNEMPLOYMENT - self.industry_unemployment_rate) * 0.1
        self.industry_unemployment_rate = np.clip(
            self.industry_unemployment_rate + unemployment_change + mean_reversion,
            0.005, 0.25
        )

    def _update_employment_status(self, households, rng: np.random.Generator):
        """에이전트별 고용 상태 전환 (벡터화)

        취업(0) → 실업급여(2): 산업별 실업률에 비례한 확률로 실직
        실업급여(2) → 실업(1): 급여 기간(6개월) 만료
        실업(1 or 2) → 취업(0): 재취업 확률 (나이, 실업 기간에 반비례)
        """
        n = households.n
        emp = households.employment_status.to_numpy()
        ind = households.industry.to_numpy()
        inc = households.income.to_numpy()
        prev_inc = households.previous_income.to_numpy()
        unemp_m = households.unemployment_months.to_numpy()
        ages = households.age.to_numpy()
        regions = households.region.to_numpy()

        rolls = rng.random(n)
        cfg = self.job_cfg

        # ★ 순서 중요: 먼저 기존 실업자 재취업 처리 → 그 다음 새 실직 처리
        # (동일 스텝에서 실직 → 즉시 재취업 방지)

        # ── 1단계: 무급 실업(1) → 재취업 ──
        unpaid_unemp = emp == 1
        unemp_m[unpaid_unemp] += 1

        reemploy_prob_unpaid = self._calc_reemployment_prob_vec(
            ages, unemp_m, regions, ind
        ) * 0.8  # 장기 실업자 페널티
        rolls_unpaid = rng.random(n)
        reemployed_unpaid = unpaid_unemp & (rolls_unpaid < reemploy_prob_unpaid)

        income_ratio_unpaid = 0.70 + rng.random(n) * 0.20
        emp[reemployed_unpaid] = 0
        inc[reemployed_unpaid] = prev_inc[reemployed_unpaid] * income_ratio_unpaid[reemployed_unpaid]
        unemp_m[reemployed_unpaid] = 0
        prev_inc[reemployed_unpaid] = 0.0

        # ── 2단계: 실업급여 수령중(2) → 재취업 or 급여만료 ──
        on_insurance = emp == 2
        unemp_m[on_insurance] += 1

        reemploy_prob_ins = self._calc_reemployment_prob_vec(
            ages, unemp_m, regions, ind
        )
        rolls_ins = rng.random(n)
        reemployed_ins = on_insurance & (rolls_ins < reemploy_prob_ins)

        income_ratio_ins = 0.80 + rng.random(n) * 0.30
        emp[reemployed_ins] = 0
        inc[reemployed_ins] = prev_inc[reemployed_ins] * income_ratio_ins[reemployed_ins]
        unemp_m[reemployed_ins] = 0
        prev_inc[reemployed_ins] = 0.0

        # 급여 만료 (재취업 실패 + 6개월 경과)
        insurance_expired = on_insurance & ~reemployed_ins & (unemp_m >= cfg.unemployment_insurance_months)
        emp[insurance_expired] = 1
        inc[insurance_expired] = 0.0

        # ── 3단계: 취업자(0) → 실직 (마지막에 처리!) ──
        employed = emp == 0
        unemp_rate_per_agent = self.industry_unemployment_rate[ind]
        base_loss = cfg.base_job_destruction_rate  # 0.015
        base_unemp_per_agent = INDUSTRY_BASE_UNEMPLOYMENT[ind]
        excess_ratio = np.clip(unemp_rate_per_agent / (base_unemp_per_agent + 1e-6), 0.5, 3.0)
        job_loss_prob = base_loss * excess_ratio * 0.35
        lose_job = employed & (rolls < job_loss_prob)

        emp[lose_job] = 2
        prev_inc[lose_job] = inc[lose_job]
        inc[lose_job] *= cfg.unemployment_insurance_rate
        unemp_m[lose_job] = 0

        households.employment_status.from_numpy(emp)
        households.income.from_numpy(inc)
        households.previous_income.from_numpy(prev_inc)
        households.unemployment_months.from_numpy(unemp_m)

    def _calc_reemployment_prob_vec(
        self, ages: np.ndarray, unemp_months: np.ndarray,
        regions: np.ndarray, industries: np.ndarray
    ) -> np.ndarray:
        """재취업 확률 (벡터화)"""
        base = self.job_cfg.reemployment_base_prob  # 0.15
        age_penalty = np.maximum(0, ages - 50) * self.job_cfg.reemployment_age_penalty
        duration_penalty = np.minimum(np.maximum(0, unemp_months - 6) * 0.005, 0.05)
        job_density_bonus = self.dynamic_job_density[regions] * 0.05
        prob = base - age_penalty - duration_penalty + job_density_bonus
        return np.clip(prob, 0.03, 0.50)

    def _update_incomes(self, households, gdp_growth: float):
        """취업자 소득 성장 (벡터화)"""
        emp = households.employment_status.to_numpy()
        ind = households.industry.to_numpy()
        inc = households.income.to_numpy()

        employed = emp == 0
        if not np.any(employed):
            return

        gdp_deviation = gdp_growth - 0.025
        sensitivity = INDUSTRY_GDP_SENSITIVITY[ind[employed]]
        monthly_growth = self.job_cfg.income_growth_employed * (1.0 + sensitivity * gdp_deviation * 2.0)
        monthly_growth = np.clip(monthly_growth, -0.01, 0.02)
        inc[employed] *= (1.0 + monthly_growth)

        households.income.from_numpy(inc)

    def check_housing_affordability(self, households, interest_rate: float):
        """주거비 납부 능력 체크 → 강제매도 트리거 (벡터화)"""
        inc = households.income.to_numpy()
        assets = households.asset.to_numpy()
        owned = households.owned_houses.to_numpy()
        unpaid = households.housing_cost_unpaid.to_numpy()
        countdown = households.forced_sale_countdown.to_numpy()
        wants_sell = households.wants_to_sell.to_numpy()
        mortgages = households.mortgage_balance.to_numpy()

        cfg = self.job_cfg
        monthly_rate = interest_rate / 12.0

        # 무주택자 리셋
        homeless = owned == 0
        unpaid[homeless] = 0
        countdown[homeless] = -1

        # 유주택자만 체크
        has_home = owned >= 1
        mortgage_payment = np.where(mortgages > 0, mortgages * monthly_rate, 0.0)
        total_cost = mortgage_payment + cfg.min_living_cost
        monthly_available = inc + assets * 0.01
        can_afford = monthly_available >= total_cost

        # 감당 불가 → 미납 증가
        cannot = has_home & ~can_afford
        unpaid[cannot] += 1
        # 미납 12개월 + 카운트다운 미시작 → 강제매도 트리거
        trigger = cannot & (unpaid >= cfg.forced_sale_months) & (countdown < 0)
        countdown[trigger] = 3

        # 감당 가능 → 미납 감소, 카운트다운 취소
        can = has_home & can_afford
        unpaid[can] = np.maximum(0, unpaid[can] - 1)
        cancel = can & (countdown > 0)
        countdown[cancel] = -1

        # 카운트다운 진행
        ticking = countdown > 0
        countdown[ticking] -= 1

        # 카운트다운 0 도달 → 강제매도 실행
        execute = countdown == 0
        wants_sell[execute] = 1
        countdown[execute] = -1
        unpaid[execute] = 0

        households.housing_cost_unpaid.from_numpy(unpaid)
        households.forced_sale_countdown.from_numpy(countdown)
        households.wants_to_sell.from_numpy(wants_sell.astype(np.int32))

    def _recalculate_regional_metrics(self, households):
        """지역별 지표 재계산 (13개 지역 루프, 벡터화 불필요)"""
        regions = households.region.to_numpy()
        incomes = households.income.to_numpy()
        emp_status = households.employment_status.to_numpy()
        industries = households.industry.to_numpy()

        for r in range(NUM_REGIONS):
            mask = regions == r
            total = np.sum(mask)
            if total == 0:
                continue

            # 지역 실업률
            emp_r = emp_status[mask]
            unemployed = np.sum((emp_r == 1) | (emp_r == 2))
            self.regional_unemployment_rate[r] = unemployed / total

            # 지역 평균 소득 (취업자만)
            employed_mask = mask & (emp_status == 0)
            n_employed = np.sum(employed_mask)
            self.regional_avg_income[r] = np.mean(incomes[employed_mask]) if n_employed > 0 else 0.0

            # 동적 일자리 밀도
            region_ind = industries[mask]
            ind_counts = np.bincount(region_ind, minlength=NUM_INDUSTRIES)
            ind_ratio = ind_counts / total
            expected_unemp = np.dot(ind_ratio, INDUSTRY_BASE_UNEMPLOYMENT)
            actual_unemp = self.regional_unemployment_rate[r]
            density_adj = np.clip(1.0 - (actual_unemp - expected_unemp) * 2.0, 0.5, 1.5)
            self.dynamic_job_density[r] = REGION_JOB_DENSITY[r] * density_adj

    def get_dynamic_job_density(self) -> np.ndarray:
        """동적 일자리 밀도 반환"""
        return self.dynamic_job_density.copy().astype(np.float32)

    def get_state_dict(self) -> dict:
        """현재 상태 딕셔너리"""
        return {
            'regional_unemployment_rate': self.regional_unemployment_rate.copy(),
            'regional_avg_income': self.regional_avg_income.copy(),
            'dynamic_job_density': self.dynamic_job_density.copy(),
            'industry_unemployment_rate': self.industry_unemployment_rate.copy(),
            'avg_unemployment_rate': float(np.mean(self.regional_unemployment_rate)),
        }

    def reset(self):
        """상태 초기화"""
        self.regional_unemployment_rate = np.zeros(NUM_REGIONS, dtype=np.float32)
        self.regional_avg_income = np.zeros(NUM_REGIONS, dtype=np.float32)
        self.dynamic_job_density = REGION_JOB_DENSITY.copy()
        self.industry_unemployment_rate = INDUSTRY_BASE_UNEMPLOYMENT.copy()
        self.history = []
