"""노동시장 - 산업별 고용/실업/소득 (기존 jobs.py 포트)"""

import numpy as np


class LaborMarket:
    """노동시장 모델"""

    def __init__(self, cfg, world):
        """
        Args:
            cfg: LaborConfig
            world: RegionSet
        """
        self.cfg = cfg
        self.world = world
        n_regions = world.n
        n_industries = world.n_industries

        self.regional_unemployment_rate = np.zeros(n_regions, dtype=np.float32)
        self.regional_avg_income = np.zeros(n_regions, dtype=np.float32)
        self.dynamic_job_density = world.job_density.copy()

        # 산업별 기본 실업률
        # TODO: industries config에서 읽기
        self.industry_base_unemployment = np.full(n_industries, 0.04, dtype=np.float32)
        self.industry_unemployment_rate = self.industry_base_unemployment.copy()
        self.industry_gdp_sensitivity = np.ones(n_industries, dtype=np.float32)
        self.industry_income_multiplier = np.ones(n_industries, dtype=np.float32)

        self.history = []

    def initialize(self, agents, rng: np.random.Generator):
        """초기화: 산업 배정 및 소득 계산"""
        d = agents.data
        n = agents.n
        world = self.world

        # 산업 배정
        cum_probs = np.cumsum(world.industry_mix, axis=1)
        rolls = rng.random(n).astype(np.float32)
        region_cum = cum_probs[d.region]
        industries = np.sum(rolls[:, None] >= region_cum, axis=1).astype(np.int32)
        industries = np.clip(industries, 0, world.n_industries - 1)
        d.industry = industries

        # 소득 재계산
        industry_mult = self.industry_income_multiplier[industries]
        region_premium = world.income_premium[d.region]
        d.income = np.clip(d.income * industry_mult * region_premium, 100.0, 15000.0)

        # 초기 실업
        base_unemp = self.industry_base_unemployment[industries]
        is_unemployed = rng.random(n) < base_unemp
        d.employment_status[is_unemployed] = 2
        d.unemployment_months[is_unemployed] = rng.integers(0, 6, size=np.sum(is_unemployed))
        d.previous_income[is_unemployed] = d.income[is_unemployed]
        d.income[is_unemployed] *= self.cfg.unemployment_insurance_rate

        d.forced_sale_countdown = np.full(n, -1, dtype=np.int32)
        d.housing_cost_unpaid = np.zeros(n, dtype=np.int32)

        self._recalculate_metrics(agents)

    def step(self, agents, gdp_growth: float, rng: np.random.Generator):
        """월간 업데이트"""
        self._update_industry_unemployment(gdp_growth)
        self._update_employment_status(agents, rng)
        self._update_incomes(agents, gdp_growth)
        self._recalculate_metrics(agents)
        self.history.append(self.get_state_dict())

    def _update_industry_unemployment(self, gdp_growth: float):
        gdp_deviation = gdp_growth - 0.025
        change = -self.industry_gdp_sensitivity * gdp_deviation * 0.5
        reversion = (self.industry_base_unemployment - self.industry_unemployment_rate) * 0.1
        self.industry_unemployment_rate = np.clip(
            self.industry_unemployment_rate + change + reversion, 0.005, 0.25
        )

    def _update_employment_status(self, agents, rng):
        d = agents.data
        n = agents.n
        cfg = self.cfg

        # 재취업 (실업자)
        unemployed = d.employment_status != 0
        if np.any(unemployed):
            d.unemployment_months[unemployed] += 1
            reemploy_prob = np.clip(
                cfg.reemployment_base_prob - np.maximum(0, d.age[unemployed] - 50) * cfg.reemployment_age_penalty,
                0.03, 0.50
            )
            reemployed = unemployed.copy()
            reemployed[unemployed] = rng.random(np.sum(unemployed)) < reemploy_prob
            d.employment_status[reemployed] = 0
            d.income[reemployed] = d.previous_income[reemployed] * rng.uniform(0.7, 1.1, size=np.sum(reemployed))
            d.unemployment_months[reemployed] = 0

        # 실업급여 만료
        expired = (d.employment_status == 2) & (d.unemployment_months >= cfg.unemployment_insurance_months)
        d.employment_status[expired] = 1
        d.income[expired] = 0.0

        # 실직 (취업자)
        employed = d.employment_status == 0
        if np.any(employed):
            unemp_rate = self.industry_unemployment_rate[d.industry[employed]]
            job_loss_prob = cfg.base_job_destruction_rate * np.clip(unemp_rate / 0.04, 0.5, 3.0) * 0.35
            lose_job = employed.copy()
            lose_job[employed] = rng.random(np.sum(employed)) < job_loss_prob
            d.employment_status[lose_job] = 2
            d.previous_income[lose_job] = d.income[lose_job]
            d.income[lose_job] *= cfg.unemployment_insurance_rate
            d.unemployment_months[lose_job] = 0

    def _update_incomes(self, agents, gdp_growth: float):
        d = agents.data
        employed = d.employment_status == 0
        if not np.any(employed):
            return
        gdp_dev = gdp_growth - 0.025
        sensitivity = self.industry_gdp_sensitivity[d.industry[employed]]
        growth = self.cfg.income_growth_employed * (1 + sensitivity * gdp_dev * 2)
        growth = np.clip(growth, -0.01, 0.02)
        d.income[employed] *= (1 + growth)

    def check_housing_affordability(self, agents, interest_rate: float):
        """주거비 체크 -> 강제매도"""
        d = agents.data
        cfg = self.cfg
        monthly_rate = interest_rate / 12.0

        has_home = d.owned_houses >= 1
        mortgage_payment = np.where(d.mortgage_balance > 0, d.mortgage_balance * monthly_rate, 0)
        total_cost = mortgage_payment + cfg.min_living_cost
        available = d.income + agents.total_assets * 0.01
        can_afford = available >= total_cost

        cannot = has_home & ~can_afford
        d.housing_cost_unpaid[cannot] += 1
        trigger = cannot & (d.housing_cost_unpaid >= cfg.forced_sale_months) & (d.forced_sale_countdown < 0)
        d.forced_sale_countdown[trigger] = 3

        can = has_home & can_afford
        d.housing_cost_unpaid[can] = np.maximum(0, d.housing_cost_unpaid[can] - 1)
        d.forced_sale_countdown[can & (d.forced_sale_countdown > 0)] = -1

        ticking = d.forced_sale_countdown > 0
        d.forced_sale_countdown[ticking] -= 1
        execute = d.forced_sale_countdown == 0
        d.wants_to_sell[execute] = 1
        d.forced_sale_countdown[execute] = -1

    def _recalculate_metrics(self, agents):
        d = agents.data
        for r in range(self.world.n):
            mask = d.region == r
            total = np.sum(mask)
            if total == 0:
                continue
            self.regional_unemployment_rate[r] = np.mean(d.employment_status[mask] != 0)
            employed_mask = mask & (d.employment_status == 0)
            if np.any(employed_mask):
                self.regional_avg_income[r] = np.mean(d.income[employed_mask])

    def get_dynamic_job_density(self) -> np.ndarray:
        return self.dynamic_job_density.copy()

    def get_state_dict(self) -> dict:
        return {
            'regional_unemployment_rate': self.regional_unemployment_rate.copy(),
            'regional_avg_income': self.regional_avg_income.copy(),
            'dynamic_job_density': self.dynamic_job_density.copy(),
        }

    def reset(self):
        self.regional_unemployment_rate[:] = 0
        self.regional_avg_income[:] = 0
        self.dynamic_job_density = self.world.job_density.copy()
        self.industry_unemployment_rate = self.industry_base_unemployment.copy()
        self.history = []
