"""시뮬레이션 엔진 - 모든 컴포넌트 통합"""

import numpy as np
from pathlib import Path

from ..config.loader import load_scenario
from ..config.schema import ScenarioConfig
from ..geography.world import RegionSet
from ..core.events import EventBus
from ..agents.population import AgentPopulation
from ..agents.cognitive import CognitiveEngine
from ..institutions.tax import TaxSystem
from ..institutions.lending import LendingSystem
from ..institutions.labor import LaborMarket
from ..institutions.monetary import MonetaryPolicy
from ..institutions.policy_timeline import PolicyTimeline
from ..housing.stock import HousingStock
from ..housing.supply import SupplyModel
from ..markets.housing_market import HousingMarket
from ..markets.price_engine import PriceEngine
from ..markets.rental_market import RentalMarket
from ..macro.economy import MacroEconomy
from .phases import Phase, DEFAULT_PHASE_ORDER
from .recorder import Recorder


class SimulationEngine:
    """ABM 시뮬레이션 엔진"""

    def __init__(self, config: ScenarioConfig, world: RegionSet):
        self.config = config
        self.world = world
        self.rng = np.random.default_rng(config.simulation.seed)
        self.current_month = 0

        n = config.simulation.num_households
        nr = world.n

        # Event bus
        self.event_bus = EventBus()

        # Core systems
        self.agents = AgentPopulation(n, nr, max_neighbors=config.agents.network.max_neighbors)
        self.cognitive = CognitiveEngine(config.agents.cognitive)
        self.houses = HousingStock(n * 2, nr)  # 2x for supply buffer

        # Institutions
        self.tax = TaxSystem(config.institutions.tax)
        self.lending = LendingSystem(config.institutions.lending)
        self.labor = LaborMarket(config.labor, world)
        self.monetary = MonetaryPolicy(config.institutions.monetary)
        self.policy_timeline = PolicyTimeline(
            config.institutions.policy_timeline, self.event_bus
        )

        # Markets
        self.market = HousingMarket(config.market, world)
        self.price_engine = PriceEngine(config.market, world)
        self.rental = RentalMarket(config.institutions.rental, world)
        self.supply_model = SupplyModel(config.supply, world)

        # Macro
        self.macro = MacroEconomy(config.macro)

        # Recorder
        self.recorder = Recorder(nr)

        # Phase order
        self.phase_order = DEFAULT_PHASE_ORDER

        # Policy event handlers
        self._register_policy_handlers()

    def _register_policy_handlers(self):
        """정책 이벤트 핸들러 등록"""
        def handle_ltv(event):
            for key, val in event.data.items():
                for rule in self.config.institutions.lending.ltv_rules:
                    if str(rule.house_count) == str(key):
                        rule.ltv = float(val)
                self.lending._build_ltv_table()

        def handle_interest_rate(event):
            if 'rate' in event.data:
                self.config.institutions.monetary.interest_rate = float(event.data['rate'])

        self.event_bus.subscribe("policy.set_ltv", handle_ltv)
        self.event_bus.subscribe("policy.set_interest_rate", handle_interest_rate)

    @classmethod
    def from_preset(cls, preset_dir: str | Path) -> 'SimulationEngine':
        """프리셋 디렉토리에서 생성"""
        preset_dir = Path(preset_dir)
        config = load_scenario(preset_dir)
        world_path = preset_dir / config.simulation.world_file
        world = RegionSet.from_json(world_path)
        return cls(config, world)

    def initialize(self):
        """시뮬레이션 초기화"""
        cfg = self.config
        agents = self.agents
        d = agents.data
        n = agents.n
        rng = self.rng
        world = self.world

        # === 에이전트 초기화 ===
        comp = cfg.agents.composition

        # 나이 분포
        young_n = int(n * comp.age_young_ratio)
        middle_n = int(n * comp.age_middle_ratio)
        senior_n = n - young_n - middle_n
        ages = np.concatenate([
            rng.integers(25, 40, young_n),
            rng.integers(40, 55, middle_n),
            rng.integers(55, 70, senior_n),
        ])
        rng.shuffle(ages)
        d.age = ages.astype(np.int32)

        # 소득 (로그정규분포)
        d.income = np.clip(
            rng.lognormal(np.log(comp.income_median), comp.income_sigma, n),
            100, 15000
        ).astype(np.float32)

        # 지역 배치
        d.region = rng.choice(world.n, size=n, p=world.household_ratio).astype(np.int32)
        d.income *= world.income_premium[d.region]

        # 자산 (파레토분포)
        raw_assets = (comp.asset_median * (rng.pareto(comp.asset_alpha, n) + 1)).astype(np.float32)
        raw_assets = np.clip(raw_assets, 100, 5000000)

        # 심리적 회계: 자산을 3개 계좌에 배분
        d.emergency_fund = np.minimum(d.income * 6.0, raw_assets * 0.2).astype(np.float32)
        remaining = raw_assets - d.emergency_fund
        d.housing_fund = (remaining * 0.6).astype(np.float32)
        d.investment_fund = (remaining * 0.4).astype(np.float32)

        # 에이전트 유형
        d.agent_type[:] = 0  # 실수요자
        investor_n = int(n * comp.investor_ratio)
        speculator_n = int(n * comp.speculator_ratio)
        investor_ids = rng.choice(n, investor_n, replace=False)
        d.agent_type[investor_ids] = 1
        remaining_ids = np.setdiff1d(np.arange(n), investor_ids)
        if speculator_n > 0 and len(remaining_ids) > speculator_n:
            spec_ids = rng.choice(remaining_ids, speculator_n, replace=False)
            d.agent_type[spec_ids] = 2

        # 성격 특성
        pers = comp.personality
        d.risk_tolerance = rng.beta(pers.risk_tolerance_alpha, pers.risk_tolerance_beta, n).astype(np.float32)
        d.patience = np.clip(rng.normal(pers.patience_mean, pers.patience_std, n), 0, 1).astype(np.float32)
        d.social_conformity = rng.beta(pers.social_conformity_alpha, pers.social_conformity_beta, n).astype(np.float32)
        d.analytical_tendency = np.clip(rng.normal(pers.analytical_tendency_mean, pers.analytical_tendency_std, n), 0, 1).astype(np.float32)
        d.fomo_sensitivity = rng.beta(pers.fomo_sensitivity_alpha, pers.fomo_sensitivity_beta, n).astype(np.float32)
        d.loss_aversion = np.clip(rng.normal(cfg.agents.behavioral.loss_aversion_mean, cfg.agents.behavioral.loss_aversion_std, n), 1.5, 3.5).astype(np.float32)
        d.status_quo_bias = np.clip(rng.normal(cfg.agents.cognitive.thinking.status_quo_bias_mean, cfg.agents.cognitive.thinking.status_quo_bias_std, n), 0, 1).astype(np.float32)

        # 전망이론 파라미터
        disc = cfg.agents.discounting
        d.discount_beta = np.clip(rng.normal(disc.beta_mean, disc.beta_std, n), 0.3, 1.0).astype(np.float32)
        d.discount_delta = np.clip(rng.normal(disc.delta_mean, disc.delta_std, n), 0.9, 1.0).astype(np.float32)

        # 주택 보유 초기화
        homeless_n = int(n * comp.initial_homeless_rate)
        oneowner_n = int(n * comp.initial_one_house_rate)
        multi_n = n - homeless_n - oneowner_n

        indices = np.arange(n)
        rng.shuffle(indices)
        homeless_ids = indices[:homeless_n]
        oneowner_ids = indices[homeless_n:homeless_n + oneowner_n]
        multi_ids = indices[homeless_n + oneowner_n:]

        d.owned_houses[homeless_ids] = 0
        d.owned_houses[oneowner_ids] = 1
        d.owned_houses[multi_ids] = rng.integers(2, 5, len(multi_ids)).astype(np.int32)

        d.homeless_months[homeless_ids] = rng.integers(0, 60, len(homeless_ids)).astype(np.int32)

        # 소셜 네트워크 초기화 (같은 지역 내 랜덤 연결)
        max_nb = agents.max_neighbors
        avg_nb = cfg.agents.network.avg_neighbors
        # 지역별 에이전트 인덱스 캐시
        region_agents = {}
        for r in range(world.n):
            region_agents[r] = np.where(d.region == r)[0]
        for i in range(n):
            pool = region_agents[d.region[i]]
            pool = pool[pool != i]
            nb_count = min(avg_nb, len(pool))
            if nb_count > 0:
                chosen = rng.choice(pool, nb_count, replace=False)
                d.neighbors[i, :nb_count] = chosen
                d.num_neighbors[i] = nb_count

        # 부모 지원 초기화
        aff = cfg.institutions.affordability
        eligible = (d.age <= aff.parent_support_age_max) & (d.owned_houses == 0)
        gets_support = eligible & (rng.random(n) < aff.parent_support_rate)
        support = np.clip(
            rng.normal(aff.parent_support_mean, aff.parent_support_std, n),
            0, aff.parent_support_mean * 3
        ).astype(np.float32)
        d.parent_support[gets_support] = support[gets_support]
        d.housing_fund[gets_support] += d.parent_support[gets_support]

        # === 주택 초기화 ===
        self.houses.initialize(self.world, rng)

        # 주택-에이전트 매칭
        self._match_initial_ownership()

        # === 노동시장 초기화 ===
        self.labor.initialize(agents, rng)

        # === 시장 초기화 ===
        self.market.aggregate_prices(self.houses)

        # 에이전트 인지 초기화 (알려진 가격 = 시장 가격 + 노이즈)
        noise = rng.normal(0, 0.05, (n, world.n)).astype(np.float32)
        d.known_prices = np.maximum(self.market.region_prices[None, :] * (1 + noise), 0)
        d.price_info_age = rng.integers(0, 6, (n, world.n)).astype(np.int32)
        d.info_quality = np.clip(rng.beta(2, 5, n), 0, 1).astype(np.float32)

        # 참조 가격
        d.reference_price = self.market.region_prices[d.region]

        print(f"  Initialized: {n:,} agents, {world.n} regions, {self.houses.n:,} housing slots")

    def _match_initial_ownership(self):
        """초기 주택-에이전트 매칭"""
        d = self.agents.data
        houses = self.houses

        owners = np.where(d.owned_houses >= 1)[0]
        available = np.where((houses.owner_id == -1) & (houses.is_active == 1))[0]

        self.rng.shuffle(owners)
        self.rng.shuffle(available)

        hi = 0
        for agent_id in owners:
            if hi >= len(available):
                break
            n_owned = d.owned_houses[agent_id]
            for _ in range(n_owned):
                if hi >= len(available):
                    break
                house_id = available[hi]
                # 같은 지역 우선
                region_match = np.where(
                    (houses.region[available[hi:]] == d.region[agent_id]) &
                    (houses.owner_id[available[hi:]] == -1)
                )[0]
                if len(region_match) > 0:
                    house_id = available[hi + region_match[0]]
                houses.owner_id[house_id] = agent_id
                if d.primary_house_id[agent_id] == -1:
                    d.primary_house_id[agent_id] = house_id
                    d.purchase_price[agent_id] = houses.price[house_id]
                d.total_purchase_price[agent_id] += houses.price[house_id]
                hi += 1

    def step(self):
        """한 달 시뮬레이션"""
        for phase in self.phase_order:
            self._execute_phase(phase)
        self.current_month += 1

    def _execute_phase(self, phase: Phase):
        """개별 페이즈 실행"""
        if phase == Phase.POLICY_CHECK:
            self.policy_timeline.check(self.current_month)

        elif phase == Phase.MACRO_UPDATE:
            avg_price_change = float(np.mean(self.market.region_price_changes))
            self.macro.step(avg_price_change, self.rng)

        elif phase == Phase.MONETARY_POLICY:
            self.monetary.step(self.macro.state.inflation, self.macro.state.output_gap)

        elif phase == Phase.LABOR_MARKET:
            self.labor.step(self.agents, self.macro.state.gdp_growth, self.rng)

        elif phase == Phase.INCOME_DISTRIBUTION:
            self.agents.distribute_income(self.config.agents.cognitive.thinking)

        elif phase == Phase.COGNITIVE_PIPELINE:
            self.cognitive.step(
                self.agents,
                self.world,
                self.market.region_prices,
                self.market.region_price_changes,
                self.monetary.get_mortgage_rate(),
                self.rng,
            )

        elif phase == Phase.PRICE_AGGREGATION:
            self.market.aggregate_prices(self.houses)

        elif phase == Phase.DEMAND_SUPPLY:
            self.market.count_demand_supply(self.agents, self.houses)

        elif phase == Phase.MARKET_MATCHING:
            # 매도 의향 → 실제 매물 등록
            self._list_houses_for_sale()
            # 대출/세금 시스템 + 금리 주입
            self.market.lending = self.lending
            self.market.tax_system = self.tax
            self.market.mortgage_rate = self.monetary.get_mortgage_rate()
            self.market.simple_matching(self.agents, self.houses, self.rng, self.current_month)

        elif phase == Phase.TAX_SETTLEMENT:
            # 월간 종부세
            d = self.agents.data
            house_values = d.total_purchase_price  # 간이 계산
            self.tax.deduct_monthly_jongbu(self.agents, house_values)

        elif phase == Phase.RENTAL_UPDATE:
            conv_rate = self.monetary.get_jeonse_conversion_rate()
            self.rental.update_conversion(self.houses, conv_rate, self.rng)

        elif phase == Phase.SUPPLY_UPDATE:
            price_changes = self.market.region_price_changes
            stock = self.market.total_houses
            self.supply_model.step(self.houses, price_changes, stock, self.current_month, self.rng)

        elif phase == Phase.DEPRECIATION:
            self.houses.update_depreciation(
                self.config.supply.depreciation_rate,
                self.config.supply.min_condition,
            )

        elif phase == Phase.HOUSING_AFFORDABILITY:
            self.labor.check_housing_affordability(self.agents, self.monetary.get_mortgage_rate())

        elif phase == Phase.LIFECYCLE_UPDATE:
            self._update_lifecycle()

        elif phase == Phase.PRICE_UPDATE:
            changes = self.price_engine.update_prices(
                self.houses,
                self.market.demand,
                self.market.supply,
                self.market.total_houses,
                self.market.region_price_changes,
            )
            self.market.region_price_changes = changes

        elif phase == Phase.RECORD_STATS:
            self.recorder.record(
                self.current_month, self.agents, self.houses,
                self.market, self.macro, self.monetary, self.labor,
            )
            self.market.record_history()

        elif phase == Phase.EVENT_PROCESS:
            self.event_bus.process()

    def _list_houses_for_sale(self):
        """매도 의향 에이전트의 주택을 매물로 등록 + 일부 자연 매물 추가"""
        d = self.agents.data
        houses = self.houses

        # 1. wants_to_sell 에이전트의 주택 등록
        sellers = np.where(d.wants_to_sell == 1)[0]
        for agent_id in sellers:
            owned = np.where(houses.owner_id == agent_id)[0]
            if len(owned) > 0:
                # 가장 비싼 주택 우선 매도 (투자 목적)
                if d.owned_houses[agent_id] > 1:
                    sell_idx = owned[np.argmax(houses.price[owned])]
                else:
                    sell_idx = owned[0]
                houses.is_for_sale[sell_idx] = 1

        # 2. 자연 매물: 소유자 중 일부가 자연스럽게 매물 등록 (이직, 이사 등)
        owned_houses = np.where((houses.owner_id >= 0) & (houses.is_active == 1) & (houses.is_for_sale == 0))[0]
        if len(owned_houses) > 0:
            natural_rate = 0.005  # 월간 0.5% 자연 매물
            natural_sell = self.rng.random(len(owned_houses)) < natural_rate
            houses.is_for_sale[owned_houses[natural_sell]] = 1

    def _update_lifecycle(self):
        """생애주기 업데이트"""
        d = self.agents.data
        rng = self.rng
        cfg = self.config.agents.lifecycle

        # 매년 나이 증가 (12개월마다)
        if self.current_month % 12 == 0:
            d.age += 1

        # 결혼
        unmarried = (d.is_married == 0) & (d.age >= cfg.marriage_urgency_age_start) & (d.age <= cfg.marriage_urgency_age_end)
        marry_prob = 0.01 * np.ones(self.agents.n, dtype=np.float32)
        marry_prob[unmarried] *= cfg.newlywed_housing_pressure
        new_married = unmarried & (rng.random(self.agents.n) < marry_prob)
        d.is_married[new_married] = 1
        d.life_stage[new_married] = 1  # NEWLYWED

        # 출산 (결혼 후)
        married_no_child = (d.is_married == 1) & (d.num_children == 0) & (d.age < 45)
        birth_prob = 0.02
        new_parent = married_no_child & (rng.random(self.agents.n) < birth_prob)
        d.num_children[new_parent] += 1
        d.eldest_child_age[new_parent] = 0
        d.life_stage[new_parent] = 2  # PARENTING

        # 자녀 나이 증가
        has_children = d.num_children > 0
        if self.current_month % 12 == 0:
            d.eldest_child_age[has_children] += 1

        # 학령기
        school_age = (d.eldest_child_age >= cfg.school_transition_age_start) & (d.eldest_child_age <= cfg.school_transition_age_end)
        d.life_stage[school_age] = 3  # SCHOOL_AGE

        # 은퇴
        retired = d.age >= cfg.retirement_start_age
        d.life_stage[retired] = 5  # RETIRED

        # 무주택 기간 업데이트
        d.homeless_months[d.owned_houses == 0] += 1
        d.homeless_months[d.owned_houses > 0] = 0

    def run(self, n_steps: int = None, progress: bool = True) -> dict:
        """시뮬레이션 실행

        Args:
            n_steps: 실행할 스텝 수 (None이면 config 기준)
            progress: 진행 상황 출력
        """
        if n_steps is None:
            n_steps = self.config.simulation.num_steps

        if progress:
            print(f"Starting simulation: {n_steps} months, {self.agents.n:,} agents, {self.world.n} regions")

        self.initialize()

        for step in range(n_steps):
            self.step()
            if progress and (step + 1) % 6 == 0:
                prices = self.market.region_prices
                avg_price = float(np.mean(prices[prices > 0]))
                homeless = float(np.mean(self.agents.data.owned_houses == 0))
                txn = int(self.market.transactions.sum())
                print(f"  Month {step+1:3d}/{n_steps}: avg_price={avg_price:,.0f} homeless={homeless:.1%} txn={txn}")

        if progress:
            print(f"\nSimulation complete. {n_steps} months elapsed.")
            summary = self.recorder.get_summary()
            if 'price_changes_pct' in summary:
                print("\n  Price changes:")
                names = self.world.get_names()
                for i, name in enumerate(names):
                    if i in summary['price_changes_pct']:
                        print(f"    {name}: {summary['price_changes_pct'][i]:+.2f}%")
            print(f"  Total transactions: {summary.get('total_transactions', 0):,}")
            print(f"  Final homeless rate: {summary.get('final_homeless_rate', 0):.1%}")

        return self.recorder.get_summary()

    def reset(self):
        """초기화"""
        self.current_month = 0
        self.rng = np.random.default_rng(self.config.simulation.seed)
        self.labor.reset()
        self.monetary.reset()
        self.rental.reset()
        self.supply_model.reset()
        self.macro.reset()
        self.recorder.reset()
