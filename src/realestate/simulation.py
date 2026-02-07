"""시뮬레이션 메인 루프 - 행동경제학 기반 한국 부동산 ABM

모듈 구조 (3-모듈 분리):
- 환경(Environment): 정책(Policy), 공급(Supply), 거시경제(Macro)
- 일자리(Jobs): 산업별 고용/실업, 소득 분배, 강제매도 트리거
- 에이전트(Agent): 감정, 자산, 연령, 소득, 의사결정
"""

import taichi as ti
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import asdict

from .config import Config, PolicyConfig, NUM_REGIONS, REGIONS
from .agents import Households
from .houses import Houses
from .market import Market
from .macro import MacroModel
from .supply import SupplyModel
from .jobs import JobMarket


class Simulation:
    """ABM 시뮬레이션"""

    def __init__(self, config: Optional[Config] = None, arch: str = "vulkan"):
        """
        Args:
            config: 시뮬레이션 설정
            arch: Taichi 백엔드 ("vulkan", "cuda", "cpu")
        """
        # Taichi 초기화
        arch_map = {
            "vulkan": ti.vulkan,
            "cuda": ti.cuda,
            "cpu": ti.cpu,
            "gpu": ti.gpu,
        }
        ti.init(arch=arch_map.get(arch, ti.vulkan))

        self.config = config or Config()
        self.rng = np.random.default_rng(self.config.seed)

        # 컴포넌트 생성
        self.households = Households(self.config)
        self.houses = Houses(self.config)
        self.market = Market(self.config)

        # 거시경제, 공급, 일자리 모델
        self.macro = MacroModel(self.config)
        self.supply = SupplyModel(self.config)
        self.job_market = JobMarket(self.config)

        # 상태
        self.current_step = 0
        self.initialized = False

        # 정책 파라미터 (Taichi field)
        self.ltv_limits = ti.field(dtype=ti.f32, shape=4)
        self.acq_tax_rates = ti.field(dtype=ti.f32, shape=3)

        # 거래 모드 설정
        self.use_double_auction = True

        # 통계
        self.stats_history = []
        self.macro_history = []
        self.supply_history = []
        self.demolition_history = []
        self.job_history = []

    def initialize(self):
        """초기화"""
        print(f"초기화 중... (가구: {self.config.num_households:,}, 주택: {self.config.num_houses:,})")

        self.households.initialize(self.rng)
        self.houses.initialize(self.rng)
        self.market.initialize()

        # 소유권 매칭 (가구와 주택 연결)
        self._match_initial_ownership()

        # ★ JobMarket 초기화 (에이전트 산업 배정 + 소득 재계산)
        self.job_market.initialize(self.households, self.rng)

        # 초기 가격 집계
        self.market.aggregate_prices(self.houses)

        # 정책 파라미터 설정
        self._update_policy_fields()

        self.initialized = True
        print("초기화 완료")

    def _match_initial_ownership(self):
        """초기 소유권 매칭"""
        owned_houses = self.households.owned_houses.to_numpy()
        regions = self.households.region.to_numpy()
        agent_types = self.households.agent_type.to_numpy()

        house_regions = self.houses.region.to_numpy()
        house_owners = self.houses.owner_id.to_numpy()

        region_houses = {r: [] for r in range(NUM_REGIONS)}
        for i, r in enumerate(house_regions):
            if house_owners[i] == -1:
                region_houses[r].append(i)

        investment_regions = [0, 1, 3, 2]
        investment_region_probs_investor = np.array([0.35, 0.25, 0.20, 0.20])
        investment_region_probs_normal = np.array([0.15, 0.20, 0.25, 0.40])

        for hh_id in range(len(owned_houses)):
            n_owned = owned_houses[hh_id]
            if n_owned == 0:
                continue

            home_region = regions[hh_id]
            agent_type = agent_types[hh_id]

            if len(region_houses[home_region]) > 0:
                house_id = region_houses[home_region].pop()
                house_owners[house_id] = hh_id
                remaining = n_owned - 1
            else:
                remaining = n_owned

            for _ in range(remaining):
                target_region = None

                if agent_type in [1, 2]:
                    probs = investment_region_probs_investor
                else:
                    if len(region_houses[home_region]) > 0:
                        target_region = home_region
                    else:
                        probs = investment_region_probs_normal

                if target_region is None:
                    cumprob = 0.0
                    roll = self.rng.random()
                    for idx, inv_region in enumerate(investment_regions):
                        cumprob += probs[idx]
                        if roll < cumprob and len(region_houses[inv_region]) > 0:
                            target_region = inv_region
                            break

                    if target_region is None:
                        for r in range(NUM_REGIONS):
                            if len(region_houses[r]) > 0:
                                target_region = r
                                break

                if target_region is not None and len(region_houses[target_region]) > 0:
                    house_id = region_houses[target_region].pop()
                    house_owners[house_id] = hh_id

        self.houses.owner_id.from_numpy(house_owners.astype(np.int32))

    def _update_policy_fields(self):
        """정책 파라미터를 Taichi field로 복사"""
        policy = self.config.policy
        self.ltv_limits.from_numpy(np.array([
            policy.ltv_first_time,
            policy.ltv_1house,
            policy.ltv_2house,
            policy.ltv_3house
        ], dtype=np.float32))

        self.acq_tax_rates.from_numpy(np.array([
            policy.acq_tax_1house,
            policy.acq_tax_2house,
            policy.acq_tax_3house
        ], dtype=np.float32))

    def step(self):
        """한 스텝 실행 (1개월)

        페이즈 순서 (18단계):
        0: 거시경제 → 1: 일자리 시장 → 2: 주거비 체크 →
        3: 사회적 신호 → 4: 가격 적정성 → 5: 지역 선택 → 6: 기대 →
        7: 매수/매도 (6서브커널) → 8: 캐스케이드 → 9: 강제매도 →
        10: 매물 → 11: 수요/공급 → 12: 거래매칭 → 13: 가격 →
        14: 가격이력 → 15: 전월세 → 16: 공급/멸실 →
        17: 자산 → 18: 연간/생애/통계
        """
        if not self.initialized:
            raise RuntimeError("시뮬레이션이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        policy = self.config.policy
        pt_cfg = self.config.prospect_theory
        net_cfg = self.config.network
        aff_cfg = policy.affordability

        # === Phase 0: 거시경제 업데이트 ===
        price_changes = self.market.region_price_changes.to_numpy()
        avg_price_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0

        self.macro.step(avg_price_change, self.rng)
        current_mortgage_rate = self.macro.get_mortgage_rate()
        gdp_growth = self.macro.state.gdp_growth

        # === Phase 1: ★ 일자리 시장 업데이트 ===
        # GDP → 산업별 일자리 생성/파괴 → 고용 상태 전환 → 소득 재계산
        self.job_market.step(self.households, gdp_growth, self.rng)

        # === Phase 2: ★ 주거비 능력 체크 (강제매도 트리거) ===
        self.job_market.check_housing_affordability(
            self.households, current_mortgage_rate
        )

        # === Phase 3: 사회적 신호 및 네트워크 업데이트 ===
        trans_history = self.market.transaction_history if len(self.market.transaction_history) > 0 else []
        self.households.update_social_signals(
            self.market,
            np.array(trans_history[-6:] if len(trans_history) >= 6 else trans_history)
        )
        self.households.update_network_beliefs()

        # === Phase 4: 가격 적정성 지표 업데이트 ===
        self.market.update_price_metrics(self.households)

        # === Phase 4.5: 동적 프리미엄 업데이트 ===
        # 학군, 명성 모멘텀, 고소득 집중도를 반영한 동적 프리미엄 계산
        self.market.update_dynamic_prestige(self.households)

        # === Phase 5: 지역 선택 (동적 job_density 사용) ===
        dynamic_density = self.job_market.get_dynamic_job_density()
        self.households.select_target_regions(
            self.market, self.rng, job_density=dynamic_density
        )

        # === Phase 6: 기대 업데이트 ===
        self.households.update_expectations(
            self.market.region_price_changes,
            social_weight=self.config.behavioral.social_learning_rate
        )
        self.households.update_reference_price(
            self.market.region_prices,
            pt_cfg.reference_point_decay
        )

        # === Phase 7: 매수/매도 의사결정 (6개 독립 서브커널) ===
        assets_np = self.households.asset.to_numpy()
        wealthy_threshold = float(np.percentile(assets_np, 90))

        # 동적 job_density를 Taichi field에 복사 (compute_market_signals에서 사용)
        self.households.region_job_density.from_numpy(dynamic_density)

        # 7a: 구매력 (환경 모듈)
        self.households.compute_affordability(
            self.market.region_prices,
            self.ltv_limits,
            policy.dti_limit,
            current_mortgage_rate,
            aff_cfg.dsr_limit_end_user,
            aff_cfg.dsr_limit_investor,
            aff_cfg.dsr_limit_speculator,
            aff_cfg.normal_asset_utilization,
            aff_cfg.wealthy_asset_utilization,
            aff_cfg.homeless_asset_utilization,
            aff_cfg.loan_term_years,
            wealthy_threshold
        )

        # 7b: 생애주기 긴급도 (에이전트 모듈)
        self.households.compute_lifecycle_urgency()

        # 7c: 행동 신호 (에이전트 모듈)
        self.households.compute_behavioral_signals(
            pt_cfg.alpha,
            pt_cfg.beta,
            pt_cfg.gamma_gain,
        )

        # 7d: 시장 신호 (일자리/환경 모듈)
        self.households.compute_market_signals(wealthy_threshold)

        # 7e: 정책 페널티 (환경 모듈)
        self.households.compute_policy_penalty()

        # 7f: 최종 결정
        self.households.finalize_buy_sell_decision(
            self.market.region_prices,
            self.config.buy_threshold,
            self.config.sell_threshold,
            policy.transfer_tax_multi_long,
            policy.jongbu_rate,
            policy.jongbu_threshold_multi,
            pt_cfg.alpha,
            pt_cfg.beta,
            pt_cfg.gamma_gain,
        )

        # === Phase 8: 정보 캐스케이드 ===
        self.households.apply_information_cascade(
            net_cfg.cascade_threshold,
            net_cfg.cascade_multiplier
        )

        # === Phase 9: ★ 강제매도 처리 ===
        self._process_forced_sales()

        # === Phase 10: 매물 등록 ===
        self._register_listings()

        # === Phase 11: 수요/공급 집계 ===
        self.market.count_demand_supply(self.households, self.houses)

        # === Phase 12: 거래 매칭 ===
        if self.use_double_auction:
            self.market.double_auction_matching(
                self.households, self.houses, self.rng, self.current_step
            )
        else:
            self.market.enhanced_matching(
                self.households, self.houses, self.rng, self.current_step
            )

        # === Phase 13: 가격 업데이트 ===
        self.market.update_prices(
            self.houses,
            self.config.price_sensitivity,
            self.config.expectation_weight
        )

        # === Phase 14: 가격 집계 및 이력 ===
        self.market.aggregate_prices(self.houses)
        self.houses.update_price_history()

        # === Phase 15: 전월세 전환 ===
        conversion_rate = self.macro.get_jeonse_conversion_rate()
        self.market.update_jeonse_wolse_conversion(
            self.houses, self.households, conversion_rate, self.rng
        )
        self.houses.update_jeonse_wolse(conversion_rate, self.rng)

        # === Phase 16: 공급/감가상각/멸실 ===
        if self.current_step >= 12:
            price_history = self.market.price_history
            if len(price_history) >= 12:
                prices_12m_ago = price_history[-12]
                prices_now = self.market.region_prices.to_numpy()
                price_changes_12m = (prices_now - prices_12m_ago) / (prices_12m_ago + 1e-6)
            else:
                price_changes_12m = np.zeros(NUM_REGIONS, dtype=np.float32)

            if len(price_history) >= 60:
                prices_5y_ago = price_history[-60]
                price_history_5y = (prices_now - prices_5y_ago) / (prices_5y_ago + 1e-6)
            else:
                price_history_5y = np.zeros(NUM_REGIONS, dtype=np.float32)

            current_stock = self.houses.get_active_count_by_region()
            supply_stats = self.supply.step(
                self.houses, price_changes_12m, price_history_5y,
                current_stock, self.current_step, self.rng
            )
            self.supply_history.append(supply_stats)

        self.houses.update_depreciation()

        natural_demolished = self.houses.check_natural_demolition(self.rng)
        natural_stats = self.houses.process_demolitions(natural_demolished, self.households)
        disaster_demolished = self.houses.check_disaster_demolition(self.rng)
        disaster_stats = self.houses.process_demolitions(disaster_demolished, self.households)

        demolition_stats = {
            'step': self.current_step,
            'natural_count': natural_stats['count'],
            'disaster_count': disaster_stats['count'],
            'total_count': natural_stats['count'] + disaster_stats['count'],
            'owners_affected': natural_stats['owners_affected'] + disaster_stats['owners_affected'],
            'tenants_affected': natural_stats['tenants_affected'] + disaster_stats['tenants_affected'],
        }
        demolition_stats.update(self.houses.get_condition_stats())
        self.demolition_history.append(demolition_stats)

        # === Phase 17: 무주택기간 + 자산 업데이트 ===
        self.households.update_homeless_months()
        # 소득 성장은 Phase 1 (JobMarket)에서 처리, 여기서는 저축/소진만
        self.households.update_assets(
            savings_rate=0.1,
            min_living_cost=self.config.job_market.min_living_cost,
            mortgage_rate_monthly=current_mortgage_rate / 12.0
        )

        # === Phase 18: 연간/생애/통계 ===
        if self.current_step % 12 == 0 and self.current_step > 0:
            self.houses.update_building_age()
            self.households.update_yearly_aging()

        self.households.update_life_events(self.rng, self.current_step)

        self.market.record_history()
        self._record_stats()
        self.macro_history.append(self.macro.get_state_dict())

        # ★ 일자리 통계 기록
        if self.job_market.history:
            self.job_history.append(self.job_market.history[-1])

        self.current_step += 1

    def _process_forced_sales(self):
        """강제매도 처리 (자산 소진 에이전트)

        JobMarket.check_housing_affordability에서 설정한
        forced_sale_countdown이 0에 도달한 에이전트를 강제 매도
        """
        countdown = self.households.forced_sale_countdown.to_numpy()
        owned = self.households.owned_houses.to_numpy()
        wants_sell = self.households.wants_to_sell.to_numpy()

        # 카운트다운 0 + 주택 보유 = 강제 매도
        forced_sellers = np.where((countdown == 0) & (owned >= 1))[0]
        if len(forced_sellers) > 0:
            wants_sell[forced_sellers] = 1
            self.households.wants_to_sell.from_numpy(wants_sell.astype(np.int32))

    def _register_listings(self):
        """매도 희망자의 주택을 매물로 등록"""
        wants_sell = self.households.wants_to_sell.to_numpy()
        owned_houses = self.households.owned_houses.to_numpy()

        house_owners = self.houses.owner_id.to_numpy()
        is_for_sale = self.houses.is_for_sale.to_numpy()

        current_listing_rate = np.mean(is_for_sale)
        target_listing_rate = 0.05

        seller_ids = np.where((wants_sell == 1) & (owned_houses >= 2))[0]
        if len(seller_ids) > 0:
            seller_ids = self.rng.choice(seller_ids, size=max(1, len(seller_ids) // 2), replace=False)

        for seller_id in seller_ids:
            owned_house_ids = np.where(house_owners == seller_id)[0]
            for house_id in owned_house_ids:
                if is_for_sale[house_id] == 0:
                    is_for_sale[house_id] = 1
                    break

        one_house_owners = np.where(owned_houses == 1)[0]
        move_prob = 0.002
        movers = one_house_owners[self.rng.random(len(one_house_owners)) < move_prob]

        for seller_id in movers:
            owned_house_ids = np.where(house_owners == seller_id)[0]
            for house_id in owned_house_ids:
                if is_for_sale[house_id] == 0:
                    is_for_sale[house_id] = 1
                    break

        if current_listing_rate < target_listing_rate:
            owned_not_for_sale = np.where((house_owners >= 0) & (is_for_sale == 0))[0]
            n_to_add = int(len(is_for_sale) * (target_listing_rate - current_listing_rate) * 0.3)
            if len(owned_not_for_sale) > 0 and n_to_add > 0:
                to_add = self.rng.choice(owned_not_for_sale, size=min(n_to_add, len(owned_not_for_sale)), replace=False)
                is_for_sale[to_add] = 1

        self.houses.is_for_sale.from_numpy(is_for_sale.astype(np.int32))

    def _record_stats(self):
        """통계 기록 (고용/실업 지표 포함)"""
        owned = self.households.owned_houses.to_numpy()
        assets = self.households.asset.to_numpy()
        prices = self.market.region_prices.to_numpy()

        stats = {
            "step": self.current_step,
            "avg_price": float(np.mean(prices)),
            "price_gangnam": float(prices[0]),
            "price_gyeonggi": float(np.mean([prices[4], prices[5]])),
            "price_jibang": float(np.mean(prices[7:])),
            "transaction_total": int(np.sum(self.market.transactions.to_numpy())),
            "homeowner_rate": float(np.mean(owned > 0)),
            "multi_owner_rate": float(np.mean(owned >= 2)),
            "avg_asset": float(np.mean(assets)),
        }

        wants_buy = self.households.wants_to_buy.to_numpy()
        wants_sell = self.households.wants_to_sell.to_numpy()
        life_stages = self.households.life_stage.to_numpy()

        stats["buy_demand_rate"] = float(np.mean(wants_buy))
        stats["sell_supply_rate"] = float(np.mean(wants_sell))

        stats["newlywed_homeowner_rate"] = float(np.mean(owned[life_stages == 1] > 0)) if np.sum(life_stages == 1) > 0 else 0.0
        stats["school_age_homeowner_rate"] = float(np.mean(owned[life_stages == 3] > 0)) if np.sum(life_stages == 3) > 0 else 0.0

        supply = self.market.supply.to_numpy()
        demand = self.market.demand.to_numpy()
        total_houses = self.market.total_houses.to_numpy()
        listing_rate = np.sum(supply) / np.sum(total_houses) if np.sum(total_houses) > 0 else 0.0
        stats["listing_rate"] = float(listing_rate)

        if np.sum(supply) > 0:
            demand_supply_ratio = np.sum(demand) / np.sum(supply)
        else:
            demand_supply_ratio = 0.0
        stats["demand_supply_ratio"] = float(demand_supply_ratio)

        condition_stats = self.houses.get_condition_stats()
        stats["mean_building_condition"] = condition_stats['mean_condition']
        stats["mean_building_age"] = condition_stats['mean_age']
        stats["old_buildings_30y"] = condition_stats['old_buildings_30y']
        stats["active_houses"] = condition_stats['active_count']
        stats["demolished_houses"] = condition_stats['demolished_count']

        # ★ 고용/실업 통계 추가
        emp_status = self.households.employment_status.to_numpy()
        stats["employment_rate"] = float(np.mean(emp_status == 0))
        stats["unemployment_rate"] = float(np.mean(emp_status != 0))
        stats["avg_unemployment_rate"] = float(np.mean(self.job_market.regional_unemployment_rate))
        stats["avg_income"] = float(np.mean(self.households.income.to_numpy()))

        # 강제매도 통계
        countdown = self.households.forced_sale_countdown.to_numpy()
        stats["forced_sale_count"] = int(np.sum(countdown == 0))
        stats["at_risk_count"] = int(np.sum(countdown > 0))

        self.stats_history.append(stats)

    def run(self, steps: Optional[int] = None, verbose: bool = True):
        """시뮬레이션 실행"""
        if not self.initialized:
            self.initialize()

        steps = steps or self.config.num_steps

        if verbose:
            print(f"시뮬레이션 시작 (총 {steps}개월)")

        for i in range(steps):
            self.step()

            if verbose and (i + 1) % 12 == 0:
                year = (i + 1) // 12
                stats = self.stats_history[-1]
                unemp = stats.get('unemployment_rate', 0) * 100
                print(f"  {year}년차: 강남 {stats['price_gangnam']/10000:.1f}억, "
                      f"거래 {stats['transaction_total']:,}건, "
                      f"자가율 {stats['homeowner_rate']*100:.1f}%, "
                      f"실업률 {unemp:.1f}%")

        if verbose:
            print("시뮬레이션 완료")

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """결과 반환"""
        return {
            "config": asdict(self.config),
            "price_history": np.array(self.market.price_history),
            "transaction_history": np.array(self.market.transaction_history),
            "jeonse_ratio_history": np.array(self.market.jeonse_ratio_history),
            "bid_ask_spread_history": np.array(self.market.bid_ask_spread_history),
            "stats_history": self.stats_history,
            "macro_history": self.macro_history,
            "supply_history": self.supply_history,
            "demolition_history": self.demolition_history,
            "job_history": self.job_history,
            "regions": REGIONS,
        }

    def set_policy(self, **kwargs):
        """정책 변경"""
        for key, value in kwargs.items():
            if hasattr(self.config.policy, key):
                setattr(self.config.policy, key, value)
            else:
                raise ValueError(f"알 수 없는 정책 파라미터: {key}")
        self._update_policy_fields()

    def reset(self):
        """리셋"""
        self.current_step = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.market.price_history.clear()
        self.market.transaction_history.clear()
        self.market.jeonse_ratio_history.clear()
        self.market.bid_ask_spread_history.clear()
        self.stats_history.clear()
        self.macro_history.clear()
        self.supply_history.clear()
        self.demolition_history.clear()
        self.job_history.clear()
        self.macro.reset()
        self.supply.reset()
        self.job_market.reset()
        self.initialized = False

    def set_use_double_auction(self, enabled: bool):
        """Double Auction 거래 모드 설정"""
        self.use_double_auction = enabled
