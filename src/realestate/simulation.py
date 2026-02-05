"""시뮬레이션 메인 루프 - 행동경제학 기반 한국 부동산 ABM"""

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

        # 거시경제 및 공급 모델
        self.macro = MacroModel(self.config)
        self.supply = SupplyModel(self.config)

        # 상태
        self.current_step = 0
        self.initialized = False

        # 정책 파라미터 (Taichi field)
        self.ltv_limits = ti.field(dtype=ti.f32, shape=4)  # [무주택, 1주택, 2주택, 3주택+]
        self.acq_tax_rates = ti.field(dtype=ti.f32, shape=3)  # 취득세율 (똘똘한 한채 정책)

        # 거래 모드 설정 (True: Double Auction, False: Enhanced Matching)
        self.use_double_auction = True

        # 통계
        self.stats_history = []
        self.macro_history = []
        self.supply_history = []
        self.demolition_history = []

    def initialize(self):
        """초기화"""
        print(f"초기화 중... (가구: {self.config.num_households:,}, 주택: {self.config.num_houses:,})")

        self.households.initialize(self.rng)
        self.houses.initialize(self.rng)
        self.market.initialize()

        # 소유권 매칭 (가구와 주택 연결)
        self._match_initial_ownership()

        # 초기 가격 집계
        self.market.aggregate_prices(self.houses)

        # 정책 파라미터 설정
        self._update_policy_fields()

        self.initialized = True
        print("초기화 완료")

    def _match_initial_ownership(self):
        """초기 소유권 매칭

        수정 (2024): 다주택자가 타지역 주택을 소유할 수 있도록 개선
        - 1주택자: 거주 지역에 주택 소유
        - 다주택자: 첫 번째 주택은 거주 지역, 추가 주택은 투자 가치 높은 지역에 소유 가능
        - 투자자/투기자: 핵심 지역(강남, 마용성, 분당) 주택 소유 확률 증가
        """
        owned_houses = self.households.owned_houses.to_numpy()
        regions = self.households.region.to_numpy()
        agent_types = self.households.agent_type.to_numpy()

        house_regions = self.houses.region.to_numpy()
        house_owners = self.houses.owner_id.to_numpy()
        house_prices = self.houses.price.to_numpy()

        # 지역별로 주택 인덱스 정리
        region_houses = {r: [] for r in range(NUM_REGIONS)}
        for i, r in enumerate(house_regions):
            if house_owners[i] == -1:
                region_houses[r].append(i)

        # 투자 가치가 높은 지역 (다주택자 추가 주택 소유 대상)
        # 강남(0), 마용성(1), 분당(3), 기타서울(2) 순
        investment_regions = [0, 1, 3, 2]

        # 투자 지역 선택 확률 (투자자/투기자 vs 실수요자)
        investment_region_probs_investor = np.array([0.35, 0.25, 0.20, 0.20])  # 강남 선호
        investment_region_probs_normal = np.array([0.15, 0.20, 0.25, 0.40])    # 분산 투자

        # 가구별로 주택 할당
        for hh_id in range(len(owned_houses)):
            n_owned = owned_houses[hh_id]
            if n_owned == 0:
                continue

            home_region = regions[hh_id]
            agent_type = agent_types[hh_id]

            # 첫 번째 주택: 거주 지역
            if len(region_houses[home_region]) > 0:
                house_id = region_houses[home_region].pop()
                house_owners[house_id] = hh_id
                remaining = n_owned - 1
            else:
                remaining = n_owned

            # 추가 주택: 다주택자의 경우 타지역 투자
            for _ in range(remaining):
                target_region = None

                if agent_type in [1, 2]:  # 투자자 또는 투기자
                    # 투자 지역에서 선택 (강남 선호)
                    probs = investment_region_probs_investor
                else:
                    # 실수요자는 거주 지역 우선, 없으면 투자 지역
                    if len(region_houses[home_region]) > 0:
                        target_region = home_region
                    else:
                        probs = investment_region_probs_normal

                # 투자 지역에서 선택
                if target_region is None:
                    # 확률에 따라 투자 지역 선택
                    cumprob = 0.0
                    roll = self.rng.random()
                    for idx, inv_region in enumerate(investment_regions):
                        cumprob += probs[idx]
                        if roll < cumprob and len(region_houses[inv_region]) > 0:
                            target_region = inv_region
                            break

                    # 투자 지역에 남은 주택이 없으면 아무 지역이나
                    if target_region is None:
                        for r in range(NUM_REGIONS):
                            if len(region_houses[r]) > 0:
                                target_region = r
                                break

                # 주택 할당
                if target_region is not None and len(region_houses[target_region]) > 0:
                    house_id = region_houses[target_region].pop()
                    house_owners[house_id] = hh_id

        self.houses.owner_id.from_numpy(house_owners.astype(np.int32))

    def _update_policy_fields(self):
        """정책 파라미터를 Taichi field로 복사"""
        policy = self.config.policy
        # LTV 한도: [무주택자, 1주택자, 2주택자, 3주택자+]
        # 인덱스 = 현재 보유 주택 수
        self.ltv_limits.from_numpy(np.array([
            policy.ltv_first_time,  # 무주택자 (생애최초 70%)
            policy.ltv_1house,      # 1주택자 (50%)
            policy.ltv_2house,      # 2주택자 (30%)
            policy.ltv_3house       # 3주택자+ (0%)
        ], dtype=np.float32))

        # 취득세율 (똘똘한 한채 정책의 핵심)
        # 다주택자 취득세가 8-12%로 높아 추가 매수 억제
        # → 1주택자들이 프리미엄 지역에 집중 → 자연스러운 프리미엄 형성
        self.acq_tax_rates.from_numpy(np.array([
            policy.acq_tax_1house,   # 1주택: 1%
            policy.acq_tax_2house,   # 2주택: 8%
            policy.acq_tax_3house    # 3주택+: 12%
        ], dtype=np.float32))

    def step(self):
        """한 스텝 실행 (1개월)

        행동경제학 요소:
        - Prospect Theory (전망이론)
        - Hyperbolic Discounting (준쌍곡선 할인)
        - DeGroot Learning (네트워크 학습)
        - FOMO, 손실회피, 앵커링, 군집행동

        거시경제 연동:
        - Taylor Rule 금리
        - GDP-소득 연동
        - 전월세 전환

        내생적 공급:
        - 가격 연동 신규 공급
        - 재건축
        """
        if not self.initialized:
            raise RuntimeError("시뮬레이션이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        policy = self.config.policy
        pt_cfg = self.config.prospect_theory
        net_cfg = self.config.network

        # === Phase 0: 거시경제 업데이트 ===
        # 가격 변화율 계산
        price_changes = self.market.region_price_changes.to_numpy()
        avg_price_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0

        # Taylor Rule 금리 및 GDP 업데이트
        self.macro.step(avg_price_change, self.rng)
        current_mortgage_rate = self.macro.get_mortgage_rate()
        income_growth = self.macro.get_income_growth()

        # === Phase 1: 사회적 신호 및 네트워크 업데이트 ===
        trans_history = self.market.transaction_history if len(self.market.transaction_history) > 0 else []
        self.households.update_social_signals(
            self.market,
            np.array(trans_history[-6:] if len(trans_history) >= 6 else trans_history)
        )

        # DeGroot Learning (네트워크 기반 신념 업데이트)
        self.households.update_network_beliefs()

        # === Phase 1.5: 가격 적정성 지표 업데이트 (구조적 개선) ===
        self.market.update_price_metrics(self.households)

        # === Phase 1.6: 지역 선택 (구조적 개선) ===
        # 에이전트 유형별로 목표 지역 결정 (실수요자/투자자/투기자)
        self.households.select_target_regions(self.market, self.rng)

        # === Phase 2: 기대 업데이트 ===
        self.households.update_expectations(
            self.market.region_price_changes,
            social_weight=self.config.behavioral.social_learning_rate
        )

        # 참조 가격 업데이트 (Prospect Theory)
        self.households.update_reference_price(
            self.market.region_prices,
            pt_cfg.reference_point_decay
        )

        # === Phase 3: 매수/매도 의사결정 (Prospect Theory + Hyperbolic Discounting + DSR) ===
        # 취득세 정책이 "똘똘한 한채" 현상의 핵심 원인
        aff_cfg = policy.affordability

        # 고자산가 기준 계산 (상위 10% - 실제 한국 데이터 기준)
        # 데이터: 상위 10% 순자산 10.5억+ (전체 자산의 44.4% 점유)
        assets_np = self.households.asset.to_numpy()
        wealthy_threshold = float(np.percentile(assets_np, 90))

        self.households.decide_buy_sell(
            self.market.region_prices,
            self.ltv_limits,
            self.acq_tax_rates,  # 취득세율 (다주택 억제 정책)
            policy.dti_limit,
            current_mortgage_rate,  # 동적 금리 적용
            self.config.buy_threshold,
            self.config.sell_threshold,
            policy.transfer_tax_multi_long,
            policy.jongbu_rate,
            policy.jongbu_threshold_multi,
            # Prospect Theory 파라미터
            pt_cfg.alpha,
            pt_cfg.beta,
            pt_cfg.gamma_gain,
            pt_cfg.gamma_loss,
            # DSR 기반 affordability 파라미터 (통일 체계)
            aff_cfg.dsr_limit_end_user,
            aff_cfg.dsr_limit_investor,
            aff_cfg.dsr_limit_speculator,
            aff_cfg.normal_asset_utilization,
            aff_cfg.wealthy_asset_utilization,
            aff_cfg.homeless_asset_utilization,  # 무주택자 자산 활용률 (85%)
            aff_cfg.loan_term_years,
            1 if aff_cfg.allow_stretched_dsr else 0,
            aff_cfg.stretched_dsr_multiplier,
            wealthy_threshold
        )

        # 정보 캐스케이드 적용 (네트워크 효과)
        self.households.apply_information_cascade(
            net_cfg.cascade_threshold,
            net_cfg.cascade_multiplier
        )

        # === Phase 4: 매물 등록 ===
        self._register_listings()

        # === Phase 5: 수요/공급 집계 ===
        self.market.count_demand_supply(self.households, self.houses)

        # === Phase 6: 거래 매칭 ===
        if self.use_double_auction:
            # Double Auction 기반 거래
            self.market.double_auction_matching(
                self.households, self.houses, self.rng, self.current_step
            )
        else:
            # 기존 향상된 매칭
            self.market.enhanced_matching(
                self.households, self.houses, self.rng, self.current_step
            )

        # === Phase 7: 가격 업데이트 ===
        self.market.update_prices(
            self.houses,
            self.config.price_sensitivity,
            self.config.expectation_weight
        )

        # === Phase 8: 가격 집계 및 이력 업데이트 ===
        self.market.aggregate_prices(self.houses)
        self.houses.update_price_history()

        # === Phase 9: 전월세 전환 ===
        conversion_rate = self.macro.get_jeonse_conversion_rate()
        self.market.update_jeonse_wolse_conversion(
            self.houses, self.households, conversion_rate, self.rng
        )
        self.houses.update_jeonse_wolse(conversion_rate, self.rng)

        # === Phase 10: 공급 업데이트 ===
        if self.current_step >= 12:
            # 12개월 및 5년 가격 변화율 계산
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
                self.houses,
                price_changes_12m,
                price_history_5y,
                current_stock,
                self.current_step,
                self.rng
            )
            self.supply_history.append(supply_stats)

        # === Phase 10.5: 건물 노후화 업데이트 (매월) ===
        self.houses.update_depreciation()

        # === Phase 10.6: 멸실 처리 ===
        # 자연 멸실 (노후 건물)
        natural_demolished = self.houses.check_natural_demolition(self.rng)
        natural_stats = self.houses.process_demolitions(natural_demolished, self.households)

        # 재해 멸실 (화재, 자연재해 등)
        disaster_demolished = self.houses.check_disaster_demolition(self.rng)
        disaster_stats = self.houses.process_demolitions(disaster_demolished, self.households)

        # 멸실 통계 기록
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

        # === Phase 11: 무주택 기간 업데이트 ===
        self.households.update_homeless_months()

        # === Phase 12: 자산 업데이트 (GDP 연동 소득 성장) ===
        self.households.update_assets(
            income_growth=income_growth * 12,  # 월간 → 연간
            savings_rate=0.1
        )

        # === Phase 13: 연간 업데이트 (1월에) ===
        if self.current_step % 12 == 0 and self.current_step > 0:
            self.houses.update_building_age()
            self.households.update_yearly_aging()

        # === Phase 14: 생애 이벤트 ===
        self.households.update_life_events(self.rng, self.current_step)

        # === Phase 15: 이력 기록 ===
        self.market.record_history()
        self._record_stats()
        self.macro_history.append(self.macro.get_state_dict())

        self.current_step += 1

    def _register_listings(self):
        """매도 희망자의 주택을 매물로 등록"""
        wants_sell = self.households.wants_to_sell.to_numpy()
        owned_houses = self.households.owned_houses.to_numpy()

        house_owners = self.houses.owner_id.to_numpy()
        is_for_sale = self.houses.is_for_sale.to_numpy()

        # 현재 매물 비율 체크
        current_listing_rate = np.mean(is_for_sale)
        target_listing_rate = 0.05  # 목표 매물 비율 5%

        # 매도 희망 다주택자 중 일부 등록 (50%)
        seller_ids = np.where((wants_sell == 1) & (owned_houses >= 2))[0]
        if len(seller_ids) > 0:
            seller_ids = self.rng.choice(seller_ids, size=max(1, len(seller_ids) // 2), replace=False)

        for seller_id in seller_ids:
            owned_house_ids = np.where(house_owners == seller_id)[0]
            for house_id in owned_house_ids:
                if is_for_sale[house_id] == 0:
                    is_for_sale[house_id] = 1
                    break

        # 1주택자 갈아타기 (월 0.2%)
        one_house_owners = np.where(owned_houses == 1)[0]
        move_prob = 0.002
        movers = one_house_owners[self.rng.random(len(one_house_owners)) < move_prob]

        for seller_id in movers:
            owned_house_ids = np.where(house_owners == seller_id)[0]
            for house_id in owned_house_ids:
                if is_for_sale[house_id] == 0:
                    is_for_sale[house_id] = 1
                    break

        # 매물 비율이 목표보다 낮으면 추가 매물 등록 (시장 유동성 유지)
        if current_listing_rate < target_listing_rate:
            # 무작위 주택 추가 등록 (소유자 있는 주택 중)
            owned_not_for_sale = np.where((house_owners >= 0) & (is_for_sale == 0))[0]
            n_to_add = int(len(is_for_sale) * (target_listing_rate - current_listing_rate) * 0.3)
            if len(owned_not_for_sale) > 0 and n_to_add > 0:
                to_add = self.rng.choice(owned_not_for_sale, size=min(n_to_add, len(owned_not_for_sale)), replace=False)
                is_for_sale[to_add] = 1

        self.houses.is_for_sale.from_numpy(is_for_sale.astype(np.int32))

    def _record_stats(self):
        """통계 기록 (행동경제학 지표 포함)"""
        owned = self.households.owned_houses.to_numpy()
        assets = self.households.asset.to_numpy()
        prices = self.market.region_prices.to_numpy()

        # 기본 통계
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

        # 행동경제학 관련 통계
        wants_buy = self.households.wants_to_buy.to_numpy()
        wants_sell = self.households.wants_to_sell.to_numpy()
        life_stages = self.households.life_stage.to_numpy()

        stats["buy_demand_rate"] = float(np.mean(wants_buy))  # 매수 희망 비율
        stats["sell_supply_rate"] = float(np.mean(wants_sell))  # 매도 희망 비율

        # 생애주기별 통계
        stats["newlywed_homeowner_rate"] = float(np.mean(owned[life_stages == 1] > 0)) if np.sum(life_stages == 1) > 0 else 0.0
        stats["school_age_homeowner_rate"] = float(np.mean(owned[life_stages == 3] > 0)) if np.sum(life_stages == 3) > 0 else 0.0

        # 매물 잠김 지표 (하락장에서 매물 부족 현상)
        supply = self.market.supply.to_numpy()
        demand = self.market.demand.to_numpy()
        total_houses = self.market.total_houses.to_numpy()
        listing_rate = np.sum(supply) / np.sum(total_houses) if np.sum(total_houses) > 0 else 0.0
        stats["listing_rate"] = float(listing_rate)

        # 시장 과열 지표
        if np.sum(supply) > 0:
            demand_supply_ratio = np.sum(demand) / np.sum(supply)
        else:
            demand_supply_ratio = 0.0
        stats["demand_supply_ratio"] = float(demand_supply_ratio)

        # 건물 상태 통계
        condition_stats = self.houses.get_condition_stats()
        stats["mean_building_condition"] = condition_stats['mean_condition']
        stats["mean_building_age"] = condition_stats['mean_age']
        stats["old_buildings_30y"] = condition_stats['old_buildings_30y']
        stats["active_houses"] = condition_stats['active_count']
        stats["demolished_houses"] = condition_stats['demolished_count']

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
                print(f"  {year}년차: 강남 {stats['price_gangnam']/10000:.1f}억, "
                      f"거래 {stats['transaction_total']:,}건, "
                      f"자가율 {stats['homeowner_rate']*100:.1f}%")

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
        self.macro.reset()
        self.supply.reset()
        self.initialized = False

    def set_use_double_auction(self, enabled: bool):
        """Double Auction 거래 모드 설정

        Args:
            enabled: True면 Double Auction, False면 Enhanced Matching
        """
        self.use_double_auction = enabled
