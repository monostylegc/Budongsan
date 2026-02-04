"""시뮬레이션 메인 루프"""

import taichi as ti
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import asdict

from .config import Config, PolicyConfig, NUM_REGIONS, REGIONS
from .agents import Households
from .houses import Houses
from .market import Market


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

        # 상태
        self.current_step = 0
        self.initialized = False

        # 정책 파라미터 (Taichi field)
        self.ltv_limits = ti.field(dtype=ti.f32, shape=3)

        # 통계
        self.stats_history = []

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
        """초기 소유권 매칭"""
        owned_houses = self.households.owned_houses.to_numpy()
        regions = self.households.region.to_numpy()

        house_regions = self.houses.region.to_numpy()
        house_owners = self.houses.owner_id.to_numpy()

        # 지역별로 주택 인덱스 정리
        region_houses = {r: [] for r in range(NUM_REGIONS)}
        for i, r in enumerate(house_regions):
            if house_owners[i] == -1:
                region_houses[r].append(i)

        # 가구별로 주택 할당
        for hh_id in range(len(owned_houses)):
            n_owned = owned_houses[hh_id]
            if n_owned == 0:
                continue

            region = regions[hh_id]
            available = region_houses[region]

            for _ in range(min(n_owned, len(available))):
                if not available:
                    break
                house_id = available.pop()
                house_owners[house_id] = hh_id

        self.houses.owner_id.from_numpy(house_owners.astype(np.int32))

    def _update_policy_fields(self):
        """정책 파라미터를 Taichi field로 복사"""
        policy = self.config.policy
        self.ltv_limits.from_numpy(np.array([
            policy.ltv_1house,
            policy.ltv_2house,
            policy.ltv_3house
        ], dtype=np.float32))

    def step(self):
        """한 스텝 실행 (1개월)"""
        if not self.initialized:
            raise RuntimeError("시뮬레이션이 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        policy = self.config.policy

        # 0. 사회적 신호 업데이트 (행동경제학 요소용)
        trans_history = self.market.transaction_history if len(self.market.transaction_history) > 0 else []
        self.households.update_social_signals(self.market, np.array(trans_history[-6:] if len(trans_history) >= 6 else trans_history))

        # 1. 기대 업데이트
        self.households.update_expectations(
            self.market.region_price_changes,
            social_weight=0.1
        )

        # 2. 매수/매도 의사결정
        self.households.decide_buy_sell(
            self.market.region_prices,
            self.ltv_limits,
            policy.dti_limit,
            policy.interest_rate + policy.mortgage_spread,
            self.config.buy_threshold,
            self.config.sell_threshold,
            policy.transfer_tax_multi_long,
            policy.jongbu_rate,
            policy.jongbu_threshold_multi
        )

        # 3. 매물 등록 (매도 희망자의 주택을 매물로)
        self._register_listings()

        # 4. 수요/공급 집계
        self.market.count_demand_supply(self.households, self.houses)

        # 5. 매칭 (거래) - 매입가 기록 포함
        self.market.enhanced_matching(self.households, self.houses, self.rng, self.current_step)

        # 6. 가격 업데이트
        self.market.update_prices(
            self.houses,
            self.config.price_sensitivity,
            self.config.expectation_weight
        )

        # 7. 가격 집계 및 이력 업데이트
        self.market.aggregate_prices(self.houses)
        self.houses.update_price_history()

        # 8. 무주택 기간 업데이트
        self.households.update_homeless_months()

        # 9. 자산 업데이트 (저축, 소득 성장)
        self.households.update_assets(
            income_growth=0.03,  # 연 3% 소득 성장
            savings_rate=0.1    # 소득의 10% 저축
        )

        # 10. 연간 업데이트 (1월에)
        if self.current_step % 12 == 0 and self.current_step > 0:
            self.houses.update_building_age()
            self.households.update_yearly_aging()

        # 11. 생애 이벤트 (결혼, 출산 등) - 매월
        self.households.update_life_events(self.rng, self.current_step)

        # 12. 이력 기록
        self.market.record_history()
        self._record_stats()

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
            "stats_history": self.stats_history,
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
        self.stats_history.clear()
        self.initialized = False
