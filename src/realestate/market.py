"""시장 메커니즘 (가격 결정, 매칭) - Double Auction 및 전월세 전환 지원"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, ADJACENCY, REGION_PRESTIGE
from .order_book import OrderBook


@ti.data_oriented
class Market:
    """부동산 시장"""

    def __init__(self, config: Config):
        self.config = config

        # 지역별 집계 데이터
        self.region_prices = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        self.region_jeonse_prices = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        self.region_price_changes = ti.field(dtype=ti.f32, shape=NUM_REGIONS)

        self.demand = ti.field(dtype=ti.i32, shape=NUM_REGIONS)
        self.supply = ti.field(dtype=ti.i32, shape=NUM_REGIONS)

        self.transactions = ti.field(dtype=ti.i32, shape=NUM_REGIONS)
        self.total_houses = ti.field(dtype=ti.i32, shape=NUM_REGIONS)

        # 가격 변화율 (업데이트용)
        self.price_change_rates = ti.field(dtype=ti.f32, shape=NUM_REGIONS)

        # 인접도 행렬 (풍선효과용)
        self.adjacency = ti.field(dtype=ti.f32, shape=(NUM_REGIONS, NUM_REGIONS))

        # 이력 저장 (시각화용)
        self.price_history = []
        self.transaction_history = []
        self.jeonse_ratio_history = []

        # === Double Auction 주문장 ===
        self.order_book = OrderBook(config)

        # === 전월세 시장 데이터 ===
        self.jeonse_wolse_ratio = ti.field(dtype=ti.f32, shape=NUM_REGIONS)  # 전세 비율
        self.conversion_rate = ti.field(dtype=ti.f32, shape=())  # 전환율
        self.wolse_history = []
        self.bid_ask_spread_history = []

        # === 가격 적정성 지표 (구조적 개선) ===
        # PIR: Price-to-Income Ratio (지역별 주택가격 / 연소득)
        self.region_pir = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        # 역사적 평균 대비 현재 가격 비율
        self.price_to_historical = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        # 기대 수익률 (가격상승률 + 임대수익률)
        self.expected_return = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        # 투자 매력도 지수 (종합)
        self.investment_attractiveness = ti.field(dtype=ti.f32, shape=NUM_REGIONS)
        # 역사적 평균 가격 (장기 이동평균)
        self.historical_avg_prices = np.zeros(NUM_REGIONS, dtype=np.float32)
        # 지역별 평균 소득 (가구 기반)
        self.region_avg_income = np.zeros(NUM_REGIONS, dtype=np.float32)

        # === 동적 프리미엄 시스템 ===
        self.dynamic_prestige = REGION_PRESTIGE.copy()  # 초기값 = 구조적 프리미엄
        self.prestige_momentum = np.zeros(NUM_REGIONS, dtype=np.float32)

    def initialize(self):
        """초기화"""
        self.adjacency.from_numpy(ADJACENCY)
        self.demand.fill(0)
        self.supply.fill(0)
        self.transactions.fill(0)

    def aggregate_prices(self, houses):
        """지역별 평균 가격 계산 (NumPy 사용)"""
        regions = houses.region.to_numpy()
        prices = houses.price.to_numpy()
        jeonse_prices = houses.jeonse_price.to_numpy()

        old_prices = self.region_prices.to_numpy()

        # 지역별 집계
        avg_prices = np.zeros(NUM_REGIONS, dtype=np.float32)
        avg_jeonse = np.zeros(NUM_REGIONS, dtype=np.float32)
        counts = np.zeros(NUM_REGIONS, dtype=np.int32)

        for r in range(NUM_REGIONS):
            mask = regions == r
            count = np.sum(mask)
            counts[r] = count
            if count > 0:
                avg_prices[r] = np.mean(prices[mask])
                avg_jeonse[r] = np.mean(jeonse_prices[mask])

        # 가격 변화율
        changes = np.zeros(NUM_REGIONS, dtype=np.float32)
        valid = old_prices > 0
        changes[valid] = (avg_prices[valid] - old_prices[valid]) / old_prices[valid]

        self.region_prices.from_numpy(avg_prices)
        self.region_jeonse_prices.from_numpy(avg_jeonse)
        self.region_price_changes.from_numpy(changes)
        self.total_houses.from_numpy(counts)

    def count_demand_supply(self, households, houses):
        """수요/공급 집계 (NumPy 사용)"""
        wants_buy = households.wants_to_buy.to_numpy()
        target_regions = households.target_region.to_numpy()
        is_for_sale = houses.is_for_sale.to_numpy()
        house_regions = houses.region.to_numpy()

        demand = np.zeros(NUM_REGIONS, dtype=np.int32)
        supply = np.zeros(NUM_REGIONS, dtype=np.int32)

        for r in range(NUM_REGIONS):
            demand[r] = np.sum((wants_buy == 1) & (target_regions == r))
            supply[r] = np.sum((is_for_sale == 1) & (house_regions == r))

        self.demand.from_numpy(demand)
        self.supply.from_numpy(supply)

    def update_prices(self, houses, sensitivity: float, expectation_weight: float):
        """가격 업데이트 (수요/공급 균형 기반 + 기대 + 풍선효과)

        수정 (2024): 기본 상승률이 항상 적용되던 문제 해결
        - 수요/공급 균형에 따라 가격 변동 방향 결정
        - 기본 상승률은 인플레이션 보상 역할로만 제한
        - 수요 < 공급 시 가격 하락 가능
        """
        demand_np = self.demand.to_numpy().astype(np.float32)
        supply_np = self.supply.to_numpy().astype(np.float32)
        total_np = self.total_houses.to_numpy().astype(np.float32)
        current_prices = self.region_prices.to_numpy()

        # 수요/공급 비율 계산 (전체 주택수 대비)
        supply_np = np.maximum(supply_np, total_np * 0.02)  # 최소 2% 매물
        ds_ratio = demand_np / supply_np

        # === 1. 수요/공급 기반 가격 변화 (핵심 로직 개선) ===
        # ds_ratio = 1.0 이면 균형, >1.0 수요초과, <1.0 공급초과
        # log 대신 선형+클리핑으로 더 직관적으로 변경
        excess_demand = ds_ratio - 1.0  # 초과수요 비율
        ds_change = sensitivity * np.clip(excess_demand, -2.0, 2.0)

        # === 2. 기대 효과 (이전 가격 변화에 따른 모멘텀) ===
        price_changes = self.region_price_changes.to_numpy()
        expectation_effect = expectation_weight * price_changes

        # === 3. 풍선효과 (지역 간 전파) ===
        adjacency = self.adjacency.to_numpy()
        spillover = self.config.spillover_rate * (adjacency @ price_changes)

        # === 4. 기본 상승률 (인플레이션 보상) - 조건부 적용 ===
        # 기본 상승률은 수요/공급 균형 상태에서만 완전 적용
        # 공급 과잉 시 상승률 축소, 수요 과잉 시 유지
        base_appreciation = self.config.base_appreciation

        # 지역별 기본 상승률 조정 (tier 기반, 프리미엄 지역 우위 반영)
        tier_multipliers = np.array([
            1.5, 1.3, 1.0,  # 서울 (강남, 마용성, 기타서울) - 상향
            1.2, 0.7, 0.5, 0.6,  # 수도권 (분당, 경기남부, 경기북부, 인천)
            0.3, 0.2, 0.2, 0.25, 0.4,  # 지방광역시 (부산, 대구, 광주, 대전, 세종) - 하향
            0.1,  # 기타지방
        ], dtype=np.float32)

        # 수요/공급 균형에 따른 기본 상승률 조정
        # ds_ratio < 0.8: 공급과잉 → 기본 상승률 축소 (but not zero)
        # ds_ratio 0.8~1.2: 균형 → 기본 상승률 비례 적용
        # ds_ratio > 1.2: 수요과잉 → 기본 상승률 100%
        appreciation_factor = np.clip((ds_ratio - 0.8) / 0.4, 0.0, 1.0)
        regional_appreciation = base_appreciation * tier_multipliers * appreciation_factor

        # ================================================================
        # 인플레이션 기대 효과 (2026-02-06 추가)
        # ================================================================
        # 현실에서 주택 가격은 인플레이션 기대로 인해 쉽게 하락하지 않음
        # 프리미엄 지역(tier 1-2)은 특히 자산 보존 목적으로 수요가 유지됨
        # 최소 상승률: 인플레이션율(약 2%/년 = 0.17%/월)의 일정 비율
        # ================================================================
        inflation_expectation = 0.0017  # 월 0.17% (연 2%)

        # tier별 인플레이션 기대 효과 (프리미엄 지역이 더 강함)
        inflation_factor = np.array([
            0.8, 0.7, 0.5,  # 서울 (강남, 마용성, 기타서울)
            0.6, 0.4, 0.3, 0.4,  # 수도권
            0.3, 0.2, 0.2, 0.2, 0.3,  # 지방광역시
            0.1,  # 기타지방
        ], dtype=np.float32)

        regional_appreciation = regional_appreciation + inflation_expectation * inflation_factor

        # === 5. 가격 수준 피드백 (역사적 자기 평균 대비 회귀) ===
        # 변경: 전체 평균 대비 → 자기 지역 역사적 평균 대비
        # 이유: 전체 평균 대비 비교는 고가 지역(강남 30억)을 구조적으로 억제하고
        #       저가 지역(대구 4.7억)을 구조적으로 증폭시켜 비현실적 결과 초래
        price_to_hist = np.divide(
            current_prices, self.historical_avg_prices,
            where=self.historical_avg_prices > 0,
            out=np.ones_like(current_prices)
        )
        # 자기 역사적 평균 대비 이탈에 강하게 반응
        # 10% 이상 올랐으면 억제 시작, 20% 이상이면 강하게 억제
        # 계수 1.5: 10% 이탈 시 조정값 0.85 (15% 억제), 20% 이탈 시 0.7 (30% 억제)
        price_adjustment = np.where(
            current_prices > 0,
            np.clip(1.15 - (price_to_hist - 1.0) * 1.5, 0.5, 1.15),
            1.0
        )

        # === 5.5. 동적 프리미엄 기반 가격 지지/억제 ===
        # 프리미엄이 높은 지역: 하방 경직성 (가격 하락 저항) → "강남 불패"
        # 프리미엄이 낮은 지역: 상방 억제 (저가 지역 과열 방지)
        # 현실 근거: 프리미엄 지역은 희소성/브랜드/학군으로 가격 유지,
        #           비프리미엄 지역은 대체재 풍부하여 수요 분산
        prestige = self.dynamic_prestige
        # 하락 시: 프리미엄 높을수록 하락 억제 (강남 85% 억제, 대구 30% 억제)
        prestige_floor_factor = prestige * 0.85
        # 상승 시: 프리미엄이 가격 상승 능력을 결정
        # 강남(1.0): ds_change 100% 반영
        # 대구(0.35): ds_change ~25% 반영 (비프리미엄 지역 상승 대폭 억제)
        # 제곱 사용: 차이를 더 극적으로 만듦 (0.35² = 0.12 vs 1.0² = 1.0)
        prestige_ceiling_factor = np.clip(prestige ** 1.5, 0.10, 1.0)
        ds_change_supported = np.where(
            ds_change < 0,
            ds_change * (1.0 - prestige_floor_factor),
            ds_change * prestige_ceiling_factor,
        )

        # === 6. 최종 가격 변화율 계산 ===
        # 수요/공급이 주요 동력, 기대와 풍선효과는 보조적
        total_change = (ds_change_supported * price_adjustment +
                       regional_appreciation +
                       expectation_effect * 0.7 +  # 기대효과 축소
                       spillover * 0.5)  # 풍선효과 축소

        # 월 최대 변화율 제한 (±1.0%, 축소하여 급등/급락 방지)
        total_change = np.clip(total_change, -0.010, 0.010)

        # 필드에 저장
        self.price_change_rates.from_numpy(total_change.astype(np.float32))

        # 개별 주택 가격 업데이트 (Taichi 커널)
        self._apply_price_changes(houses.region, houses.price, houses.jeonse_price, houses.n)

    @ti.kernel
    def _apply_price_changes(
        self,
        regions: ti.template(),
        prices: ti.template(),
        jeonse_prices: ti.template(),
        n: ti.i32
    ):
        for i in range(n):
            region = regions[i]
            change = self.price_change_rates[region]
            prices[i] *= (1.0 + change)
            jeonse_prices[i] *= (1.0 + change * 0.8)

    def simple_matching(self, households, houses, rng: np.random.Generator):
        """단순 매칭 (NumPy 사용) - 하위 호환용"""
        self.enhanced_matching(households, houses, rng, 0)

    def enhanced_matching(self, households, houses, rng: np.random.Generator, current_month: int):
        """향상된 매칭 (매입가 기록 포함)

        행동경제학 요소를 위한 매입가 기록:
        - 손실 회피, 앵커링 계산에 사용
        """
        wants_buy = households.wants_to_buy.to_numpy()
        target_regions = households.target_region.to_numpy()
        buyer_assets = households.asset.to_numpy()
        owned_houses = households.owned_houses.to_numpy()

        is_for_sale = houses.is_for_sale.to_numpy()
        house_regions = houses.region.to_numpy()
        house_prices = houses.price.to_numpy()
        house_owners = houses.owner_id.to_numpy()

        # 매입가 기록용
        purchase_prices = households.purchase_price.to_numpy()
        purchase_months = households.purchase_month.to_numpy()
        total_purchase_prices = households.total_purchase_price.to_numpy()

        transactions = np.zeros(NUM_REGIONS, dtype=np.int32)

        for region in range(NUM_REGIONS):
            buyers = np.where((wants_buy == 1) & (target_regions == region))[0]
            sellers = np.where((is_for_sale == 1) & (house_regions == region))[0]

            if len(buyers) == 0 or len(sellers) == 0:
                continue

            rng.shuffle(buyers)
            rng.shuffle(sellers)

            matched = 0
            for buyer_id in buyers:
                if matched >= len(sellers):
                    break

                house_id = sellers[matched]
                price = house_prices[house_id]

                # 구매력 체크 (자산 50% + 대출 50%로 구매 가능)
                if buyer_assets[buyer_id] * 0.5 >= price * 0.3:  # 자산이 가격의 30% 이상
                    # 거래 성사
                    old_owner = house_owners[house_id]

                    is_for_sale[house_id] = 0
                    wants_buy[buyer_id] = 0
                    house_owners[house_id] = buyer_id
                    owned_houses[buyer_id] += 1

                    # 매수자 매입가 기록
                    if owned_houses[buyer_id] == 1:  # 첫 주택
                        purchase_prices[buyer_id] = price
                    total_purchase_prices[buyer_id] += price
                    purchase_months[buyer_id] = current_month

                    # 매도자 처리
                    if old_owner >= 0:
                        owned_houses[old_owner] -= 1
                        # 매도자 매입가 차감 (평균값 기준)
                        if owned_houses[old_owner] > 0:
                            avg_purchase = total_purchase_prices[old_owner] / (owned_houses[old_owner] + 1)
                            total_purchase_prices[old_owner] -= avg_purchase
                        else:
                            # 무주택자가 됨
                            purchase_prices[old_owner] = 0.0
                            total_purchase_prices[old_owner] = 0.0

                    matched += 1
                    transactions[region] += 1

        # 결과 반영
        houses.is_for_sale.from_numpy(is_for_sale.astype(np.int32))
        houses.owner_id.from_numpy(house_owners.astype(np.int32))
        households.wants_to_buy.from_numpy(wants_buy.astype(np.int32))
        households.owned_houses.from_numpy(owned_houses.astype(np.int32))
        households.purchase_price.from_numpy(purchase_prices.astype(np.float32))
        households.purchase_month.from_numpy(purchase_months.astype(np.int32))
        households.total_purchase_price.from_numpy(total_purchase_prices.astype(np.float32))
        self.transactions.from_numpy(transactions)

    def record_history(self):
        """이력 기록"""
        self.price_history.append(self.region_prices.to_numpy().copy())
        self.transaction_history.append(self.transactions.to_numpy().copy())

        prices = self.region_prices.to_numpy()
        jeonse = self.region_jeonse_prices.to_numpy()
        ratio = np.divide(jeonse, prices, where=prices > 0, out=np.zeros_like(prices))
        self.jeonse_ratio_history.append(ratio.copy())

        # 호가 스프레드 기록
        spread = self.order_book.get_bid_ask_spread()
        self.bid_ask_spread_history.append(spread.copy())

    # =========================================================================
    # 가격 적정성 지표 계산 (구조적 개선)
    # =========================================================================

    def update_price_metrics(self, households):
        """가격 적정성 지표 업데이트

        계산 지표:
        1. PIR (Price-to-Income Ratio): 주택가격 / 연소득
        2. 역사적 평균 대비 현재 가격
        3. 기대 수익률 (가격상승률 + 임대수익률)
        4. 투자 매력도 지수
        """
        prices = self.region_prices.to_numpy()
        jeonse = self.region_jeonse_prices.to_numpy()
        price_changes = self.region_price_changes.to_numpy()

        # 지역별 평균 소득 계산
        incomes = households.income.to_numpy()
        regions = households.region.to_numpy()
        for r in range(NUM_REGIONS):
            mask = regions == r
            if np.sum(mask) > 0:
                self.region_avg_income[r] = np.mean(incomes[mask])

        # 1. PIR (Price-to-Income Ratio)
        # 주택가격 / (월소득 * 12)
        annual_income = self.region_avg_income * 12
        pir = np.divide(prices, annual_income, where=annual_income > 0, out=np.zeros_like(prices))
        pir = np.clip(pir, 0, 50)  # 최대 50배
        self.region_pir.from_numpy(pir.astype(np.float32))

        # 2. 역사적 평균 대비 현재 가격
        # 이동평균 업데이트 (지수 이동평균, decay=0.98)
        if np.sum(self.historical_avg_prices) == 0:
            self.historical_avg_prices = prices.copy()
        else:
            self.historical_avg_prices = 0.98 * self.historical_avg_prices + 0.02 * prices

        price_to_hist = np.divide(
            prices, self.historical_avg_prices,
            where=self.historical_avg_prices > 0,
            out=np.ones_like(prices)
        )
        self.price_to_historical.from_numpy(price_to_hist.astype(np.float32))

        # 3. 기대 수익률 (연율)
        # = 가격상승률(연환산) + 임대수익률
        annual_appreciation = price_changes * 12  # 월간 → 연간
        rental_yield = np.divide(
            jeonse * 0.04,  # 전세가의 4%를 연 임대수익으로 추정
            prices,
            where=prices > 0,
            out=np.zeros_like(prices)
        )
        expected_return = annual_appreciation + rental_yield
        expected_return = np.clip(expected_return, -0.3, 0.5)  # -30% ~ +50%
        self.expected_return.from_numpy(expected_return.astype(np.float32))

        # 4. 투자 매력도 지수 (종합)
        # = 기대수익률 - PIR 페널티 + 유동성 보너스
        # PIR이 높을수록 매력도 감소 (과대평가)
        # PIR 기준: 10 이하 적정, 15 이상 고평가, 20 이상 과열
        pir_penalty = np.clip((pir - 10) * 0.02, -0.1, 0.2)

        # 거래 활성도 (유동성)
        transactions = self.transactions.to_numpy().astype(np.float32)
        total_houses = self.total_houses.to_numpy().astype(np.float32)
        liquidity = np.divide(transactions, total_houses, where=total_houses > 0, out=np.zeros_like(prices))
        liquidity_bonus = np.clip(liquidity * 2, 0, 0.05)

        attractiveness = expected_return - pir_penalty + liquidity_bonus
        attractiveness = np.clip(attractiveness, -0.3, 0.3)
        self.investment_attractiveness.from_numpy(attractiveness.astype(np.float32))

    def update_dynamic_prestige(self, households):
        """동적 프리미엄 업데이트 (매 스텝 호출)

        구조적 프리미엄(REGION_PRESTIGE)에 3가지 동적 요소를 가산:
        1. 학군 프리미엄: 학령기 자녀 밀집도
        2. 명성 모멘텀: 가격 추세 누적 (강남 불패 심리)
        3. 고소득 집중도: 젠트리피케이션 효과
        """
        dp_cfg = self.config.dynamic_prestige

        regions = households.region.to_numpy()
        child_ages = households.eldest_child_age.to_numpy()
        num_children = households.num_children.to_numpy()
        incomes = households.income.to_numpy()

        # --- 1. 학군 프리미엄 (학령기 자녀 밀집도) ---
        school_age_count = np.zeros(NUM_REGIONS, dtype=np.float32)
        total_count = np.zeros(NUM_REGIONS, dtype=np.float32)

        for r in range(NUM_REGIONS):
            mask = regions == r
            total_count[r] = np.sum(mask)
            if total_count[r] > 0:
                # 학령기 자녀가 있는 가구 비율
                has_school_child = (
                    (num_children[mask] > 0) &
                    (child_ages[mask] >= dp_cfg.school_age_min) &
                    (child_ages[mask] <= dp_cfg.school_age_max)
                )
                school_age_count[r] = np.sum(has_school_child)

        school_age_ratio = np.divide(
            school_age_count, total_count,
            where=total_count > 0, out=np.zeros_like(school_age_count)
        )
        avg_school_ratio = np.mean(school_age_ratio[total_count > 0]) if np.any(total_count > 0) else 0.0
        school_premium = np.clip(
            (school_age_ratio - avg_school_ratio) * dp_cfg.school_premium_weight,
            dp_cfg.school_premium_min, dp_cfg.school_premium_max
        )

        # --- 2. 명성 모멘텀 (가격 추세 누적) ---
        price_changes = self.region_price_changes.to_numpy()
        # 지수이동평균: 기존 상승이 계속 영향 (감쇠하면서)
        self.prestige_momentum = (
            dp_cfg.momentum_decay * self.prestige_momentum +
            (1.0 - dp_cfg.momentum_decay) * price_changes * dp_cfg.momentum_sensitivity
        )
        momentum_premium = np.clip(
            self.prestige_momentum,
            dp_cfg.momentum_premium_min, dp_cfg.momentum_premium_max
        )

        # --- 3. 고소득 집중도 (젠트리피케이션) ---
        high_income_threshold = np.percentile(incomes, dp_cfg.high_income_percentile)
        high_income_ratio = np.zeros(NUM_REGIONS, dtype=np.float32)

        for r in range(NUM_REGIONS):
            mask = regions == r
            if np.sum(mask) > 0:
                high_income_ratio[r] = np.mean(incomes[mask] >= high_income_threshold)

        avg_high_ratio = np.mean(high_income_ratio[total_count > 0]) if np.any(total_count > 0) else 0.2
        concentration_premium = np.clip(
            (high_income_ratio / (avg_high_ratio + 1e-6) - 1.0) * dp_cfg.concentration_weight,
            dp_cfg.concentration_premium_min, dp_cfg.concentration_premium_max
        )

        # --- 최종 동적 프리미엄 ---
        self.dynamic_prestige = np.clip(
            REGION_PRESTIGE + school_premium + momentum_premium + concentration_premium,
            0.0, 1.5
        ).astype(np.float32)

    def get_region_metrics(self) -> dict:
        """지역별 지표 반환 (디버깅/분석용)"""
        return {
            'pir': self.region_pir.to_numpy(),
            'price_to_historical': self.price_to_historical.to_numpy(),
            'expected_return': self.expected_return.to_numpy(),
            'investment_attractiveness': self.investment_attractiveness.to_numpy(),
        }

    # =========================================================================
    # Double Auction 기반 거래 메커니즘
    # =========================================================================

    def double_auction_matching(
        self,
        households,
        houses,
        rng: np.random.Generator,
        current_month: int
    ) -> int:
        """Double Auction 기반 매칭

        1. 주문장 초기화
        2. 매수/매도 호가 생성
        3. 호가 매칭
        4. 거래 처리

        Args:
            households: Households 인스턴스
            houses: Houses 인스턴스
            rng: 난수 생성기
            current_month: 현재 월

        Returns:
            총 거래 건수
        """
        region_prices = self.region_prices.to_numpy()

        # 1. 주문장 초기화
        self.order_book.clear_orders()

        # 2. 매수 호가 생성
        self.order_book.generate_bid_orders(households, region_prices, rng)

        # 3. 매도 호가 생성
        self.order_book.generate_ask_orders(households, houses, region_prices, rng)

        # 4. 호가 매칭
        num_matched = self.order_book.match_orders()

        if num_matched == 0:
            return 0

        # 5. 거래 처리
        buyers, sellers, house_ids, prices = self.order_book.get_matched_transactions()

        self._process_transactions(
            households, houses, buyers, sellers, house_ids, prices, current_month
        )

        # 6. 거래 통계 업데이트
        house_regions = houses.region.to_numpy()
        transactions = np.zeros(NUM_REGIONS, dtype=np.int32)
        for house_id in house_ids:
            transactions[house_regions[house_id]] += 1
        self.transactions.from_numpy(transactions)

        return num_matched

    def _process_transactions(
        self,
        households,
        houses,
        buyers: np.ndarray,
        sellers: np.ndarray,
        house_ids: np.ndarray,
        prices: np.ndarray,
        current_month: int
    ):
        """거래 처리 (소유권 이전, 기록 업데이트)

        Args:
            households: Households 인스턴스
            houses: Houses 인스턴스
            buyers: 매수자 ID 배열
            sellers: 매도자 ID 배열
            house_ids: 거래 주택 ID 배열
            prices: 체결 가격 배열
            current_month: 현재 월
        """
        is_for_sale = houses.is_for_sale.to_numpy()
        house_owners = houses.owner_id.to_numpy()
        house_prices = houses.price.to_numpy()

        owned_houses = households.owned_houses.to_numpy()
        wants_buy = households.wants_to_buy.to_numpy()
        purchase_prices = households.purchase_price.to_numpy()
        purchase_months = households.purchase_month.to_numpy()
        total_purchase_prices = households.total_purchase_price.to_numpy()
        buyer_assets = households.asset.to_numpy()

        for i in range(len(buyers)):
            buyer_id = buyers[i]
            seller_id = sellers[i]
            house_id = house_ids[i]
            price = prices[i]

            # 주택 상태 업데이트
            is_for_sale[house_id] = 0
            house_owners[house_id] = buyer_id
            house_prices[house_id] = price  # 체결가로 업데이트

            # 매수자 업데이트
            wants_buy[buyer_id] = 0
            owned_houses[buyer_id] += 1
            buyer_assets[buyer_id] -= price * 0.3  # 계약금 차감

            if owned_houses[buyer_id] == 1:
                purchase_prices[buyer_id] = price
            total_purchase_prices[buyer_id] += price
            purchase_months[buyer_id] = current_month

            # 매도자 업데이트
            if seller_id >= 0:
                owned_houses[seller_id] -= 1
                if owned_houses[seller_id] > 0:
                    avg_purchase = total_purchase_prices[seller_id] / (owned_houses[seller_id] + 1)
                    total_purchase_prices[seller_id] -= avg_purchase
                else:
                    purchase_prices[seller_id] = 0.0
                    total_purchase_prices[seller_id] = 0.0

        # 필드 업데이트
        houses.is_for_sale.from_numpy(is_for_sale.astype(np.int32))
        houses.owner_id.from_numpy(house_owners.astype(np.int32))
        houses.price.from_numpy(house_prices.astype(np.float32))

        households.wants_to_buy.from_numpy(wants_buy.astype(np.int32))
        households.owned_houses.from_numpy(owned_houses.astype(np.int32))
        households.purchase_price.from_numpy(purchase_prices.astype(np.float32))
        households.purchase_month.from_numpy(purchase_months.astype(np.int32))
        households.total_purchase_price.from_numpy(total_purchase_prices.astype(np.float32))
        households.asset.from_numpy(buyer_assets.astype(np.float32))

    # =========================================================================
    # 전월세 전환 메커니즘
    # =========================================================================

    def update_jeonse_wolse_conversion(
        self,
        houses,
        households,
        conversion_rate: float,
        rng: np.random.Generator
    ):
        """전월세 전환 업데이트

        금리 연동 전환 모델:
        - 월세 = (전세보증금 - 월세보증금) * 전환율 / 12
        - 금리 상승 → 월세 선호 증가
        - 금리 하락 → 전세 선호 증가

        Args:
            houses: Houses 인스턴스
            households: Households 인스턴스
            conversion_rate: 전월세 전환율 (연율)
            rng: 난수 생성기
        """
        self.conversion_rate[None] = conversion_rate

        is_jeonse = houses.is_jeonse.to_numpy()
        house_prices = houses.price.to_numpy()
        jeonse_prices = houses.jeonse_price.to_numpy()
        owner_ids = houses.owner_id.to_numpy()
        tenant_ids = houses.tenant_id.to_numpy()

        # 임대인 (다주택자) 전월세 전환 결정
        # 금리 높으면 월세 선호

        # 기준 금리 (전환율에서 추정)
        interest_threshold = 0.045  # 4.5%

        for i in range(houses.n):
            if owner_ids[i] >= 0 and owner_ids[i] != tenant_ids[i]:
                # 임대 중인 주택

                if conversion_rate > interest_threshold:
                    # 고금리: 월세 전환 확률 증가
                    if is_jeonse[i] == 1 and rng.random() < 0.02:  # 월 2% 전환
                        is_jeonse[i] = 0  # 월세로 전환

                        # 월세 계산
                        jeonse_deposit = jeonse_prices[i]
                        wolse_deposit = jeonse_deposit * 0.3  # 30% 보증금
                        monthly_rent = (jeonse_deposit - wolse_deposit) * conversion_rate / 12
                        # monthly_rent는 별도 필드에 저장 필요

                else:
                    # 저금리: 전세 전환 확률 증가
                    if is_jeonse[i] == 0 and rng.random() < 0.01:  # 월 1% 전환
                        is_jeonse[i] = 1  # 전세로 전환

        houses.is_jeonse.from_numpy(is_jeonse.astype(np.int32))

        # 지역별 전세 비율 계산
        regions = houses.region.to_numpy()
        jeonse_ratio = np.zeros(NUM_REGIONS, dtype=np.float32)

        for r in range(NUM_REGIONS):
            region_mask = regions == r
            region_count = np.sum(region_mask)
            if region_count > 0:
                jeonse_ratio[r] = np.sum(is_jeonse[region_mask]) / region_count

        self.jeonse_wolse_ratio.from_numpy(jeonse_ratio)

    def calculate_wolse_from_jeonse(
        self,
        jeonse_deposit: float,
        wolse_deposit_ratio: float,
        conversion_rate: float
    ) -> float:
        """전세보증금에서 월세 계산

        Args:
            jeonse_deposit: 전세 보증금
            wolse_deposit_ratio: 월세 보증금 비율 (0~1)
            conversion_rate: 전환율 (연율)

        Returns:
            월세 금액
        """
        wolse_deposit = jeonse_deposit * wolse_deposit_ratio
        monthly_rent = (jeonse_deposit - wolse_deposit) * conversion_rate / 12
        return monthly_rent

    def tenant_utility_comparison(
        self,
        jeonse_deposit: float,
        monthly_rent: float,
        opportunity_cost: float,
        housing_utility: float
    ) -> tuple:
        """임차인 효용 비교 (전세 vs 월세)

        Args:
            jeonse_deposit: 전세 보증금
            monthly_rent: 월세 금액
            opportunity_cost: 자금 기회비용
            housing_utility: 주거 효용

        Returns:
            (전세 효용, 월세 효용)
        """
        # 전세 효용 = -보증금*기회비용 + 주거효용
        utility_jeonse = -jeonse_deposit * opportunity_cost + housing_utility

        # 월세 효용 = -월세*12 + 보증금이자 + 주거효용
        wolse_deposit = jeonse_deposit * 0.3  # 가정
        deposit_interest = wolse_deposit * opportunity_cost
        utility_wolse = -monthly_rent * 12 + deposit_interest + housing_utility

        return utility_jeonse, utility_wolse

    def landlord_return_comparison(
        self,
        jeonse_deposit: float,
        monthly_rent: float,
        expected_return: float,
        interest_rate: float
    ) -> tuple:
        """임대인 수익률 비교 (전세 vs 월세)

        Args:
            jeonse_deposit: 전세 보증금
            monthly_rent: 월세 금액
            expected_return: 기대 수익률
            interest_rate: 시장 금리

        Returns:
            (전세 수익, 월세 수익)
        """
        wolse_deposit = jeonse_deposit * 0.3

        # 전세 수익 = 보증금 * (기대수익률 - 금리)
        jeonse_return = jeonse_deposit * (expected_return - interest_rate)

        # 월세 수익 = 월세*12 + 보증금*금리
        wolse_return = monthly_rent * 12 + wolse_deposit * interest_rate

        return jeonse_return, wolse_return
