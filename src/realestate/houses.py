"""주택 데이터 (Taichi fields)"""

import taichi as ti
import numpy as np
from .config import Config, NUM_REGIONS, REGIONS, REGION_HOUSEHOLD_RATIO


@ti.data_oriented
class Houses:
    """주택들을 관리하는 클래스"""

    def __init__(self, config: Config):
        self.n = config.num_houses
        self.config = config

        # 기본 속성
        self.region = ti.field(dtype=ti.i32, shape=self.n)
        self.price = ti.field(dtype=ti.f32, shape=self.n)         # 매매가 (만원)
        self.jeonse_price = ti.field(dtype=ti.f32, shape=self.n)  # 전세가 (만원)
        self.size = ti.field(dtype=ti.f32, shape=self.n)          # 전용면적 (평)
        self.building_age = ti.field(dtype=ti.i32, shape=self.n)  # 건물 연식

        # 소유 및 상태
        self.owner_id = ti.field(dtype=ti.i32, shape=self.n)      # 소유자 ID (-1이면 미분양)
        self.is_for_sale = ti.field(dtype=ti.i32, shape=self.n)   # 매물 여부
        self.tenant_id = ti.field(dtype=ti.i32, shape=self.n)     # 세입자 ID (-1이면 없음)
        self.is_jeonse = ti.field(dtype=ti.i32, shape=self.n)     # 전세 여부 (0: 자가/월세, 1: 전세)

        # 가격 이력 (최근 6개월)
        self.price_history = ti.field(dtype=ti.f32, shape=(self.n, 6))

        # 거래
        self.months_on_market = ti.field(dtype=ti.i32, shape=self.n)  # 매물 등록 후 경과 월

        # === 공급 관련 필드 ===
        self.is_active = ti.field(dtype=ti.i32, shape=self.n)  # 활성 주택 (0: 철거/건설중)
        self.construction_complete_month = ti.field(dtype=ti.i32, shape=self.n)  # 완공 예정 월

        # === 전월세 상세 필드 ===
        self.monthly_rent = ti.field(dtype=ti.f32, shape=self.n)  # 월세 (만원)
        self.wolse_deposit = ti.field(dtype=ti.f32, shape=self.n)  # 월세 보증금 (만원)

        # === 노후화/멸실 관련 필드 ===
        self.condition = ti.field(dtype=ti.f32, shape=self.n)     # 건물 상태 (0~1, 1=신축)
        self.maintenance_cost = ti.field(dtype=ti.f32, shape=self.n)  # 월간 유지보수 비용
        self.is_demolished = ti.field(dtype=ti.i32, shape=self.n)  # 멸실 여부

        # 노후화 파라미터
        self.depreciation_rate = 0.003  # 월간 감가상각률 (연 ~3.6%)
        self.min_condition = 0.3        # 최소 상태 (30%)
        self.demolition_threshold = 0.35  # 멸실 임계값
        self.disaster_rate = 0.0001     # 월간 재해 멸실 확률 (0.01%)

    def initialize(self, rng: np.random.Generator):
        """초기 상태 설정"""
        # 지역 분포 (가구 분포와 유사하게)
        regions = rng.choice(NUM_REGIONS, size=self.n, p=REGION_HOUSEHOLD_RATIO).astype(np.int32)

        # 가격 설정 (지역 기준가 기반 + 변동)
        prices = np.zeros(self.n, dtype=np.float32)
        for region_id, info in REGIONS.items():
            mask = regions == region_id
            count = np.sum(mask)
            if count > 0:
                base = info["base_price"]
                # 로그정규 분포로 가격 변동
                prices[mask] = rng.lognormal(
                    mean=np.log(base),
                    sigma=0.3,
                    size=count
                ).astype(np.float32)

        # 전세가 (전세가율 60-80%)
        jeonse_ratio = rng.uniform(0.6, 0.8, size=self.n).astype(np.float32)
        jeonse_prices = prices * jeonse_ratio

        # 면적 (10-80평, 중위값 25평)
        sizes = rng.lognormal(mean=np.log(25), sigma=0.4, size=self.n).astype(np.float32)
        sizes = np.clip(sizes, 10, 80)

        # 건물 연식 (0-40년)
        building_ages = rng.integers(0, 41, size=self.n, dtype=np.int32)

        # 소유자 (-1로 초기화, 나중에 매칭)
        owners = np.full(self.n, -1, dtype=np.int32)
        tenants = np.full(self.n, -1, dtype=np.int32)

        # 매물 (초기 5%만 매물)
        for_sale = (rng.random(self.n) < 0.05).astype(np.int32)

        # 전세 여부 (55%가 전세)
        is_jeonse = (rng.random(self.n) < 0.55).astype(np.int32)

        # Taichi 필드에 복사
        self.region.from_numpy(regions)
        self.price.from_numpy(prices)
        self.jeonse_price.from_numpy(jeonse_prices)
        self.size.from_numpy(sizes)
        self.building_age.from_numpy(building_ages)
        self.owner_id.from_numpy(owners)
        self.is_for_sale.from_numpy(for_sale)
        self.tenant_id.from_numpy(tenants)
        self.is_jeonse.from_numpy(is_jeonse)
        self.months_on_market.from_numpy(np.zeros(self.n, dtype=np.int32))

        # 가격 이력 초기화
        price_hist = np.tile(prices.reshape(-1, 1), (1, 6))
        self.price_history.from_numpy(price_hist)

        # 공급 관련 초기화
        self.is_active.from_numpy(np.ones(self.n, dtype=np.int32))
        self.construction_complete_month.from_numpy(np.full(self.n, -1, dtype=np.int32))

        # 월세 관련 초기화 (월세 계약 비율 45%)
        wolse_mask = is_jeonse == 0
        monthly_rent = np.zeros(self.n, dtype=np.float32)
        wolse_deposit = np.zeros(self.n, dtype=np.float32)

        # 월세 = (전세보증금 - 월세보증금) * 전환율 / 12
        conversion_rate = 0.05  # 기본 5%
        wolse_deposit[wolse_mask] = prices[wolse_mask] * 0.3
        monthly_rent[wolse_mask] = (prices[wolse_mask] - wolse_deposit[wolse_mask]) * conversion_rate / 12

        self.monthly_rent.from_numpy(monthly_rent)
        self.wolse_deposit.from_numpy(wolse_deposit)

        # 노후화 관련 초기화
        # 건물 상태 = 1.0 - (연식/50) * 0.7 (신축=1.0, 50년=0.3)
        condition = 1.0 - (building_ages / 50.0) * 0.7
        condition = np.clip(condition, self.min_condition, 1.0).astype(np.float32)
        self.condition.from_numpy(condition)

        # 유지보수 비용 = 기본비용 * (2 - 상태)  (상태 나쁠수록 비용 증가)
        base_maintenance = prices * 0.001  # 가격의 0.1%
        maintenance_cost = base_maintenance * (2.0 - condition)
        self.maintenance_cost.from_numpy(maintenance_cost.astype(np.float32))

        self.is_demolished.from_numpy(np.zeros(self.n, dtype=np.int32))

    @ti.kernel
    def update_building_age(self):
        """건물 연식 증가 (연 1회 호출)"""
        for i in range(self.n):
            self.building_age[i] += 1

    @ti.kernel
    def update_price_history(self):
        """가격 이력 업데이트 (월 1회)"""
        for i in range(self.n):
            # 이력 시프트 (5->4->3->2->1->0)
            self.price_history[i, 5] = self.price_history[i, 4]
            self.price_history[i, 4] = self.price_history[i, 3]
            self.price_history[i, 3] = self.price_history[i, 2]
            self.price_history[i, 2] = self.price_history[i, 1]
            self.price_history[i, 1] = self.price_history[i, 0]
            self.price_history[i, 0] = self.price[i]

    @ti.kernel
    def count_for_sale_by_region(self, counts: ti.template()):
        """지역별 매물 수 계산"""
        for i in range(self.n):
            if self.is_for_sale[i] == 1:
                ti.atomic_add(counts[self.region[i]], 1)

    @ti.kernel
    def count_by_region(self, counts: ti.template()):
        """지역별 주택 수 계산"""
        for i in range(self.n):
            ti.atomic_add(counts[self.region[i]], 1)

    @ti.kernel
    def sum_prices_by_region(self, sums: ti.template(), counts: ti.template()):
        """지역별 가격 합계 및 개수"""
        for i in range(self.n):
            region = self.region[i]
            ti.atomic_add(sums[region], self.price[i])
            ti.atomic_add(counts[region], 1)

    @ti.kernel
    def update_months_on_market(self):
        """매물 등록 기간 업데이트"""
        for i in range(self.n):
            if self.is_for_sale[i] == 1:
                self.months_on_market[i] += 1
            else:
                self.months_on_market[i] = 0

    def add_new_houses(
        self,
        region: int,
        num_units: int,
        base_price: float,
        rng: np.random.Generator
    ) -> int:
        """신규 주택 추가

        Args:
            region: 지역 ID
            num_units: 추가할 주택 수
            base_price: 지역 기준가
            rng: 난수 생성기

        Returns:
            실제 추가된 주택 수
        """
        prices = self.price.to_numpy()
        regions = self.region.to_numpy()
        is_active = self.is_active.to_numpy()
        owner_ids = self.owner_id.to_numpy()
        building_ages = self.building_age.to_numpy()
        sizes = self.size.to_numpy()
        is_for_sale = self.is_for_sale.to_numpy()
        jeonse_prices = self.jeonse_price.to_numpy()
        is_jeonse = self.is_jeonse.to_numpy()

        # 비활성 슬롯 찾기
        empty_slots = np.where((is_active == 0) | ((owner_ids == -1) & (prices == 0)))[0]

        if len(empty_slots) < num_units:
            num_units = len(empty_slots)

        if num_units == 0:
            return 0

        slots = empty_slots[:num_units]

        for slot in slots:
            # 신규 분양가 (기준가 대비 0~20% 프리미엄)
            prices[slot] = base_price * rng.uniform(1.0, 1.2)
            jeonse_prices[slot] = prices[slot] * rng.uniform(0.6, 0.75)
            regions[slot] = region
            building_ages[slot] = 0  # 신축
            sizes[slot] = rng.uniform(20, 40)  # 중형 위주
            is_for_sale[slot] = 1  # 분양 중
            owner_ids[slot] = -1  # 미분양
            is_jeonse[slot] = 0
            is_active[slot] = 1

        # 필드 업데이트
        self.price.from_numpy(prices)
        self.jeonse_price.from_numpy(jeonse_prices)
        self.region.from_numpy(regions)
        self.building_age.from_numpy(building_ages)
        self.size.from_numpy(sizes)
        self.is_for_sale.from_numpy(is_for_sale)
        self.owner_id.from_numpy(owner_ids)
        self.is_jeonse.from_numpy(is_jeonse)
        self.is_active.from_numpy(is_active)

        return num_units

    def mark_demolished(self, house_ids: np.ndarray):
        """주택 철거 처리

        Args:
            house_ids: 철거할 주택 ID 배열
        """
        if len(house_ids) == 0:
            return

        is_active = self.is_active.to_numpy()
        prices = self.price.to_numpy()
        owner_ids = self.owner_id.to_numpy()
        tenant_ids = self.tenant_id.to_numpy()
        is_for_sale = self.is_for_sale.to_numpy()

        for house_id in house_ids:
            is_active[house_id] = 0
            prices[house_id] = 0
            owner_ids[house_id] = -1
            tenant_ids[house_id] = -1
            is_for_sale[house_id] = 0

        self.is_active.from_numpy(is_active)
        self.price.from_numpy(prices)
        self.owner_id.from_numpy(owner_ids)
        self.tenant_id.from_numpy(tenant_ids)
        self.is_for_sale.from_numpy(is_for_sale)

    def get_active_count_by_region(self) -> np.ndarray:
        """지역별 활성 주택 수 반환"""
        regions = self.region.to_numpy()
        is_active = self.is_active.to_numpy()

        counts = np.zeros(NUM_REGIONS, dtype=np.int32)
        for r in range(NUM_REGIONS):
            counts[r] = np.sum((regions == r) & (is_active == 1))

        return counts

    def update_jeonse_wolse(self, conversion_rate: float, rng: np.random.Generator):
        """전월세 상태 업데이트

        Args:
            conversion_rate: 전월세 전환율 (연율)
            rng: 난수 생성기
        """
        is_jeonse = self.is_jeonse.to_numpy()
        jeonse_prices = self.jeonse_price.to_numpy()
        monthly_rent = self.monthly_rent.to_numpy()
        wolse_deposit = self.wolse_deposit.to_numpy()

        # 월세인 경우 월세 금액 재계산
        wolse_mask = is_jeonse == 0

        # 보증금 = 전세가의 30%
        wolse_deposit[wolse_mask] = jeonse_prices[wolse_mask] * 0.3

        # 월세 = (전세가 - 보증금) * 전환율 / 12
        monthly_rent[wolse_mask] = (
            (jeonse_prices[wolse_mask] - wolse_deposit[wolse_mask]) * conversion_rate / 12
        )

        self.monthly_rent.from_numpy(monthly_rent)
        self.wolse_deposit.from_numpy(wolse_deposit)

    def update_depreciation(self):
        """건물 노후화 및 감가상각 업데이트 (월간)

        - 건물 상태 감소 (노후화)
        - 가격에 상태 반영 (감가상각)
        - 유지보수 비용 업데이트
        """
        condition = self.condition.to_numpy()
        prices = self.price.to_numpy()
        building_ages = self.building_age.to_numpy()
        is_active = self.is_active.to_numpy()
        maintenance = self.maintenance_cost.to_numpy()

        # 활성 주택만 대상
        active_mask = is_active == 1

        # 1. 건물 상태 감소 (월간 감가상각)
        # 연식이 높을수록 상태 감소 빠름
        age_factor = 1.0 + building_ages / 30.0  # 30년 건물은 2배 속도
        depreciation = self.depreciation_rate * age_factor
        condition[active_mask] -= depreciation[active_mask]
        condition = np.clip(condition, self.min_condition, 1.0)

        # 2. 가격에 상태 반영 (상태 나쁘면 가격 하락)
        # 신축(1.0) 대비 상태 0.5면 가격 25% 하락
        condition_factor = 0.5 + condition * 0.5  # 0.65 ~ 1.0
        # 가격 조정은 점진적으로 (월 1%씩)
        target_adjustment = condition_factor
        current_adjustment = prices / (prices + 1e-6)  # 현재 조정 비율 추정

        # 3. 유지보수 비용 업데이트 (상태 나쁠수록 증가)
        base_maintenance = prices * 0.001
        maintenance = base_maintenance * (2.0 - condition)

        self.condition.from_numpy(condition.astype(np.float32))
        self.maintenance_cost.from_numpy(maintenance.astype(np.float32))

    def check_natural_demolition(self, rng: np.random.Generator) -> np.ndarray:
        """자연 멸실 확인 (노후 건물 자연 소멸)

        Args:
            rng: 난수 생성기

        Returns:
            멸실된 주택 ID 배열
        """
        condition = self.condition.to_numpy()
        building_ages = self.building_age.to_numpy()
        is_active = self.is_active.to_numpy()
        is_demolished = self.is_demolished.to_numpy()

        # 멸실 조건:
        # 1. 상태가 임계값 이하
        # 2. 연식 40년 이상
        # 3. 랜덤 확률

        active_mask = is_active == 1
        old_mask = building_ages >= 40
        poor_condition = condition <= self.demolition_threshold

        # 멸실 후보
        candidates = np.where(active_mask & old_mask & poor_condition)[0]

        if len(candidates) == 0:
            return np.array([], dtype=np.int32)

        # 상태가 나쁠수록 멸실 확률 증가
        demolition_prob = (self.demolition_threshold - condition[candidates]) * 0.1
        demolition_prob = np.clip(demolition_prob, 0, 0.05)  # 최대 월 5%

        rolls = rng.random(len(candidates))
        demolished = candidates[rolls < demolition_prob]

        return demolished

    def check_disaster_demolition(self, rng: np.random.Generator) -> np.ndarray:
        """재해에 의한 멸실 (화재, 자연재해 등)

        Args:
            rng: 난수 생성기

        Returns:
            멸실된 주택 ID 배열
        """
        is_active = self.is_active.to_numpy()
        active_indices = np.where(is_active == 1)[0]

        if len(active_indices) == 0:
            return np.array([], dtype=np.int32)

        # 재해 멸실 (랜덤)
        rolls = rng.random(len(active_indices))
        demolished = active_indices[rolls < self.disaster_rate]

        return demolished

    def process_demolitions(self, demolished_ids: np.ndarray, households=None) -> dict:
        """멸실 처리

        Args:
            demolished_ids: 멸실된 주택 ID 배열
            households: Households 인스턴스 (소유자/세입자 처리용)

        Returns:
            멸실 통계 딕셔너리
        """
        if len(demolished_ids) == 0:
            return {'count': 0, 'owners_affected': 0, 'tenants_affected': 0}

        is_active = self.is_active.to_numpy()
        is_demolished = self.is_demolished.to_numpy()
        prices = self.price.to_numpy()
        owner_ids = self.owner_id.to_numpy()
        tenant_ids = self.tenant_id.to_numpy()
        is_for_sale = self.is_for_sale.to_numpy()

        owners_affected = 0
        tenants_affected = 0
        affected_owners_list = []

        for house_id in demolished_ids:
            # 소유자/세입자 카운트 (먼저 저장)
            if owner_ids[house_id] >= 0:
                owners_affected += 1
                affected_owners_list.append(owner_ids[house_id])
            if tenant_ids[house_id] >= 0:
                tenants_affected += 1

            # 주택 비활성화
            is_active[house_id] = 0
            is_demolished[house_id] = 1
            prices[house_id] = 0
            owner_ids[house_id] = -1
            tenant_ids[house_id] = -1
            is_for_sale[house_id] = 0

        # 가구 처리 (소유주택수 감소 등)
        if households is not None and len(affected_owners_list) > 0:
            owned_houses = households.owned_houses.to_numpy()
            for owner in affected_owners_list:
                owned_houses[owner] = max(0, owned_houses[owner] - 1)
            households.owned_houses.from_numpy(owned_houses)

        self.is_active.from_numpy(is_active)
        self.is_demolished.from_numpy(is_demolished)
        self.price.from_numpy(prices)
        self.owner_id.from_numpy(owner_ids)
        self.tenant_id.from_numpy(tenant_ids)
        self.is_for_sale.from_numpy(is_for_sale)

        return {
            'count': len(demolished_ids),
            'owners_affected': owners_affected,
            'tenants_affected': tenants_affected
        }

    def get_condition_stats(self) -> dict:
        """건물 상태 통계 반환"""
        condition = self.condition.to_numpy()
        building_ages = self.building_age.to_numpy()
        is_active = self.is_active.to_numpy()

        active_mask = is_active == 1
        active_condition = condition[active_mask]
        active_ages = building_ages[active_mask]

        return {
            'mean_condition': float(np.mean(active_condition)) if len(active_condition) > 0 else 0,
            'min_condition': float(np.min(active_condition)) if len(active_condition) > 0 else 0,
            'mean_age': float(np.mean(active_ages)) if len(active_ages) > 0 else 0,
            'old_buildings_30y': int(np.sum(active_ages >= 30)),
            'old_buildings_40y': int(np.sum(active_ages >= 40)),
            'poor_condition_count': int(np.sum(active_condition < 0.5)),
            'active_count': int(np.sum(active_mask)),
            'demolished_count': int(np.sum(is_active == 0))
        }
