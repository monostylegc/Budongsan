"""주택 공급 모듈 - 내생적 공급 및 재건축 모델"""

import numpy as np
import taichi as ti
from dataclasses import dataclass, field
from typing import List
from .config import Config, NUM_REGIONS, REGIONS


@dataclass
class ConstructionProject:
    """건설 프로젝트"""
    region: int
    num_units: int
    start_month: int
    completion_month: int
    is_redevelopment: bool = False
    demolished_houses: List[int] = field(default_factory=list)


class SupplyModel:
    """주택 공급 모델

    학술적 근거:
    - Saiz (2010): 공급 탄력성과 토지 가용성
    - 한국 재건축/재개발 시장 특성 반영

    공급 탄력성:
        new_supply_rate = base_rate * (price_change / threshold)^elasticity

    재건축:
        - 조건: 건물 연식 30년+, 5년간 가격 상승률 30%+
        - 건설 기간: 24개월
    """

    def __init__(self, config: Config):
        self.config = config
        self.supply_cfg = config.supply

        # 지역별 공급 탄력성 매핑
        self.elasticity = self._get_regional_elasticity()

        # 건설 파이프라인
        self.construction_pipeline: List[ConstructionProject] = []

        # 통계
        self.monthly_new_supply = np.zeros(NUM_REGIONS, dtype=np.int32)
        self.monthly_demolished = np.zeros(NUM_REGIONS, dtype=np.int32)
        self.total_under_construction = 0

        # 기록
        self.supply_history = []
        self.redevelopment_history = []

    def _get_regional_elasticity(self) -> np.ndarray:
        """지역별 공급 탄력성 설정

        Returns:
            지역별 탄력성 배열
        """
        cfg = self.supply_cfg
        elasticity = np.zeros(NUM_REGIONS, dtype=np.float32)

        for region_id, region_info in REGIONS.items():
            tier = region_info['tier']
            name = region_info['name']

            if '강남' in name:
                elasticity[region_id] = cfg.elasticity_gangnam
            elif tier == 1:  # 서울 핵심
                elasticity[region_id] = cfg.elasticity_gangnam * 1.2
            elif tier == 2:  # 서울/수도권
                elasticity[region_id] = cfg.elasticity_seoul
            elif '경기' in name:
                elasticity[region_id] = cfg.elasticity_gyeonggi
            else:  # 지방
                elasticity[region_id] = cfg.elasticity_local

        return elasticity

    def calculate_new_supply(
        self,
        price_changes_12m: np.ndarray,
        current_stock: np.ndarray
    ) -> np.ndarray:
        """신규 공급 계산

        new_supply_rate = base_rate * (price_change / threshold)^elasticity

        Args:
            price_changes_12m: 12개월 가격 변화율 (지역별)
            current_stock: 현재 주택 재고 (지역별)

        Returns:
            지역별 신규 공급 물량
        """
        cfg = self.supply_cfg

        # 가격 변화에 따른 공급 반응
        price_ratio = price_changes_12m / cfg.price_threshold
        price_ratio = np.clip(price_ratio, 0, 5)  # 최대 5배

        # 공급률 계산
        supply_rate = cfg.base_supply_rate * np.power(price_ratio + 1, self.elasticity)
        supply_rate = np.clip(supply_rate, 0, cfg.max_construction_ratio)

        # 신규 공급 물량
        new_supply = (current_stock * supply_rate).astype(np.int32)

        return new_supply

    def check_redevelopment_eligibility(
        self,
        houses,
        price_history_5y: np.ndarray,
        current_month: int
    ) -> np.ndarray:
        """재건축 가능 주택 확인

        조건:
        - 건물 연식 30년 이상
        - 5년간 가격 상승률 30% 이상

        Args:
            houses: Houses 인스턴스
            price_history_5y: 5년간 가격 변화율 (지역별)
            current_month: 현재 월

        Returns:
            재건축 대상 주택 ID 배열
        """
        cfg = self.supply_cfg

        building_age = houses.building_age.to_numpy()
        regions = houses.region.to_numpy()
        is_for_sale = houses.is_for_sale.to_numpy()

        # 재건축 조건 체크
        age_eligible = building_age >= cfg.redevelopment_age_threshold

        # 지역별 가격 상승률 조건
        price_eligible = np.zeros(len(building_age), dtype=bool)
        for region in range(NUM_REGIONS):
            region_mask = regions == region
            if price_history_5y[region] >= cfg.redevelopment_price_threshold:
                price_eligible[region_mask] = True

        # 매물 아닌 주택만 대상
        not_for_sale = is_for_sale == 0

        eligible = age_eligible & price_eligible & not_for_sale

        return np.where(eligible)[0]

    def initiate_redevelopment(
        self,
        eligible_houses: np.ndarray,
        houses,
        current_month: int,
        rng: np.random.Generator
    ) -> int:
        """재건축 프로젝트 시작

        Args:
            eligible_houses: 재건축 대상 주택 ID
            houses: Houses 인스턴스
            current_month: 현재 월
            rng: 난수 생성기

        Returns:
            시작된 재건축 프로젝트 수
        """
        if len(eligible_houses) == 0:
            return 0

        cfg = self.supply_cfg

        # 월간 재건축 확률 적용
        redevelopment_prob = rng.random(len(eligible_houses))
        selected = eligible_houses[redevelopment_prob < cfg.redevelopment_base_prob]

        if len(selected) == 0:
            return 0

        regions = houses.region.to_numpy()

        # 지역별 그룹화하여 프로젝트 생성
        projects_created = 0
        for region in range(NUM_REGIONS):
            region_houses = selected[regions[selected] == region]
            if len(region_houses) == 0:
                continue

            # 재건축 시 용적률 상향으로 1.3배 공급
            new_units = int(len(region_houses) * 1.3)

            project = ConstructionProject(
                region=region,
                num_units=new_units,
                start_month=current_month,
                completion_month=current_month + cfg.construction_period,
                is_redevelopment=True,
                demolished_houses=region_houses.tolist()
            )

            self.construction_pipeline.append(project)
            self.monthly_demolished[region] += len(region_houses)
            projects_created += 1

            # 철거 처리 (주택 비활성화)
            self._mark_houses_demolished(houses, region_houses)

        return projects_created

    def _mark_houses_demolished(self, houses, house_ids: np.ndarray):
        """주택 철거 처리"""
        # 소유자/세입자 퇴거 처리 필요
        owner_ids = houses.owner_id.to_numpy()
        tenant_ids = houses.tenant_id.to_numpy()

        for house_id in house_ids:
            # 해당 주택 비활성화 (가격 0, 소유자 없음)
            houses.price.to_numpy()[house_id] = 0
            houses.owner_id.to_numpy()[house_id] = -1
            houses.tenant_id.to_numpy()[house_id] = -1
            houses.is_for_sale.to_numpy()[house_id] = 0

    def add_new_construction(
        self,
        new_supply: np.ndarray,
        current_month: int
    ):
        """신규 건설 프로젝트 추가

        Args:
            new_supply: 지역별 신규 공급 물량
            current_month: 현재 월
        """
        cfg = self.supply_cfg

        for region in range(NUM_REGIONS):
            if new_supply[region] > 0:
                project = ConstructionProject(
                    region=region,
                    num_units=new_supply[region],
                    start_month=current_month,
                    completion_month=current_month + cfg.construction_period,
                    is_redevelopment=False
                )
                self.construction_pipeline.append(project)

    def complete_construction(
        self,
        houses,
        current_month: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """완공된 프로젝트 처리

        Args:
            houses: Houses 인스턴스
            current_month: 현재 월
            rng: 난수 생성기

        Returns:
            지역별 완공 물량
        """
        completed = np.zeros(NUM_REGIONS, dtype=np.int32)

        completed_projects = [
            p for p in self.construction_pipeline
            if p.completion_month <= current_month
        ]

        for project in completed_projects:
            completed[project.region] += project.num_units
            self.construction_pipeline.remove(project)

            # 주택 추가 (Houses 클래스의 add_new_houses 호출)
            self._add_houses_to_market(
                houses, project.region, project.num_units, rng
            )

        self.monthly_new_supply = completed
        return completed

    def _add_houses_to_market(
        self,
        houses,
        region: int,
        num_units: int,
        rng: np.random.Generator
    ):
        """신규 주택을 시장에 추가

        기존 주택 재고에 새 주택 정보 추가
        (실제 구현은 Houses 클래스에서 처리)
        """
        # 빈 슬롯 찾기 (owner_id == -1 && price == 0)
        prices = houses.price.to_numpy()
        owner_ids = houses.owner_id.to_numpy()

        empty_slots = np.where((owner_ids == -1) & (prices == 0))[0]

        if len(empty_slots) < num_units:
            # 슬롯 부족 시 경고
            num_units = len(empty_slots)

        if num_units == 0:
            return

        # 지역 기준가
        base_price = REGIONS[region]['base_price']

        # 신규 주택 정보 설정
        slots_to_use = empty_slots[:num_units]

        for slot in slots_to_use:
            # 가격 (신규 분양가: 기존가 대비 프리미엄)
            prices[slot] = base_price * rng.uniform(1.0, 1.2)
            houses.region.to_numpy()[slot] = region
            houses.building_age.to_numpy()[slot] = 0  # 신축
            houses.size.to_numpy()[slot] = rng.uniform(20, 40)  # 중형 위주
            houses.is_for_sale.to_numpy()[slot] = 1  # 분양 중
            houses.owner_id.to_numpy()[slot] = -1  # 미분양
            houses.is_jeonse.to_numpy()[slot] = 0

        # 필드 업데이트
        houses.price.from_numpy(prices)

    def step(
        self,
        houses,
        price_changes_12m: np.ndarray,
        price_history_5y: np.ndarray,
        current_stock: np.ndarray,
        current_month: int,
        rng: np.random.Generator
    ) -> dict:
        """월간 공급 업데이트

        Args:
            houses: Houses 인스턴스
            price_changes_12m: 12개월 가격 변화율
            price_history_5y: 5년간 가격 상승률
            current_stock: 현재 주택 재고
            current_month: 현재 월
            rng: 난수 생성기

        Returns:
            공급 통계 딕셔너리
        """
        # 1. 완공 처리
        completed = self.complete_construction(houses, current_month, rng)

        # 2. 신규 공급 계산
        new_supply = self.calculate_new_supply(price_changes_12m, current_stock)

        # 3. 재건축 확인 및 시작
        eligible = self.check_redevelopment_eligibility(
            houses, price_history_5y, current_month
        )
        redevelopments = self.initiate_redevelopment(
            eligible, houses, current_month, rng
        )

        # 4. 신규 건설 추가
        self.add_new_construction(new_supply, current_month)

        # 5. 통계 업데이트
        self.total_under_construction = sum(p.num_units for p in self.construction_pipeline)

        stats = {
            'completed': completed.sum(),
            'new_starts': new_supply.sum(),
            'redevelopments': redevelopments,
            'under_construction': self.total_under_construction,
            'demolished': self.monthly_demolished.sum()
        }

        self.supply_history.append(stats)
        self.monthly_demolished = np.zeros(NUM_REGIONS, dtype=np.int32)

        return stats

    def get_pipeline_by_region(self) -> np.ndarray:
        """지역별 건설 중 물량 반환"""
        pipeline = np.zeros(NUM_REGIONS, dtype=np.int32)
        for project in self.construction_pipeline:
            pipeline[project.region] += project.num_units
        return pipeline

    def reset(self):
        """상태 초기화"""
        self.construction_pipeline = []
        self.monthly_new_supply = np.zeros(NUM_REGIONS, dtype=np.int32)
        self.monthly_demolished = np.zeros(NUM_REGIONS, dtype=np.int32)
        self.total_under_construction = 0
        self.supply_history = []
        self.redevelopment_history = []
