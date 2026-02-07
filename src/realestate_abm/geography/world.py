"""지역 시스템 - JSON 기반 동적 지역 정의"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from .distance import compute_distances, compute_adjacency


@dataclass
class Region:
    """하나의 지역"""
    id: str
    name: str
    x: float
    y: float
    tier: int
    base_price: float
    population: int = 100000
    housing_stock: int = 50000
    prestige: float = 0.5
    job_density: float = 0.5
    supply_elasticity: float = 1.0
    industry_mix: dict[str, float] = field(default_factory=dict)
    income_premium: float = 1.0
    amenities: dict[str, float] = field(default_factory=dict)
    household_ratio: float = 0.0  # 가구 분포 비율 (자동 계산 가능)


class RegionSet:
    """지역 집합 - 거리/인접도 자동 계산"""

    def __init__(
        self,
        regions: list[Region],
        max_commute_km: float = 40.0,
        adjacency_decay: float = 0.03,
    ):
        self.regions = regions
        self.n = len(regions)
        self.max_commute_km = max_commute_km
        self.adjacency_decay = adjacency_decay

        # ID → index 매핑
        self.id_to_idx: dict[str, int] = {r.id: i for i, r in enumerate(regions)}

        # 좌표 배열
        coords = np.array([[r.x, r.y] for r in regions], dtype=np.float64)
        self.coords = coords

        # 거리 행렬 계산
        self.distances = compute_distances(coords)

        # 인접도 행렬 계산
        self.adjacency = compute_adjacency(self.distances, adjacency_decay)

        # 통근 가능 여부
        self.commutable = (self.distances <= max_commute_km).astype(np.float32)

        # 속성 배열 추출
        self.base_prices = np.array([r.base_price for r in regions], dtype=np.float32)
        self.prestige = np.array([r.prestige for r in regions], dtype=np.float32)
        self.job_density = np.array([r.job_density for r in regions], dtype=np.float32)
        self.supply_elasticity = np.array([r.supply_elasticity for r in regions], dtype=np.float32)
        self.income_premium = np.array([r.income_premium for r in regions], dtype=np.float32)
        self.tiers = np.array([r.tier for r in regions], dtype=np.int32)

        # 가구 분포 비율
        ratios = np.array([r.household_ratio for r in regions], dtype=np.float32)
        if ratios.sum() == 0:
            # population 기반 자동 계산
            pops = np.array([r.population for r in regions], dtype=np.float64)
            ratios = (pops / pops.sum()).astype(np.float32)
        else:
            ratios = ratios / ratios.sum()
        self.household_ratio = ratios

        # 산업 구성 행렬
        self._build_industry_matrix()

    def _build_industry_matrix(self):
        """산업 구성 행렬 생성"""
        # 모든 산업 종류 수집
        all_industries = set()
        for r in self.regions:
            all_industries.update(r.industry_mix.keys())
        self.industry_names = sorted(all_industries)
        self.n_industries = len(self.industry_names)
        industry_idx = {name: i for i, name in enumerate(self.industry_names)}

        # (n_regions, n_industries) 행렬
        self.industry_mix = np.zeros((self.n, self.n_industries), dtype=np.float32)
        for i, r in enumerate(self.regions):
            for name, ratio in r.industry_mix.items():
                if name in industry_idx:
                    self.industry_mix[i, industry_idx[name]] = ratio
            # 정규화
            row_sum = self.industry_mix[i].sum()
            if row_sum > 0:
                self.industry_mix[i] /= row_sum

    def get_index(self, region_id: str) -> int:
        return self.id_to_idx[region_id]

    def get_region(self, idx: int) -> Region:
        return self.regions[idx]

    def get_names(self) -> list[str]:
        return [r.name for r in self.regions]

    @classmethod
    def from_json(cls, path: str | Path) -> RegionSet:
        """JSON 파일에서 로드"""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        max_commute = data.get("max_commute_km", 40.0)
        decay = data.get("adjacency_decay", 0.03)

        regions = []
        for rd in data["regions"]:
            regions.append(Region(
                id=rd["id"],
                name=rd["name"],
                x=rd["x"],
                y=rd["y"],
                tier=rd.get("tier", 2),
                base_price=rd["base_price"],
                population=rd.get("population", 100000),
                housing_stock=rd.get("housing_stock", 50000),
                prestige=rd.get("prestige", 0.5),
                job_density=rd.get("job_density", 0.5),
                supply_elasticity=rd.get("supply_elasticity", 1.0),
                industry_mix=rd.get("industry_mix", {}),
                income_premium=rd.get("income_premium", 1.0),
                amenities=rd.get("amenities", {}),
                household_ratio=rd.get("household_ratio", 0.0),
            ))

        return cls(regions, max_commute, decay)

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "max_commute_km": self.max_commute_km,
            "adjacency_decay": self.adjacency_decay,
            "regions": [
                {
                    "id": r.id, "name": r.name, "x": r.x, "y": r.y,
                    "tier": r.tier, "base_price": r.base_price,
                    "population": r.population, "housing_stock": r.housing_stock,
                    "prestige": r.prestige, "job_density": r.job_density,
                    "supply_elasticity": r.supply_elasticity,
                    "industry_mix": r.industry_mix,
                    "income_premium": r.income_premium,
                    "amenities": r.amenities,
                    "household_ratio": r.household_ratio,
                }
                for r in self.regions
            ]
        }
