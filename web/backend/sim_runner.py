"""시뮬레이션 래퍼 - 웹 API용 (전체 파라미터 지원)"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import asdict
import sys
import os

# 상위 디렉토리의 시뮬레이션 모듈 임포트
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.realestate.simulation import Simulation
from src.realestate.config import (
    Config, PolicyConfig, BehavioralConfig, LifeCycleConfig,
    ProspectTheoryConfig, DiscountingConfig, SupplyConfig,
    MacroConfig, NetworkConfig, AgentCompositionConfig, REGIONS, NUM_REGIONS
)


class SimulationRunner:
    """웹 API용 시뮬레이션 래퍼"""

    def __init__(self, params: Dict[str, Any], arch: str = "vulkan"):
        """
        Args:
            params: 시뮬레이션 파라미터 딕셔너리
            arch: Taichi 백엔드 ("vulkan", "cuda", "cpu")
        """
        self.params = params
        self.config = self._create_config(params)
        self.sim: Optional[Simulation] = None
        self.arch = arch
        self.is_running = False
        self.should_stop = False

    def _create_config(self, params: Dict[str, Any]) -> Config:
        """파라미터 딕셔너리에서 Config 객체 생성"""

        # 정책 설정
        policy_params = params.get('policy', {})
        policy = PolicyConfig(
            ltv_1house=policy_params.get('ltv_1house', 0.50),
            ltv_2house=policy_params.get('ltv_2house', 0.30),
            ltv_3house=policy_params.get('ltv_3house', 0.00),
            dti_limit=policy_params.get('dti_limit', 0.40),
            dsr_limit=policy_params.get('dsr_limit', 0.40),
            acq_tax_1house=policy_params.get('acq_tax_1house', 0.01),
            acq_tax_2house=policy_params.get('acq_tax_2house', 0.08),
            acq_tax_3house=policy_params.get('acq_tax_3house', 0.12),
            transfer_tax_short=policy_params.get('transfer_tax_short', 0.70),
            transfer_tax_long=policy_params.get('transfer_tax_long', 0.40),
            transfer_tax_multi_short=policy_params.get('transfer_tax_multi_short', 0.75),
            transfer_tax_multi_long=policy_params.get('transfer_tax_multi_long', 0.60),
            jongbu_threshold_1house=policy_params.get('jongbu_threshold_1house', 110000),
            jongbu_threshold_multi=policy_params.get('jongbu_threshold_multi', 60000),
            jongbu_rate=policy_params.get('jongbu_rate', 0.02),
            interest_rate=policy_params.get('interest_rate', 0.035),
            mortgage_spread=policy_params.get('mortgage_spread', 0.015),
            jeonse_loan_limit=policy_params.get('jeonse_loan_limit', 50000),
            rent_increase_cap=policy_params.get('rent_increase_cap', 0.05),
        )

        # 행동경제학 설정
        behavioral_params = params.get('behavioral', {})
        behavioral = BehavioralConfig(
            fomo_trigger_threshold=behavioral_params.get('fomo_trigger_threshold', 0.05),
            fomo_intensity=behavioral_params.get('fomo_intensity', 50.0),
            loss_aversion_mean=behavioral_params.get('loss_aversion_mean', 2.5),
            loss_aversion_std=behavioral_params.get('loss_aversion_std', 0.35),
            loss_aversion_decay=behavioral_params.get('loss_aversion_decay', 5.0),
            anchoring_threshold=behavioral_params.get('anchoring_threshold', 0.1),
            anchoring_penalty=behavioral_params.get('anchoring_penalty', 0.5),
            herding_trigger=behavioral_params.get('herding_trigger', 0.03),
            herding_intensity=behavioral_params.get('herding_intensity', 10.0),
            social_learning_rate=behavioral_params.get('social_learning_rate', 0.1),
            news_impact=behavioral_params.get('news_impact', 0.2),
        )

        # 생애주기 설정
        lifecycle_params = params.get('lifecycle', {})
        lifecycle = LifeCycleConfig(
            marriage_urgency_age_start=lifecycle_params.get('marriage_urgency_age_start', 28),
            marriage_urgency_age_end=lifecycle_params.get('marriage_urgency_age_end', 35),
            newlywed_housing_pressure=lifecycle_params.get('newlywed_housing_pressure', 1.5),
            parenting_housing_pressure=lifecycle_params.get('parenting_housing_pressure', 1.3),
            school_transition_age_start=lifecycle_params.get('school_transition_age_start', 10),
            school_transition_age_end=lifecycle_params.get('school_transition_age_end', 15),
            school_district_premium=lifecycle_params.get('school_district_premium', 1.2),
            retirement_start_age=lifecycle_params.get('retirement_start_age', 55),
            downsizing_probability=lifecycle_params.get('downsizing_probability', 0.1),
        )

        # 네트워크 설정
        network_params = params.get('network', {})
        network = NetworkConfig(
            avg_neighbors=network_params.get('avg_neighbors', 10),
            rewiring_prob=network_params.get('rewiring_prob', 0.1),
            cascade_threshold=network_params.get('cascade_threshold', 0.3),
            cascade_multiplier=network_params.get('cascade_multiplier', 2.0),
            self_weight=network_params.get('self_weight', 0.6),
            neighbor_weight=1.0 - network_params.get('self_weight', 0.6),
        )

        # 공급 설정
        supply_params = params.get('supply', {})
        supply_config = SupplyConfig(
            base_supply_rate=supply_params.get('base_supply_rate', 0.001),
            elasticity_gangnam=supply_params.get('elasticity_gangnam', 0.3),
            elasticity_seoul=supply_params.get('elasticity_seoul', 0.5),
            elasticity_gyeonggi=supply_params.get('elasticity_gyeonggi', 1.5),
            elasticity_local=supply_params.get('elasticity_local', 2.0),
            redevelopment_base_prob=supply_params.get('redevelopment_base_prob', 0.001),
            redevelopment_age_threshold=supply_params.get('redevelopment_age_threshold', 30),
            construction_period=supply_params.get('construction_period', 24),
        )

        # 거시경제 설정
        macro_params = params.get('macro', {})
        macro = MacroConfig(
            gdp_growth_mean=macro_params.get('gdp_growth_mean', 0.025),
            gdp_growth_volatility=macro_params.get('gdp_growth_volatility', 0.01),
            inflation_target=macro_params.get('inflation_target', 0.02),
            income_gdp_beta=macro_params.get('income_gdp_beta', 0.8),
        )

        # 시장 파라미터
        market_params = params.get('market', {})

        # 에이전트 구성 설정
        agent_comp_params = params.get('agent_composition', {})
        agent_composition = AgentCompositionConfig(
            investor_ratio=agent_comp_params.get('investor_ratio', 0.15),
            speculator_ratio=agent_comp_params.get('speculator_ratio', 0.05),
            speculator_risk_multiplier=agent_comp_params.get('speculator_risk_multiplier', 1.5),
            speculator_fomo_multiplier=agent_comp_params.get('speculator_fomo_multiplier', 1.3),
            speculator_horizon_min=agent_comp_params.get('speculator_horizon_min', 6),
            speculator_horizon_max=agent_comp_params.get('speculator_horizon_max', 24),
            initial_homeless_rate=agent_comp_params.get('initial_homeless_rate', 0.45),
            initial_one_house_rate=agent_comp_params.get('initial_one_house_rate', 0.40),
            initial_multi_house_rate=agent_comp_params.get('initial_multi_house_rate', 0.15),
            income_median=agent_comp_params.get('income_median', 300.0),
            income_sigma=agent_comp_params.get('income_sigma', 0.6),
            asset_median=agent_comp_params.get('asset_median', 5000.0),
            asset_alpha=agent_comp_params.get('asset_alpha', 1.5),
            age_young_ratio=agent_comp_params.get('age_young_ratio', 0.45),
            age_middle_ratio=agent_comp_params.get('age_middle_ratio', 0.43),
            age_senior_ratio=agent_comp_params.get('age_senior_ratio', 0.12),
        )

        # Config 생성
        config = Config(
            num_households=params.get('num_households', 100_000),
            num_houses=params.get('num_houses', 60_000),
            num_steps=params.get('num_steps', 120),
            seed=params.get('seed', 42),
            policy=policy,
            behavioral=behavioral,
            lifecycle=lifecycle,
            network=network,
            supply=supply_config,
            macro=macro,
            agent_composition=agent_composition,
            price_sensitivity=market_params.get('price_sensitivity', 0.001),
            expectation_weight=market_params.get('expectation_weight', 0.015),
            base_appreciation=market_params.get('base_appreciation', 0.002),
            buy_threshold=market_params.get('buy_threshold', 0.25),
            sell_threshold=market_params.get('sell_threshold', 0.30),
            spillover_rate=market_params.get('spillover_rate', 0.005),
        )

        return config

    def initialize(self):
        """시뮬레이션 초기화"""
        self.sim = Simulation(self.config, arch=self.arch)
        self.sim.initialize()

    async def run_streaming(self, websocket) -> None:
        """매월 결과를 WebSocket으로 스트리밍"""
        if self.sim is None:
            raise RuntimeError("시뮬레이션이 초기화되지 않았습니다.")

        self.is_running = True
        self.should_stop = False

        try:
            for month in range(self.config.num_steps):
                if self.should_stop:
                    await websocket.send_json({
                        'type': 'stopped',
                        'month': month
                    })
                    break

                # 시뮬레이션 스텝 실행
                self.sim.step()

                # 상태 수집 및 전송
                state = self._get_monthly_state(month)
                await websocket.send_json({
                    'type': 'state',
                    'data': state
                })

                # 브라우저 렌더링 시간 확보
                await asyncio.sleep(0.05)

            # 완료 메시지
            if not self.should_stop:
                results = self.sim.get_results()
                await websocket.send_json({
                    'type': 'completed',
                    'summary': self._get_summary(results)
                })

        except Exception as e:
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
            raise
        finally:
            self.is_running = False

    def stop(self):
        """시뮬레이션 중지"""
        self.should_stop = True

    def _get_monthly_state(self, month: int) -> Dict[str, Any]:
        """월별 상태 데이터 수집"""
        sim = self.sim

        # 지역별 가격
        prices = sim.market.region_prices.to_numpy()
        price_changes = sim.market.region_price_changes.to_numpy()

        # 거래량
        transactions = sim.market.transactions.to_numpy()

        # 수요/공급
        demand = sim.market.demand.to_numpy()
        supply = sim.market.supply.to_numpy()

        # 지역별 에이전트 분포
        agent_dist = self._get_agent_distribution()

        # 최근 거래 (지도 표시용)
        recent_trans = self._get_recent_transactions()

        # 지역별 통계 구성
        regions_data = []
        for r in range(NUM_REGIONS):
            regions_data.append({
                'region_id': r,
                'name': REGIONS[r]['name'],
                'price': float(prices[r]),
                'price_change': float(price_changes[r]),
                'transactions': int(transactions[r]),
                'demand': int(demand[r]),
                'supply': int(supply[r]),
                'jeonse_ratio': 0.6,
                'homeless_count': agent_dist[r]['homeless'],
                'one_house_count': agent_dist[r]['one_house'],
                'multi_house_count': agent_dist[r]['multi_house'],
            })

        # 거시경제 지표
        macro_state = sim.macro.get_state_dict()

        # 전체 통계
        stats = sim.stats_history[-1] if sim.stats_history else {}

        return {
            'month': month,
            'year': month // 12 + 1,
            'avg_price': float(np.mean(prices)),
            'total_transactions': int(np.sum(transactions)),
            'homeowner_rate': stats.get('homeowner_rate', 0.55),
            'multi_owner_rate': stats.get('multi_owner_rate', 0.15),
            'demand_supply_ratio': stats.get('demand_supply_ratio', 1.0),
            'listing_rate': stats.get('listing_rate', 0.05),
            'interest_rate': macro_state.get('policy_rate', 0.035),
            'inflation': macro_state.get('inflation', 0.02),
            'gdp_growth': macro_state.get('gdp_growth', 0.025),
            'm2_growth': self.params.get('macro', {}).get('m2_growth', 0.08),
            'mean_building_age': stats.get('mean_building_age', 15),
            'mean_building_condition': stats.get('mean_building_condition', 0.7),
            'demolished_count': stats.get('demolished_houses', 0),
            'regions': regions_data,
            'recent_transactions': recent_trans,
        }

    def _get_agent_distribution(self) -> Dict[int, Dict[str, int]]:
        """지역별 에이전트 분포 계산"""
        owned = self.sim.households.owned_houses.to_numpy()
        regions = self.sim.households.region.to_numpy()

        distribution = {}
        for r in range(NUM_REGIONS):
            region_mask = regions == r
            distribution[r] = {
                'homeless': int(np.sum(region_mask & (owned == 0))),
                'one_house': int(np.sum(region_mask & (owned == 1))),
                'multi_house': int(np.sum(region_mask & (owned >= 2))),
            }

        return distribution

    def _get_recent_transactions(self, limit: int = 50) -> list:
        """최근 거래 내역 (지도 표시용)"""
        transactions = self.sim.market.transactions.to_numpy()
        prices = self.sim.market.region_prices.to_numpy()

        recent = []
        for r in range(NUM_REGIONS):
            if transactions[r] > 0:
                region_info = REGIONS[r]
                coords = self._get_region_coords(r)
                recent.append({
                    'region_id': r,
                    'region_name': region_info['name'],
                    'count': int(transactions[r]),
                    'avg_price': float(prices[r]),
                    'lat': coords['lat'],
                    'lng': coords['lng'],
                })

        return recent

    def _get_region_coords(self, region_id: int) -> Dict[str, float]:
        """지역 중심 좌표 반환"""
        coords = {
            0: {'lat': 37.517, 'lng': 127.047},
            1: {'lat': 37.556, 'lng': 127.010},
            2: {'lat': 37.570, 'lng': 126.977},
            3: {'lat': 37.359, 'lng': 127.105},
            4: {'lat': 37.275, 'lng': 127.009},
            5: {'lat': 37.742, 'lng': 127.047},
            6: {'lat': 37.456, 'lng': 126.705},
            7: {'lat': 35.180, 'lng': 129.076},
            8: {'lat': 35.871, 'lng': 128.602},
            9: {'lat': 35.160, 'lng': 126.851},
            10: {'lat': 36.351, 'lng': 127.385},
            11: {'lat': 36.480, 'lng': 127.289},
            12: {'lat': 35.800, 'lng': 127.800},
        }
        return coords.get(region_id, {'lat': 36.5, 'lng': 127.5})

    def _get_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """시뮬레이션 결과 요약"""
        price_history = results.get('price_history', [])
        stats_history = results.get('stats_history', [])

        if len(price_history) < 2:
            return {'error': '데이터 부족'}

        initial_prices = price_history[0]
        final_prices = price_history[-1]
        total_change = (final_prices - initial_prices) / (initial_prices + 1e-6)

        initial_stats = stats_history[0] if stats_history else {}
        final_stats = stats_history[-1] if stats_history else {}

        return {
            'duration_months': len(price_history),
            'price_change': {
                'gangnam': float(total_change[0]) if len(total_change) > 0 else 0,
                'seoul_avg': float(np.mean(total_change[:3])) if len(total_change) >= 3 else 0,
                'national_avg': float(np.mean(total_change)),
            },
            'homeowner_rate': {
                'initial': initial_stats.get('homeowner_rate', 0),
                'final': final_stats.get('homeowner_rate', 0),
            },
            'total_transactions': sum(s.get('transaction_total', 0) for s in stats_history),
        }


def get_scenario_presets() -> Dict[str, Any]:
    """시나리오 프리셋 목록 반환 (딕셔너리 형태)"""
    return {
        "default": {
            "name": "기본값",
            "description": "시뮬레이션 기본 설정",
            "params": {}
        },
        "korea_reality": {
            "name": "한국 현실",
            "description": "2024년 한국 통계 기반 설정",
            "params": {
                "agent_composition": {
                    "income_median": 350,
                    "income_sigma": 0.55,
                    "asset_median": 6000,
                    "asset_alpha": 1.3,
                    "initial_homeless_rate": 0.44,
                    "initial_one_house_rate": 0.41,
                    "initial_multi_house_rate": 0.15,
                    "age_young_ratio": 0.35,
                    "age_middle_ratio": 0.45,
                    "age_senior_ratio": 0.20,
                    "investor_ratio": 0.18,
                    "speculator_ratio": 0.08,
                }
            }
        },
        "tight_reg": {
            "name": "규제 강화",
            "description": "LTV/DTI 규제 강화 시나리오",
            "params": {
                "policy": {
                    "ltv_1house": 0.40,
                    "ltv_2house": 0.20,
                    "dti_limit": 0.35,
                    "acq_tax_2house": 0.12,
                    "jongbu_rate": 0.03,
                }
            }
        },
        "loose_reg": {
            "name": "규제 완화",
            "description": "LTV/DTI 규제 완화 시나리오",
            "params": {
                "policy": {
                    "ltv_1house": 0.70,
                    "ltv_2house": 0.50,
                    "dti_limit": 0.50,
                    "acq_tax_2house": 0.04,
                    "jongbu_rate": 0.01,
                }
            }
        },
        "supply_crisis": {
            "name": "공급 부족",
            "description": "주택 공급 급감 시나리오",
            "params": {
                "supply": {
                    "base_supply_rate": 0.0003,
                    "elasticity_gangnam": 0.1,
                }
            }
        }
    }
