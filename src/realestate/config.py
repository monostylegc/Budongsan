"""설정 및 상수 정의"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


# 지역 정보
REGIONS = {
    0: {"name": "강남3구", "tier": 1, "base_price": 200000},  # 20억
    1: {"name": "마용성", "tier": 1, "base_price": 150000},   # 15억
    2: {"name": "기타서울", "tier": 2, "base_price": 80000},  # 8억
    3: {"name": "분당판교", "tier": 1, "base_price": 120000}, # 12억
    4: {"name": "경기남부", "tier": 2, "base_price": 60000},  # 6억
    5: {"name": "경기북부", "tier": 3, "base_price": 40000},  # 4억
    6: {"name": "인천", "tier": 2, "base_price": 50000},      # 5억
    7: {"name": "부산", "tier": 2, "base_price": 45000},      # 4.5억
    8: {"name": "대구", "tier": 3, "base_price": 35000},      # 3.5억
    9: {"name": "광주", "tier": 3, "base_price": 30000},      # 3억
    10: {"name": "대전", "tier": 3, "base_price": 32000},     # 3.2억
    11: {"name": "세종", "tier": 2, "base_price": 45000},     # 4.5억
    12: {"name": "기타지방", "tier": 4, "base_price": 20000}, # 2억
}

NUM_REGIONS = len(REGIONS)

# 지역별 심리적 프리미엄 지수 (구조적 개선)
# 학군, 명성, 브랜드 가치 등 비경제적 요인 반영
# 값이 클수록 고자산가가 선호하는 지역
REGION_PRESTIGE = np.array([
    1.0,   # 강남3구 - 최고 프리미엄
    0.9,   # 마용성 - 높은 프리미엄
    0.6,   # 기타서울 - 중간 프리미엄
    0.85,  # 분당판교 - 높은 프리미엄 (학군, IT)
    0.5,   # 경기남부 - 낮은 프리미엄
    0.3,   # 경기북부 - 매우 낮음
    0.4,   # 인천 - 낮음
    0.45,  # 부산 - 지방 최고
    0.35,  # 대구
    0.3,   # 광주
    0.35,  # 대전
    0.5,   # 세종 - 정부 효과
    0.2,   # 기타지방 - 최저
], dtype=np.float32)

# 지역별 일자리 밀도 (고소득 일자리 집중도)
# 값이 클수록 고소득 일자리가 많음 → 주거 수요 증가
# 출처: 통계청 사업체조사 기반 상대적 밀도
REGION_JOB_DENSITY = np.array([
    1.0,   # 강남3구 - 테헤란로, 금융/IT/서비스업 집중
    0.85,  # 마용성 - 성수 IT, 마포 미디어, 용산 재개발
    0.5,   # 기타서울 - 도심, 여의도 등 분산
    0.9,   # 분당판교 - 판교 테크노밸리, IT/게임/바이오
    0.35,  # 경기남부 - 공업단지, 중견기업
    0.2,   # 경기북부 - 일자리 적음
    0.3,   # 인천 - 항만, 공항 관련
    0.25,  # 부산 - 지방 최대, 항만/조선
    0.2,   # 대구 - 섬유/기계
    0.15,  # 광주 - 자동차/광산업
    0.2,   # 대전 - 연구단지 (KAIST, ETRI)
    0.4,   # 세종 - 정부기관 이전
    0.1,   # 기타지방 - 일자리 부족
], dtype=np.float32)

# 지역간 인접도 행렬 (풍선효과 전파에 사용)
# 값이 클수록 영향을 많이 받음
ADJACENCY = np.array([
    # 강남 마용성 기타서울 분당 경기남 경기북 인천 부산 대구 광주 대전 세종 기타
    [1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 강남3구
    [0.8, 1.0, 0.8, 0.5, 0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 마용성
    [0.7, 0.8, 1.0, 0.5, 0.6, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 기타서울
    [0.6, 0.5, 0.5, 1.0, 0.7, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 분당판교
    [0.5, 0.5, 0.6, 0.7, 1.0, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 경기남부
    [0.3, 0.4, 0.5, 0.3, 0.4, 1.0, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 경기북부
    [0.3, 0.3, 0.4, 0.3, 0.5, 0.4, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],  # 인천
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 0.3, 0.2, 0.2, 0.1, 0.2],   # 부산
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 1.0, 0.2, 0.3, 0.2, 0.2],   # 대구
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2],   # 광주
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 1.0, 0.5, 0.2],   # 대전
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.5, 1.0, 0.2],   # 세종
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0],  # 기타지방
], dtype=np.float32)

# 지역별 가구 분포 비율
REGION_HOUSEHOLD_RATIO = np.array([
    0.05,  # 강남3구
    0.04,  # 마용성
    0.10,  # 기타서울
    0.04,  # 분당판교
    0.12,  # 경기남부
    0.08,  # 경기북부
    0.06,  # 인천
    0.07,  # 부산
    0.05,  # 대구
    0.03,  # 광주
    0.03,  # 대전
    0.02,  # 세종
    0.31,  # 기타지방
], dtype=np.float32)


@dataclass
class PolicyConfig:
    """정책 설정"""
    # 대출 규제
    ltv_1house: float = 0.50
    ltv_2house: float = 0.30
    ltv_3house: float = 0.00
    dti_limit: float = 0.40
    dsr_limit: float = 0.40

    # 취득세 (%)
    acq_tax_1house: float = 0.01
    acq_tax_2house: float = 0.08
    acq_tax_3house: float = 0.12

    # 양도세 (%)
    transfer_tax_short: float = 0.70  # 2년 미만
    transfer_tax_long: float = 0.40   # 2년 이상
    transfer_tax_multi_short: float = 0.75
    transfer_tax_multi_long: float = 0.60

    # 종부세
    jongbu_threshold_1house: float = 110000  # 11억
    jongbu_threshold_multi: float = 60000    # 6억
    jongbu_rate: float = 0.02

    # 금리
    interest_rate: float = 0.035
    mortgage_spread: float = 0.015

    # 전세
    jeonse_loan_limit: float = 50000  # 5억
    rent_increase_cap: float = 0.05   # 5%


@dataclass
class BehavioralConfig:
    """행동경제학 파라미터 설정

    참고 문헌:
    - Tversky & Kahneman (1992): 전망이론 (Prospect Theory)
    - Genesove & Mayer (2001, QJE): 부동산 시장 손실 회피 실증
    - Shiller (2005): 비이성적 과열 (Irrational Exuberance)
    - Banerjee (1992): 군집 행동 모델
    """
    # FOMO (Fear Of Missing Out)
    fomo_trigger_threshold: float = 0.05   # FOMO 발동 가격 상승률 (6개월 기준)
    fomo_intensity: float = 50.0           # FOMO 강도 계수

    # 손실 회피 (Loss Aversion)
    loss_aversion_mean: float = 2.5        # 손실 회피 계수 평균 (Genesove & Mayer: 2.5)
    loss_aversion_std: float = 0.35        # 손실 회피 계수 표준편차
    loss_aversion_decay: float = 5.0       # 손실에 따른 매도 확률 감소 속도

    # 앵커링 (Anchoring)
    anchoring_threshold: float = 0.1       # 앵커링 발동 이익률 (매입가 대비)
    anchoring_penalty: float = 0.5         # 앵커링 페널티 강도

    # 군집 행동 (Herding)
    herding_trigger: float = 0.03          # 군집 발동 매수 비율
    herding_intensity: float = 10.0        # 군집 강도 계수

    # 사회적 학습
    social_learning_rate: float = 0.1      # 사회적 학습 속도
    news_impact: float = 0.2               # 뉴스/미디어 영향도


@dataclass
class LifeCycleConfig:
    """생애주기 파라미터 설정"""
    # 결혼
    marriage_urgency_age_start: int = 28   # 결혼 주거 압박 시작 나이
    marriage_urgency_age_end: int = 35     # 결혼 주거 압박 종료 나이
    newlywed_housing_pressure: float = 1.5 # 신혼 주거 압박 배율

    # 출산/육아
    parenting_housing_pressure: float = 1.3 # 육아기 주거 압박 배율

    # 학군
    school_transition_age_start: int = 10  # 학군 이동 시작 자녀 나이
    school_transition_age_end: int = 15    # 학군 이동 종료 자녀 나이
    school_district_premium: float = 1.2   # 학군 지역 선호 배율

    # 은퇴
    retirement_start_age: int = 55         # 은퇴 고려 시작 나이
    downsizing_probability: float = 0.1    # 연간 다운사이징 확률


@dataclass
class ProspectTheoryConfig:
    """전망이론 파라미터 설정

    참고 문헌:
    - Tversky & Kahneman (1992): 전망이론 가치 함수
    - Prelec (1998): 확률 가중 함수
    - Genesove & Mayer (2001, QJE): 부동산 손실 회피 실증
    """
    # 가치 함수 파라미터 (Tversky & Kahneman, 1992)
    alpha: float = 0.88               # 이득 곡률 (diminishing sensitivity)
    beta: float = 0.88                # 손실 곡률
    lambda_general: float = 2.25      # 일반 손실 회피 계수
    lambda_realestate: float = 2.5    # 부동산 손실 회피 계수 (Genesove & Mayer)

    # 확률 가중 함수 파라미터 (Prelec, 1998)
    gamma_gain: float = 0.61          # 이득 확률 가중 파라미터
    gamma_loss: float = 0.69          # 손실 확률 가중 파라미터

    # 참조점 설정
    # 수정 (2024): 0.02 → 0.008로 낮춤
    # Genesove & Mayer (2001) 실증 연구에 따르면 손실 회피가 매우 오래 지속됨
    # 월 0.8%면 약 7년 후 참조점이 현재가로 수렴 (기존 3년에서 연장)
    reference_point_decay: float = 0.008  # 월간 참조점 적응 속도
    use_purchase_price_as_reference: bool = True  # 매입가를 참조점으로 사용


@dataclass
class DiscountingConfig:
    """시간 할인 파라미터 설정

    참고 문헌:
    - Laibson (1997): β-δ 준쌍곡선 할인 모델
    """
    # β-δ 모델 파라미터
    beta: float = 0.7                 # 현재 편향 (present bias)
    delta: float = 0.99               # 월간 기하 할인율 (연 ~11% 할인율)

    # 투자 기간 설정
    investment_horizon: int = 60      # 기대 투자 기간 (월, 5년)

    # 할인율 이질성
    beta_mean: float = 0.7
    beta_std: float = 0.1
    delta_mean: float = 0.99
    delta_std: float = 0.005


@dataclass
class SupplyConfig:
    """주택 공급 파라미터 설정

    참고 문헌:
    - Saiz (2010): 공급 탄력성
    """
    # 기본 공급률
    base_supply_rate: float = 0.001   # 월간 기본 공급률 (0.1%)
    price_threshold: float = 0.05     # 공급 반응 가격 변화 임계값 (5%)

    # 지역별 공급 탄력성 (토지 가용성 반영)
    elasticity_gangnam: float = 0.3   # 강남 (토지 희소)
    elasticity_seoul: float = 0.5     # 기타 서울 (재개발 잠재력)
    elasticity_gyeonggi: float = 1.5  # 경기 (가용 토지)
    elasticity_local: float = 2.0     # 지방 (풍부한 토지)

    # 재건축/재개발
    redevelopment_base_prob: float = 0.001  # 월간 기본 재건축 확률
    redevelopment_age_threshold: int = 30   # 재건축 가능 건물 연식
    redevelopment_price_threshold: float = 0.30  # 5년간 가격 상승률 조건
    construction_period: int = 24           # 건설 기간 (월)

    # 공급 파이프라인 제한
    max_construction_ratio: float = 0.02    # 최대 동시 건설 비율


@dataclass
class MacroConfig:
    """거시경제 파라미터 설정

    참고 문헌:
    - Taylor (1993): Taylor Rule
    """
    # Taylor Rule 파라미터
    neutral_real_rate: float = 0.02   # 중립 실질금리 (r*)
    inflation_target: float = 0.02    # 인플레이션 목표 (π*)
    alpha_inflation: float = 1.5      # 인플레이션 반응 계수
    alpha_output: float = 0.5         # 산출갭 반응 계수

    # GDP 성장 모델 (AR(1))
    gdp_growth_mean: float = 0.025    # 평균 GDP 성장률 (연 2.5%)
    gdp_growth_persistence: float = 0.8  # AR(1) 계수 (ρ)
    gdp_growth_volatility: float = 0.01  # 충격 표준편차 (σ)

    # 소득/자산 연동
    income_gdp_beta: float = 0.8      # 소득의 GDP 탄력성
    credit_spread: float = 0.015      # 신용 스프레드

    # 초기 상태
    initial_inflation: float = 0.02   # 초기 인플레이션
    initial_gdp_growth: float = 0.025 # 초기 GDP 성장률


@dataclass
class NetworkConfig:
    """사회적 네트워크 파라미터 설정

    참고 문헌:
    - Watts & Strogatz (1998): Small-World 네트워크
    - DeGroot (1974): 신념 업데이트 모델
    """
    # Small-World 네트워크 구조
    avg_neighbors: int = 10           # 평균 이웃 수
    rewiring_prob: float = 0.1        # 재연결 확률
    max_neighbors: int = 20           # 최대 이웃 수

    # DeGroot Learning 파라미터
    self_weight: float = 0.6          # 자기 신호 가중치
    neighbor_weight: float = 0.4      # 이웃 신호 가중치

    # 정보 캐스케이드
    cascade_threshold: float = 0.3    # 캐스케이드 발동 이웃 매수 비율
    cascade_multiplier: float = 2.0   # 캐스케이드 매수 확률 배율

    # 네트워크 업데이트
    belief_update_frequency: int = 1  # 신념 업데이트 주기 (월)


@dataclass
class AgentCompositionConfig:
    """에이전트 구성 파라미터 설정"""
    # 에이전트 유형 비율
    investor_ratio: float = 0.15      # 투자자 비율 (임대 수익 목적)
    speculator_ratio: float = 0.05    # 투기자 비율 (단기 시세차익 목적)

    # 투기자 특성 배율
    speculator_risk_multiplier: float = 1.5   # 위험 허용도 배율
    speculator_fomo_multiplier: float = 1.3   # FOMO 민감도 배율
    speculator_horizon_min: int = 6           # 최소 보유 기간 (월)
    speculator_horizon_max: int = 24          # 최대 보유 기간 (월)

    # 초기 주택 보유 분포
    initial_homeless_rate: float = 0.45       # 무주택자 비율
    initial_one_house_rate: float = 0.40      # 1주택자 비율
    initial_multi_house_rate: float = 0.15    # 다주택자 비율

    # 소득 분포 (로그정규 분포)
    income_median: float = 300.0      # 월소득 중위값 (만원)
    income_sigma: float = 0.6         # 로그 표준편차 (분산도)

    # 자산 분포 (파레토 분포)
    asset_median: float = 5000.0      # 순자산 중위값 (만원)
    asset_alpha: float = 1.5          # 파레토 알파 (낮을수록 불평등)

    # 연령 분포
    age_young_ratio: float = 0.45     # 청년층 (25-34세) 비율
    age_middle_ratio: float = 0.43    # 중년층 (35-54세) 비율
    age_senior_ratio: float = 0.12    # 장년층 (55세+) 비율


@dataclass
class Config:
    """시뮬레이션 전체 설정"""
    # 규모
    num_households: int = 1_000_000
    num_houses: int = 500_000

    # 시간
    num_steps: int = 120  # 10년 (월 단위)

    # 초기화 시드
    seed: int = 42

    # 정책
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    # 행동경제학 설정
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)

    # 생애주기 설정
    lifecycle: LifeCycleConfig = field(default_factory=LifeCycleConfig)

    # 전망이론 설정
    prospect_theory: ProspectTheoryConfig = field(default_factory=ProspectTheoryConfig)

    # 시간 할인 설정
    discounting: DiscountingConfig = field(default_factory=DiscountingConfig)

    # 공급 설정
    supply: SupplyConfig = field(default_factory=SupplyConfig)

    # 거시경제 설정
    macro: MacroConfig = field(default_factory=MacroConfig)

    # 네트워크 설정
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # 에이전트 구성 설정
    agent_composition: AgentCompositionConfig = field(default_factory=AgentCompositionConfig)

    # 시장 파라미터
    # 수정 (2024): 수요/공급 불균형이 가격에 더 강하게 반영되도록 조정
    price_sensitivity: float = 0.008  # 수요/공급에 대한 가격 민감도 (0.001 → 0.008)
    expectation_weight: float = 0.010  # 기대가 가격에 미치는 영향 (0.015 → 0.010, 축소)
    base_appreciation: float = 0.0015  # 기본 가격 상승률 (월 0.15% ≈ 연 1.8%, 축소)

    # 행동 파라미터
    # 수정 (2024): 매수/매도 점수가 덧셈 기반으로 변경
    # urgency 값도 하향 조정되어 전체 점수 스케일 축소
    buy_threshold: float = 0.40  # 매수 확률 임계값
    sell_threshold: float = 0.08  # 매도 확률 임계값

    # 풍선효과
    spillover_rate: float = 0.005  # 풍선효과 전파 속도 (더 낮춤)
