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

    # 시장 파라미터
    price_sensitivity: float = 0.0003  # 수요/공급에 대한 가격 민감도 (더 낮춤)
    expectation_weight: float = 0.005  # 기대가 가격에 미치는 영향 (더 낮춤)
    base_appreciation: float = 0.002  # 기본 가격 상승률 (월 0.2% ≈ 연 2.4%)

    # 행동 파라미터
    buy_threshold: float = 0.25  # 매수 확률 임계값
    sell_threshold: float = 0.30  # 매도 확률 임계값

    # 풍선효과
    spillover_rate: float = 0.005  # 풍선효과 전파 속도 (더 낮춤)
