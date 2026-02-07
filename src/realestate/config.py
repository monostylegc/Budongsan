"""설정 및 상수 정의"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


# 지역 정보 (2024-2025 실거래가 기준, 국민평형 84㎡ 기준)
# 데이터 출처: KB부동산, 한국부동산원, 스마트에프엔 (2024-2025)
REGIONS = {
    0: {"name": "강남3구", "tier": 1, "base_price": 300000},  # 30억 (평당 1.2억 x 25평)
    1: {"name": "마용성", "tier": 1, "base_price": 200000},   # 20억 (한강벨트)
    2: {"name": "기타서울", "tier": 2, "base_price": 120000}, # 12억 (서울 평균 15억 고려)
    3: {"name": "분당판교", "tier": 1, "base_price": 150000}, # 15억 (판교테크노밸리)
    4: {"name": "경기남부", "tier": 2, "base_price": 70000},  # 7억 (수원/화성/평택)
    5: {"name": "경기북부", "tier": 3, "base_price": 45000},  # 4.5억 (의정부/양주/파주)
    6: {"name": "인천", "tier": 2, "base_price": 50000},      # 5억 (실거래 4.9억)
    7: {"name": "부산", "tier": 2, "base_price": 50000},      # 5억 (실거래 5억 돌파)
    8: {"name": "대구", "tier": 3, "base_price": 47000},      # 4.7억 (실거래 4.68억)
    9: {"name": "광주", "tier": 3, "base_price": 38000},      # 3.8억 (실거래 3.79억)
    10: {"name": "대전", "tier": 3, "base_price": 44000},     # 4.4억 (실거래 4.37억)
    11: {"name": "세종", "tier": 2, "base_price": 50000},     # 5억 (정부세종청사)
    12: {"name": "기타지방", "tier": 4, "base_price": 25000}, # 2.5억 (울산 4.3억 등 평균)
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

# =============================================================================
# 지역별 일자리 밀도 (고소득 일자리 집중도)
# =============================================================================
#
# 정의: 경제활동인구 1,000명당 고소득(연 5,000만원+) 일자리 수의 상대적 비율
# 최대값(강남3구)을 1.0으로 정규화
#
# 데이터 출처:
# - 통계청 「지역별 고용조사」(2023)
# - 통계청 「전국사업체조사」(2022)
# - 국토교통부 「주거실태조사」(2022)
#
# 계산 방법:
# 1. 지역별 사업체 종사자 수 (전국사업체조사)
# 2. 지역별 고소득 업종 비율 (금융/IT/전문서비스/공공)
# 3. 지역별 평균 임금 수준 (지역별 고용조사)
# 4. 상대적 밀도 = (종사자수 × 고소득비율 × 임금수준) / 지역면적
# 5. 정규화: 강남3구 = 1.0 기준
#
# 검증: 2023년 기준 강남구 평균 소득 약 7,200만원 (전국 4,100만원 대비 1.76배)
#       강남3구 사업체 밀도: 서울 평균의 약 2배
#
# 참고 문헌:
# - 손승영 외 (2021), "수도권 직주근접 변화와 주거이동", 국토연구원
# - 이창무 외 (2019), "일자리 접근성이 주택가격에 미치는 영향", 부동산학연구
# =============================================================================
REGION_JOB_DENSITY = np.array([
    1.00,  # 강남3구 - 테헤란로 IT/금융, 역삼/선릉 스타트업, 압구정/청담 서비스업
    0.85,  # 마용성 - 성수 IT/소호, 마포 미디어/출판, 용산 재개발 기대
    0.50,  # 기타서울 - 도심(종로/중구), 여의도 금융, 구로/가산 IT단지 (분산)
    0.90,  # 분당판교 - 판교테크노밸리 IT/게임/바이오, 분당 금융/서비스
    0.35,  # 경기남부 - 수원/화성 제조업, 평택 반도체 (고소득 비율 낮음)
    0.20,  # 경기북부 - 의정부/양주/파주, 제조업 중심 (고소득 일자리 적음)
    0.30,  # 인천 - 송도 바이오/IT, 항만/공항 물류 (지역 내 편차 큼)
    0.25,  # 부산 - 해운대/서면 서비스업, 항만/조선 (지방 최대)
    0.20,  # 대구 - 섬유/기계 제조업 중심 (고소득 비율 낮음)
    0.15,  # 광주 - 자동차/가전 제조업, 광산업 (고소득 비율 낮음)
    0.20,  # 대전 - 대덕연구단지 (KAIST, ETRI), 정부출연연 (한정적)
    0.40,  # 세종 - 중앙부처 이전, 공무원/공공기관 (안정적 고소득)
    0.10,  # 기타지방 - 농어촌/소도시, 일자리 부족 (인구 유출 지역)
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
class DynamicPrestigeConfig:
    """동적 프리미엄 파라미터 설정

    프리미엄 = 구조적(REGION_PRESTIGE, 고정) + 동적(매 스텝 업데이트)
    동적 요소: 학군, 명성 모멘텀, 고소득 집중도
    """
    # 학군 프리미엄 (학령기 자녀 밀집도 기반)
    school_premium_weight: float = 2.0       # 학군 영향 가중치
    school_age_min: int = 6                  # 학령기 최소 나이
    school_age_max: int = 18                 # 학령기 최대 나이

    # 명성 모멘텀 (가격 추세 기반, "강남 불패" 심리)
    momentum_decay: float = 0.95             # 모멘텀 감쇠율 (높을수록 오래 지속)
    momentum_sensitivity: float = 5.0        # 가격 변화 → 모멘텀 변환 계수

    # 고소득 집중도 (젠트리피케이션)
    concentration_weight: float = 0.15       # 고소득 집중 영향 가중치
    high_income_percentile: float = 80.0     # 고소득 기준 백분위

    # 클리핑 범위
    school_premium_min: float = -0.1
    school_premium_max: float = 0.2
    momentum_premium_min: float = -0.15
    momentum_premium_max: float = 0.15
    concentration_premium_min: float = -0.1
    concentration_premium_max: float = 0.15


@dataclass
class AffordabilityConfig:
    """affordability (구매력) 설정 - DSR 기반 통일 체계

    DSR(Debt Service Ratio) = 연간 원리금 상환액 / 연간 소득
    한국 규제: DSR 40% 제한

    에이전트 유형별 리스크 선호도에 따라 차등 적용:
    - 실수요자: 보수적 (DSR 35%)
    - 투자자: 중간 (DSR 40%)
    - 투기자: 공격적 (DSR 45%, 규제 초과 감수)

    고자산가 정의 (백분위수 기반 통일):
    - 기준: 자산 상위 20% (시뮬레이션 시점 동적 계산)
    - 근거: 통계청 가계금융복지조사(2023) 기준 순자산 상위 20% = 약 7억원
    - 특성: 자산 활용 비율 70% (일반 50%), 프리미엄 지역 선호
    - 주의: 절대값(예: 1억)이 아닌 백분위수로 정의하여 자산 분포 변화에 적응
    """
    # DSR 한도 (에이전트 유형별)
    dsr_limit_end_user: float = 0.35      # 실수요자: 보수적
    dsr_limit_investor: float = 0.40      # 투자자: 규제 기준
    dsr_limit_speculator: float = 0.45    # 투기자: 공격적

    # ==========================================================================
    # 고자산가 설정 (상위 10% 기준)
    # ==========================================================================
    # 데이터 출처: 통계청 2024 가계금융복지조사
    # - 상위 10% 순자산: 10억 5000만원 이상 (전체 자산의 44.4% 점유)
    # - 상위 1% 순자산: 33억원 이상
    #
    # 고자산가 판별: np.percentile(assets, 90)로 동적 계산 (simulation.py)
    # 절대값이 아닌 백분위수 사용 이유:
    # 1. 자산 분포가 시뮬레이션 중 변화함
    # 2. 초기 설정(파레토 분포)에 의존하지 않음
    # 3. 인플레이션/자산 증가에 자동 적응
    wealthy_asset_utilization: float = 0.7   # 고자산가: 자산의 70% 활용 가능
    normal_asset_utilization: float = 0.5    # 유주택자: 자산의 50% 활용 (여유 자금만)
    homeless_asset_utilization: float = 0.85  # 무주택자: 자산의 85% 활용 (첫 집 마련에 올인)

    # ================================================================
    # 부모 지원 설정 (2024년 실제 데이터 기반)
    # ================================================================
    # 데이터 출처: KB부동산 조사, 2024년 주담대 차주 분석
    # - 한국 30대 주택 구입자의 61%가 부모 지원(증여/차용) 수반
    # - 수도권 주택 구입 시 평균 증여액: 2~4억원
    # - 2024년 기준 증여세 면제 한도: 5천만원 (결혼 시 3.2억까지)
    parent_support_rate: float = 0.6         # 부모 지원 받는 비율 (60%)
    parent_support_mean: float = 25000.0     # 평균 지원액 (2.5억, 만원 단위)
    parent_support_std: float = 12000.0      # 표준편차 (1.2억)
    parent_support_age_max: int = 40         # 지원 대상 최대 연령 (40세까지)

    # [DEPRECATED] 영끌 허용 여부 - JobMarket 모듈 도입으로 불필요
    # 소득이 지역×산업 기반이므로 고소득 지역은 자연스럽게 DSR이 낮아짐
    allow_stretched_dsr: bool = False
    stretched_dsr_multiplier: float = 1.0    # 비활성화

    # 대출 상환 기간 (DSR 계산용)
    loan_term_years: int = 30                # 30년 원리금균등상환 가정


@dataclass
class PolicyConfig:
    """정책 설정 (2024-2025 규제 기준)

    데이터 출처: 금융위원회 보도자료, 토스뱅크 대출규제 가이드
    """
    # 대출 규제 (수도권/규제지역 기준)
    # 생애최초: 수도권 70%, 비수도권 80%
    # 2주택 이상 수도권/규제지역: LTV 0%
    ltv_first_time: float = 0.70  # 생애최초 주택구입자 LTV (무주택자)
    ltv_1house: float = 0.50      # 1주택자 추가 매수 시 LTV
    ltv_2house: float = 0.00      # 2주택자 (수도권/규제지역 LTV 0% 적용)
    ltv_3house: float = 0.00      # 3주택 이상
    dti_limit: float = 0.40       # DTI 한도
    dsr_limit: float = 0.40       # DSR 한도 (은행권 40%, 비은행권 50%)

    # affordability 설정
    affordability: AffordabilityConfig = field(default_factory=AffordabilityConfig)

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


# =============================================================================
# 산업 및 일자리 시장 설정
# =============================================================================
# 산업 유형 (5개)
NUM_INDUSTRIES = 5

INDUSTRY_NAMES = {
    0: "IT/금융",
    1: "전문서비스",
    2: "제조업",
    3: "서비스업",
    4: "공공/교육",
}

# 지역별 산업 구성 비율 (행=지역, 열=산업)
# 데이터 출처: 통계청 전국사업체조사(2022), 지역별 고용조사(2023)
# 각 행의 합이 1.0
REGION_INDUSTRY_MIX = np.array([
    # IT/금융  전문서비스  제조업  서비스업  공공/교육
    [0.40, 0.25, 0.05, 0.20, 0.10],  # 강남3구 - 테헤란로 IT, 역삼 금융
    [0.30, 0.20, 0.05, 0.35, 0.10],  # 마용성 - 성수 IT, 마포 미디어
    [0.15, 0.15, 0.10, 0.40, 0.20],  # 기타서울 - 종로 공공, 구로 IT
    [0.45, 0.20, 0.05, 0.15, 0.15],  # 분당판교 - 판교테크노밸리
    [0.10, 0.10, 0.40, 0.25, 0.15],  # 경기남부 - 수원/화성 반도체
    [0.05, 0.05, 0.45, 0.30, 0.15],  # 경기북부 - 의정부 제조업
    [0.10, 0.10, 0.25, 0.35, 0.20],  # 인천 - 송도 바이오, 항만
    [0.08, 0.10, 0.20, 0.40, 0.22],  # 부산 - 항만/조선, 서비스
    [0.05, 0.08, 0.35, 0.35, 0.17],  # 대구 - 섬유/기계
    [0.04, 0.06, 0.30, 0.40, 0.20],  # 광주 - 자동차/가전
    [0.10, 0.10, 0.15, 0.30, 0.35],  # 대전 - KAIST, 대덕연구단지
    [0.05, 0.05, 0.10, 0.20, 0.60],  # 세종 - 중앙부처 공무원
    [0.03, 0.05, 0.35, 0.40, 0.17],  # 기타지방 - 제조/서비스
], dtype=np.float32)

# 산업별 기본 월소득 배율 (전국 중위소득 400만원 대비)
# 데이터 출처: 통계청 임금근로일자리 소득 분위별 현황(2023)
INDUSTRY_INCOME_MULTIPLIER = np.array([
    1.80,  # IT/금융: 월 ~720만원
    1.40,  # 전문서비스: 월 ~560만원
    1.00,  # 제조업: 월 ~400만원
    0.70,  # 서비스업: 월 ~280만원
    1.20,  # 공공/교육: 월 ~480만원
], dtype=np.float32)

# 지역별 소득 프리미엄 (수도권 프리미엄 반영)
# 데이터 출처: 통계청 지역별 고용조사(2023), 강남구 평균 소득 전국 대비 1.76배
REGION_INCOME_PREMIUM = np.array([
    1.50,  # 강남3구 - 전국 대비 1.5배
    1.30,  # 마용성
    1.10,  # 기타서울
    1.40,  # 분당판교
    1.00,  # 경기남부
    0.90,  # 경기북부
    0.95,  # 인천
    0.90,  # 부산
    0.85,  # 대구
    0.80,  # 광주
    0.90,  # 대전
    1.05,  # 세종 (공무원 수당)
    0.75,  # 기타지방
], dtype=np.float32)

# 산업별 GDP 탄력성 (경기 변동에 대한 고용 민감도)
# IT/금융은 호황에 크게 반응, 공공은 거의 불변
INDUSTRY_GDP_SENSITIVITY = np.array([
    1.50,  # IT/금융 - 경기에 매우 민감
    1.20,  # 전문서비스 - 민감
    1.00,  # 제조업 - 보통
    0.80,  # 서비스업 - 덜 민감
    0.20,  # 공공/교육 - 거의 불변
], dtype=np.float32)

# 산업별 기본 실업률 (안정기 기준)
# 데이터 출처: 통계청 경제활동인구조사(2023) - 산업별 실업률
INDUSTRY_BASE_UNEMPLOYMENT = np.array([
    0.03,  # IT/금융: 3%
    0.04,  # 전문서비스: 4%
    0.05,  # 제조업: 5%
    0.06,  # 서비스업: 6%
    0.01,  # 공공/교육: 1%
], dtype=np.float32)


@dataclass
class JobMarketConfig:
    """일자리 시장 설정

    데이터 출처:
    - 고용노동부 고용보험통계(2023): 실업급여 지급 현황
    - 통계청 경제활동인구조사(2023): 산업별 취업/실업
    """
    # 실업급여 설정 (2024년 한국 기준)
    unemployment_insurance_rate: float = 0.60     # 기존 소득의 60%
    unemployment_insurance_months: int = 6        # 최대 6개월

    # 일자리 생성/파괴 (월간)
    base_job_creation_rate: float = 0.02          # 월간 기본 재취업 가능률
    base_job_destruction_rate: float = 0.015      # 월간 기본 실직률

    # 재취업 확률
    reemployment_base_prob: float = 0.15          # 월간 기본 재취업 확률
    reemployment_age_penalty: float = 0.005       # 나이 1세당 재취업 확률 감소

    # 강제 매도 설정
    forced_sale_months: int = 12                  # 주거비 연속 미납 n개월 시 강제매도
    min_living_cost: float = 200.0                # 최소 생활비 (만원/월)

    # 소득 성장
    income_growth_employed: float = 0.003         # 취업자 월간 소득 성장률 (연 ~3.6%)


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

    # 초기 주택 보유 분포 (2024년 주택소유통계 기준)
    # 데이터 출처: 국가데이터처 2024년 주택소유통계
    initial_homeless_rate: float = 0.43       # 무주택자 비율 (실제 43.1%)
    initial_one_house_rate: float = 0.42      # 1주택자 비율 (56.9% * 74%)
    initial_multi_house_rate: float = 0.15    # 다주택자 비율 (56.9% * 26%)

    # 소득 분포 (로그정규 분포)
    # 데이터 출처: 통계청 2024 Q4 가계동향조사
    # 가구 평균 월소득 522만원, 중위값은 약 400만원으로 추정
    income_median: float = 400.0      # 월소득 중위값 (만원)
    income_sigma: float = 0.65        # 로그 표준편차 (분산도, 상위 10% 약 1000만원)

    # 자산 분포 (파레토 분포)
    # 데이터 출처: 통계청 2024 가계금융복지조사
    # 순자산 평균 4.5억, 상위 10% 10.5억+, 상위 1% 33억+
    # 상위 10%가 전체의 44.4% 보유 → 파레토 알파 약 1.16
    asset_median: float = 30000.0     # 순자산 중위값 (3억, 만원 단위) - 평균 4.5억보다 낮음
    asset_alpha: float = 1.16         # 파레토 알파 (실제 자산 불평등 반영)

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

    # 일자리 시장 설정
    job_market: JobMarketConfig = field(default_factory=JobMarketConfig)

    # 동적 프리미엄 설정
    dynamic_prestige: DynamicPrestigeConfig = field(default_factory=DynamicPrestigeConfig)

    # 시장 파라미터
    # 수정 (2024): 수요/공급 불균형이 가격에 더 강하게 반영되도록 조정
    price_sensitivity: float = 0.008  # 수요/공급에 대한 가격 민감도 (0.001 → 0.008)
    expectation_weight: float = 0.010  # 기대가 가격에 미치는 영향 (0.015 → 0.010, 축소)
    base_appreciation: float = 0.0015  # 기본 가격 상승률 (월 0.15% ≈ 연 1.8%, 축소)

    # 행동 파라미터
    # 수정 (2024): 매수/매도 점수가 덧셈 기반으로 변경
    # urgency 값도 하향 조정되어 전체 점수 스케일 축소
    buy_threshold: float = 0.08  # 매수 확률 임계값 (0.40 → 0.08 하향, urgency 상향과 맞춤)
    sell_threshold: float = 0.12  # 매도 확률 임계값 (0.08 → 0.12 상향)

    # 풍선효과
    spillover_rate: float = 0.005  # 풍선효과 전파 속도 (더 낮춤)
