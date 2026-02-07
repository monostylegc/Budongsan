"""시뮬레이션 페이즈 정의"""

from enum import IntEnum


class Phase(IntEnum):
    """시뮬레이션 단계 (매월 순서대로 실행)"""
    POLICY_CHECK = 0           # 정책 타임라인 체크
    MACRO_UPDATE = 1           # 거시경제 업데이트
    MONETARY_POLICY = 2        # 통화정책 업데이트
    LABOR_MARKET = 3           # 노동시장 업데이트
    INCOME_DISTRIBUTION = 4    # 소득 배분 (심리적 회계)
    COGNITIVE_PIPELINE = 5     # 인지 파이프라인 (인지→감정→사고→행동)
    PRICE_AGGREGATION = 6      # 가격 집계
    DEMAND_SUPPLY = 7          # 수요/공급 집계
    MARKET_MATCHING = 8        # 시장 매칭 (거래)
    TAX_SETTLEMENT = 9         # 세금 정산
    RENTAL_UPDATE = 10         # 임대시장 업데이트
    SUPPLY_UPDATE = 11         # 공급 업데이트
    DEPRECIATION = 12          # 감가상각
    HOUSING_AFFORDABILITY = 13 # 주거비 체크 (강제매도)
    LIFECYCLE_UPDATE = 14      # 생애주기 업데이트
    PRICE_UPDATE = 15          # 가격 업데이트
    RECORD_STATS = 16          # 통계 기록
    EVENT_PROCESS = 17         # 이벤트 처리


DEFAULT_PHASE_ORDER = list(Phase)
