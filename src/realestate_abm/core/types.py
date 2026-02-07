"""공통 타입 및 열거형"""

from enum import IntEnum
from typing import TypeAlias
import numpy as np

# Array type aliases
ArrayF32: TypeAlias = np.ndarray  # float32 array
ArrayI32: TypeAlias = np.ndarray  # int32 array
ArrayBool: TypeAlias = np.ndarray  # bool array

class AgentType(IntEnum):
    END_USER = 0      # 실수요자
    INVESTOR = 1      # 투자자
    SPECULATOR = 2    # 투기자

class EmploymentStatus(IntEnum):
    EMPLOYED = 0
    UNEMPLOYED = 1
    ON_INSURANCE = 2  # 실업급여 수령중

class LifeStage(IntEnum):
    SINGLE = 0        # 미혼
    NEWLYWED = 1      # 신혼
    PARENTING = 2     # 육아기
    SCHOOL_AGE = 3    # 학령기
    EMPTY_NEST = 4    # 빈둥지
    RETIRED = 5       # 은퇴기

class TenureType(IntEnum):
    OWNER_OCCUPIED = 0
    JEONSE = 1        # 전세
    WOLSE = 2         # 월세

class RegionTier(IntEnum):
    PREMIUM = 1       # 프리미엄 (강남, 분당)
    STANDARD = 2      # 표준 (기타서울, 인천)
    SUBURBAN = 3      # 외곽 (경기북부, 대구)
    RURAL = 4         # 지방

class DecisionType(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2
    UPGRADE = 3       # 갈아타기 (sell+buy)

class EmotionType(IntEnum):
    ANXIETY = 0       # 불안
    SATISFACTION = 1  # 만족
    FOMO = 2          # FOMO
    REGRET = 3        # 후회
