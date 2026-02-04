# 행동경제학 기반 에이전트 행동 모델

## 개요

본 문서는 한국 부동산 시장 ABM(Agent-Based Model) 시뮬레이션의 에이전트 행동 모델을 설명합니다.
기존의 단순한 점수 기반 선형 계산에서 벗어나, 인간의 실제 의사결정 패턴을 반영한 행동경제학 모델을 적용했습니다.

## 1. 심리적 요인

### 1.1 FOMO (Fear Of Missing Out)

**개념**: 주변 가격이 상승할 때 매수 욕구가 비선형적으로 급증하는 현상

**구현**:
```python
if price_trend > 0.05:  # 6개월간 5% 이상 상승
    excess_rise = price_trend - 0.05
    fomo_factor = 1.0 + fomo_sensitivity * excess_rise ** 2 * 50.0
    fomo_factor = min(fomo_factor, 3.0)  # 최대 3배
```

**효과**:
- 상승장에서 매수 수요 급증
- 가격 상승 → 매수 욕구 증가 → 추가 가격 상승의 피드백 루프
- 버블 형성 메커니즘 재현

**참고 문헌**:
- Shiller, R.J. (2005). "Irrational Exuberance"
- Case, K.E. & Shiller, R.J. (2003). "Is There a Bubble in the Housing Market?"

### 1.2 손실 회피 (Loss Aversion)

**개념**: 손실의 고통이 동일한 크기의 이익의 기쁨보다 크게 느껴지는 현상
- Kahneman & Tversky의 전망이론에서 손실 회피 계수(λ) ≈ 2.25
- 부동산 시장에서는 더 강하게 나타남 (Genesove & Mayer: λ ≈ 2.5)

**구현**:
```python
if gain_loss_ratio < 0:  # 손실 상태
    loss_ratio = -gain_loss_ratio
    loss_aversion_factor = exp(-loss_aversion_coef * loss_ratio * 5.0)
    # 예: 10% 손실 시 매도 확률 약 30%로 감소
```

**효과**:
- 하락장에서 매물 잠김 (Lock-in Effect)
- 거래 절벽 현상 재현
- "손해보고 못 팔아요" 심리 반영

**참고 문헌**:
- Kahneman, D. & Tversky, A. (1979). "Prospect Theory"
- Genesove, D. & Mayer, C. (2001). "Loss Aversion and Seller Behavior: Evidence from the Housing Market" (QJE)

### 1.3 앵커링 (Anchoring)

**개념**: 최초 정보(매입가)에 집착하여 그 기준으로 판단하는 경향

**구현**:
```python
if gain_loss_ratio < 0.1:  # 10% 미만 수익
    anchoring_penalty = (0.1 - gain_loss_ratio) * 0.5
    sell_score -= anchoring_penalty
```

**효과**:
- 매입가 근처에서 매도 꺼림
- "본전" 심리 반영
- 가격 하방 경직성

## 2. 사회적 영향

### 2.1 군집 행동 (Herding)

**개념**: 주변 사람들의 행동을 모방하는 경향

**구현**:
```python
region_buying = region_buy_rate[region]
if region_buying > 0.03:  # 지역 내 3% 이상이 매수 시도
    herding_factor = 1.0 + herding_tendency * (region_buying - 0.03) * 10.0
    herding_factor = min(herding_factor, 2.0)
```

**효과**:
- "친구도 샀는데 나도 사야지"
- 특정 지역 쏠림 현상
- 집단적 의사결정 편향

**참고 문헌**:
- Banerjee, A.V. (1992). "A Simple Model of Herd Behavior" (QJE)

### 2.2 사회적 비교

**개념**: 동료, 친척 등과의 비교를 통한 주거 욕구 형성

**구현**:
- 지역 내 매수 비율 관측
- 6개월 가격 추세 관측
- 적응적 기대 형성

## 3. 생애주기 요인

### 3.1 생애주기 단계

| 단계 | 코드 | 설명 | 주거 특성 |
|------|------|------|-----------|
| 미혼 | 0 | 결혼 전 | 1인 가구, 소형 |
| 신혼 | 1 | 결혼 후 자녀 전 | 내집마련 압박 최대 |
| 육아기 | 2 | 자녀 0-6세 | 넓은 집 필요 |
| 학령기 | 3 | 자녀 7-18세 | 학군 중요 |
| 빈둥지 | 4 | 자녀 독립 | 다운사이징 고려 |
| 은퇴기 | 5 | 55세 이상 | 현금화/소형 이동 |

### 3.2 결혼 (28-35세)

```python
if life_stage == 1:  # 신혼
    urgency += 0.4  # 신혼집 마련 압박 최대
    life_stage_factor = 1.5
```

### 3.3 육아/출산 (30-40세)

```python
if life_stage == 2:  # 육아기
    urgency += 0.35  # 넓은 집 필요
    life_stage_factor = 1.3
```

### 3.4 학군 (자녀 10-18세)

```python
if life_stage == 3:  # 학령기
    urgency += 0.3  # 학군 이동 수요
    if 10 <= eldest_child <= 15:  # 중학교 입학 전후
        urgency += 0.15
```

### 3.5 은퇴 (55-65세)

```python
if life_stage == 5:  # 은퇴기
    if age >= 60:
        sell_score = 0.1
        if gain_loss_ratio > 0.5:  # 50% 이상 이익 시
            sell_score += 0.15
```

## 4. 제한된 합리성

### 4.1 불완전한 정보

- 지역 가격만 대략적으로 인지
- 전체 시장 동향 파악 제한
- 6개월 가격 추세 기준 판단

### 4.2 계산 능력 한계

- 복잡한 수익률 계산 불가
- 단순 휴리스틱 사용
- 감정적 판단 비중 높음

### 4.3 비대칭적 정보 처리

```python
if observed_change > 0:
    # 상승장: 빠르게 기대 조정 (FOMO 효과)
    adaptation_rate = 0.4 + fomo_sensitivity * 0.2
else:
    # 하락장: 느리게 기대 조정 (확증 편향)
    adaptation_rate = 0.2
```

## 5. 예상 시뮬레이션 결과

### 5.1 상승장

- FOMO로 인한 매수 수요 급증
- 거래량 증가
- 가격 추가 상승 (피드백 루프)
- 신혼/육아 가구의 패닉 바잉

### 5.2 하락장

- 손실 회피로 매물 잠김
- 거래 절벽 (극심한 거래량 감소)
- 가격 하방 경직성
- 앵커링으로 호가 유지

### 5.3 정책 변화

- 단기 과잉 반응
- 이후 적응
- 군집 행동에 의한 변동성

## 6. 파라미터 설정

### 6.1 BehavioralConfig

```python
@dataclass
class BehavioralConfig:
    # FOMO
    fomo_trigger_threshold: float = 0.05   # 5% 상승 시 발동
    fomo_intensity: float = 50.0           # 강도 계수

    # 손실 회피
    loss_aversion_mean: float = 2.5        # 평균 (Genesove & Mayer)
    loss_aversion_std: float = 0.35        # 표준편차

    # 앵커링
    anchoring_threshold: float = 0.1       # 10% 이익률

    # 군집 행동
    herding_trigger: float = 0.03          # 3% 매수 시 발동
    herding_intensity: float = 10.0        # 강도 계수
```

### 6.2 LifeCycleConfig

```python
@dataclass
class LifeCycleConfig:
    marriage_urgency_age_start: int = 28
    marriage_urgency_age_end: int = 35
    newlywed_housing_pressure: float = 1.5
    school_transition_age_start: int = 10
    school_transition_age_end: int = 15
```

## 7. 학술적 근거

1. **Kahneman, D. & Tversky, A. (1979)**
   - "Prospect Theory: An Analysis of Decision under Risk"
   - Econometrica, 47(2), 263-291

2. **Genesove, D. & Mayer, C. (2001)**
   - "Loss Aversion and Seller Behavior: Evidence from the Housing Market"
   - Quarterly Journal of Economics, 116(4), 1233-1260

3. **Shiller, R.J. (2005)**
   - "Irrational Exuberance" (2nd ed.)
   - Princeton University Press

4. **Banerjee, A.V. (1992)**
   - "A Simple Model of Herd Behavior"
   - Quarterly Journal of Economics, 107(3), 797-817

5. **Case, K.E. & Shiller, R.J. (2003)**
   - "Is There a Bubble in the Housing Market?"
   - Brookings Papers on Economic Activity, 2003(2), 299-362

## 8. 향후 개선 방향

1. **뉴스/미디어 효과**
   - 언론 보도에 따른 기대 급변
   - SNS 정보 전파 모델

2. **정보 네트워크**
   - 가구별 사회적 연결
   - 네트워크 기반 정보 전파

3. **이질적 기대**
   - 낙관론자/비관론자 분포
   - 전문가 vs 일반인

4. **학습 및 적응**
   - 과거 경험에 따른 행동 변화
   - 정책 학습 효과
