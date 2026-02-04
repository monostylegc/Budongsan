# 학술 참고 문헌 (Academic References)

이 문서는 한국 부동산 시장 ABM 시뮬레이션 모델에 사용된 학술적 근거를 정리합니다.

## 1. 손실 회피 (Loss Aversion)

### 핵심 파라미터
- **손실 회피 계수 (lambda)**: 평균 2.5, 범위 1.5-3.5
- **호가 프리미엄**: 손실 상황시 25-35% 높게 책정

### 근거 문헌

#### Tversky & Kahneman (1992)
- **제목**: "Advances in Prospect Theory: Cumulative Representation of Uncertainty"
- **출처**: Journal of Risk and Uncertainty, Vol. 5, pp. 297-323
- **핵심 발견**:
  - 전망이론의 손실 회피 계수 lambda = 2.25
  - 위험 태도 계수 alpha = beta = 0.88
  - "손실은 이득보다 약 2.25배 크게 느껴진다"

#### Genesove & Mayer (2001)
- **제목**: "Loss Aversion and Seller Behavior: Evidence from the Housing Market"
- **출처**: The Quarterly Journal of Economics, Vol. 116(4), pp. 1233-1260
- **URL**: https://academic.oup.com/qje/article/116/4/1233/1903212
- **데이터**: 1990년대 보스턴 콘도미니엄 시장
- **핵심 발견**:
  1. **호가 프리미엄**: 손실 상황의 매도자는 호가(asking price)를 기대 판매가 대비 **25-35%** 높게 책정
  2. **실거래 프리미엄**: 실제 판매가도 **3-18%** 높게 달성
  3. **매도 지연**: 손실 상황 매도자의 매도 확률(sale hazard)이 크게 감소
  4. **소유자 유형별 차이**: 자가 거주자(owner-occupant)의 효과가 투자자보다 2배 큼
- **모델 적용**:
  ```python
  loss_aversion_coefficient = 2.5  # 평균값
  asking_price_premium = 0.25 ~ 0.35  # 손실 상황시
  ```

#### 후속 연구 (덴마크 데이터)
- "손실이 이득보다 약 2.5배 더 크게 작용"
- 이는 Genesove & Mayer의 발견을 다른 시장에서 재확인

### 모델 구현
```python
# agents.py
loss_aversion = rng.normal(2.5, 0.35, size=self.n)
loss_aversion = np.clip(loss_aversion, 1.5, 3.5)
```

---

## 2. ABM 주택시장 모델

### JASSS 2020: 한국 주택시장 ABM
- **제목**: "Housing Market Agent-Based Model with LTV and DTI"
- **URL**: https://www.jasss.org/23/4/5.html
- **핵심 파라미터**:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Willingness to Pay (WTP) | 0.7 | 지불 의향 |
| Sale Probability (SP) | 0.3 | 매도 확률 |
| House Moving Rate (HM) | 0.1 | 이동 비율 |
| Market Participation (서울) | 0.007 | 시장 참여율 |
| Market Participation (지방) | 0.002 | 시장 참여율 |
| Price Increase Rate | 0.007 | 가격 상승률 |
| Price Decrease Rate | 0.2 | 가격 하락률 |

- **구매력 공식**:
  ```
  Affordable budget = WTP * [min(amountLTV, amountDTI) + savings]
  Maximum price = savings + amountDTI
  ```

- **에이전트 유형**:
  - 실수요자 (Owner-occupier)
  - 투자자 (Buy-to-let)
  - 투기자 (Speculator)

### JASSS 2024: UK 금융 충격 ABM
- **제목**: "Behavioural ABM for Housing Markets: UK Financial Shocks"
- **URL**: https://www.jasss.org/27/4/5.html
- **핵심 파라미터**:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Affordability (alpha) | 0.33 | 소득의 33% 한도 |
| Propensity threshold (omega) | 0.2 | 투자 성향 임계값 |
| Price decay | 3%/월 | 미판매시 가격 하락 |

- **금융 충격 효과**:
  - 금리 +4.3%p: 주택가격 44.2% 하락 (50년)
  - 금리 -4.3%p: 주택가격 62.3% 상승 (23년)
  - LTV 하락 (90%->69%): 주택가격 86.28% 폭락

---

## 3. 행동경제학 편향

### Nature 2023: 부동산 투자의 행동 편향
- **제목**: "Behavioural Biases in Real Estate Investment"
- **URL**: https://www.nature.com/articles/s41599-023-02366-7
- **주요 편향**:
  1. **손실 회피 (Loss Aversion)**: 가장 많이 연구됨
  2. **앵커링 (Anchoring)**: 매입가에 집착
  3. **군집 행동 (Herding)**: 주변 행동 모방
  4. **과신 (Overconfidence)**: 자신의 판단 과대평가
  5. **처분 효과 (Disposition Effect)**: 이익은 빨리, 손실은 늦게 실현
  6. **후회 회피 (Regret Aversion)**: 후회 가능한 결정 회피

### 모델 적용
```python
# FOMO (Fear Of Missing Out)
fomo_factor = 1.0 + fomo_sensitivity * excess_rise^2 * 50.0

# Herding (군집 행동)
herding_factor = 1.0 + herding * (region_buying - 0.03) * 10.0

# Anchoring (앵커링)
if gain_loss_ratio < 0.1:
    anchoring_penalty = (0.1 - gain_loss_ratio) * 0.5
```

---

## 4. 한국 주택시장 정책

### Springer 2024: 한국 주택시장 정책 효과
- **제목**: "Impacts of Demand/Supply Interventions on Korea Housing"
- **URL**: https://link.springer.com/article/10.1007/s00168-024-01274-1
- **핵심 발견**:
  - 공급 정책이 세금 정책보다 효과적 (CGE 모델)
  - 수요 억제 정책의 한계

### ScienceDirect 2023: 지역별 주택 사이클
- **제목**: "Regionally Heterogeneous Housing Cycles in Korea"
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0264999323000044
- **핵심 발견**:
  - 지역별 LTV 차등 정책 효과
  - 서울/수도권 vs 지방의 비대칭적 반응

---

## 5. 에이전트 유형별 파라미터

### 실수요자 (Owner-Occupier)
- 비율: 80%
- 손실 회피: 높음 (lambda = 2.5)
- FOMO: 중간
- 목적: 거주

### 투자자 (Investor)
- 비율: 15%
- 손실 회피: 중간 (lambda = 2.0)
- FOMO: 중간
- 목적: 임대 수익

### 투기자 (Speculator)
- 비율: 5%
- 손실 회피: 낮음 (lambda = 1.8)
- FOMO: 높음
- 위험 허용: 높음
- 군집 성향: 높음
- 보유 기간: 6-24개월
- 목적: 시세차익

---

## 6. 캘리브레이션 데이터 출처

### 한국 통계
- 국토교통부 실거래가
- 한국부동산원 주택가격지수
- 통계청 가계금융복지조사
- KB국민은행 주택가격동향

### 파라미터 캘리브레이션
| 파라미터 | 값 | 근거 |
|---------|-----|------|
| 무주택 비율 | 45% | 통계청 |
| 1주택 비율 | 40% | 통계청 |
| 다주택 비율 | 15% | 통계청 |
| 월 소득 중위값 | 300만원 | 통계청 |
| 자가보유율 | 55% | 통계청 |

---

## 참고 문헌 목록

1. Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5, 297-323.

2. Genesove, D., & Mayer, C. (2001). Loss aversion and seller behavior: Evidence from the housing market. *The Quarterly Journal of Economics*, 116(4), 1233-1260.

3. Yun, H., & Jin, J. (2024). Impact of Loss Aversion on Seller Behavior. *SSRN Working Paper*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4890355

4. JASSS (2020). Housing Market Agent-Based Model with LTV and DTI. *Journal of Artificial Societies and Social Simulation*, 23(4), 5. https://www.jasss.org/23/4/5.html

5. JASSS (2024). Behavioural Agent-Based Model for Housing Markets: UK Financial Shocks. *Journal of Artificial Societies and Social Simulation*, 27(4), 5. https://www.jasss.org/27/4/5.html

6. Taylor & Francis (2024). Mitigating Housing Market Shocks: ABM + Reinforcement Learning. *Journal of Simulation*. https://www.tandfonline.com/doi/full/10.1080/17477778.2024.2375446

7. Nature Humanities & Social Sciences Communications (2023). Behavioural biases in real estate investment. https://www.nature.com/articles/s41599-023-02366-7

8. Springer (2024). Impacts of demand/supply interventions on Korea housing. *Annals of Regional Science*. https://link.springer.com/article/10.1007/s00168-024-01274-1

9. ScienceDirect (2023). Regionally heterogeneous housing cycles in Korea. *Economic Modelling*. https://www.sciencedirect.com/science/article/abs/pii/S0264999323000044
