# 부동산 시뮬레이션 논리적 문제점 검토 보고서

## 핵심 문제: "결과를 정해놓고 역설계"

현재 시뮬레이션은 **"한국은 1주택 집중화가 일어난다"는 결론을 먼저 정하고, 그 결과를 만들기 위해 파라미터를 조정**한 흔적이 많습니다.

---

## 1. 억지로 끼워맞춘 확률들

| 규제 | 설정값 | 문제 |
|------|--------|------|
| 1주택 추가매수 | 5% 확률 | 근거 없음 |
| 2주택 추가매수 | 3% 확률 | 취득세 8% 고려시 0.5% 이하여야 |
| 3주택+ 추가매수 | 1% 확률 | 대출불가+12% 취득세면 0.1% 이하 |

**문제**: 실제 세부담을 계산하지 않고 "적당해 보이는" 확률을 넣음

**위치**: `agents.py` lines 1171-1192

```python
# 1주택자 추가매수
final_buy_decision = 1 if add_buy_chance < 0.05 else 0

# 2주택자
final_buy_decision = 1 if rare_chance_2 < 0.03 else 0

# 3주택+
final_buy_decision = 1 if rare_chance_3 < 0.01 else 0
```

**개선 방안**:
- 취득세, 종부세, 양도세를 **명시적 비용**으로 계산
- 기대수익률 < 세금비용이면 자동으로 매수 거부
- 확률이 아닌 경제적 합리성으로 판단

---

## 2. 일자리 밀도 값의 출처 불명확

**위치**: `config.py` lines 49-63

```python
REGION_JOB_DENSITY = np.array([
    1.0,   # 강남3구 - 테헤란로, 금융/IT/서비스업 집중
    0.85,  # 마용성
    0.5,   # 기타서울
    0.9,   # 분당판교
    0.35,  # 경기남부
    0.2,   # 경기북부
    ...
])
```

**문제**:
- "통계청 사업체조사 기반"이라 했지만 **실제 계산식 없음**
- 강남 1.0이 "절대값"인지 "상대값"인지 불명확
- 0.5를 기준으로 나누는 근거도 없음

**개선 방안**:
- 구체적 지표 선택: "경제활동인구 1000명당 고소득(3000만원+) 일자리 수"
- 실제 통계청 데이터 인용 (특정 년도 명시)
- 지역별 계산식 공개

---

## 3. 고자산가 기준 불일치

**위치**: `agents.py` 여러 곳

```python
# 한 곳에서는 절대값 (line 1055-1056)
is_wealthy = asset >= 10000.0  # 1억

# 다른 곳에서는 백분위 (line 728)
asset_threshold_80 = np.percentile(assets, 80)  # 상위 20%
```

**문제**:
- 1억이 상위 몇 %인지 시뮬레이션 내에서도 일관되지 않음
- 자산 분포(파레토)에서 상위 20%는 약 1.2~1.5억 수준
- 두 기준이 충돌

**개선 방안**:
- **통일**: 모든 곳에서 백분위수(예: 상위 20%) 사용
- 또는 절대값 사용시 근거 명시

---

## 4. affordability 임계값 혼재

| 대상 | 임계값 | 위치 |
|------|--------|------|
| 실수요자 (일자리↑) | 0.15 | 지역선택 |
| 실수요자 (일자리↓) | 0.20 | 지역선택 |
| 고자산가 | 0.10 | 지역선택 |
| 투자자/투기자 | 0.12 | 지역선택 |
| 매수결정 | 0.25 | decide_buy_sell |

**문제**:
- 5가지 다른 기준이 혼재
- 왜 이 값들인지 근거 없음
- affordability 정의 자체도 모호

**개선 방안**:
- affordability 개념을 **DSR(부채상환비율)**로 전환
- 에이전트 타입별 리스크 선호도에 따라 차등 설정
- 일관된 체계 구축

---

## 5. 가중치 불균형

**위치**: `agents.py` lines 1082-1102

```python
prestige_bonus = prestige * 0.08  # 기본
# 투기자: prestige * 0.15
# 고자산가: * 1.5
# 최대: 1.0 * 0.15 * 1.5 = 0.225

job_bonus = job_density * 0.15  # 최대 0.15
```

**문제**:
- 심리적 프리미엄(prestige)과 일자리(job)가 비슷한 가중치
- 논리적으로 **일자리가 거주 필수요건**, prestige는 부가요소
- job_bonus가 2배 이상 높아야 함

**현실과의 괴리**:
```
논리적 우선순위:
1. 일자리 (통근 필수) → 높은 가중치
2. 학군/교통 (생활 편의) → 중간 가중치
3. 심리적 프리미엄 (선호) → 낮은 가중치

현재 시뮬레이션:
→ prestige와 job이 거의 동등
→ prestige가 오히려 더 높을 수 있음 (1.5배 배수 적용)
```

**개선 방안**:
```python
# prestige는 보조 인자로만 역할 (최대 0.05-0.08)
prestige_bonus = prestige * 0.05

# job_bonus는 강화 (실수요자에게 필수)
job_bonus = job_density * 0.20
```

---

## 6. "갈아타기" 로직의 미완성

**위치**: `agents.py` lines 1156-1167

```python
elif owned == 1:  # 1주택자
    has_lifecycle_reason = (life_stage == 2) or (life_stage == 3) or (life_stage == 5)
    if has_lifecycle_reason:
        # 갈아타기 목적: 정상적인 매수 결정
        final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0
        # 하지만 wants_to_sell은 별도 계산!
```

**문제**:
- 갈아타기 = 팔고 사기 = wants_to_buy=1 AND wants_to_sell=1
- 하지만 **둘이 연계되지 않음**
- wants_to_sell은 별도의 로직에서 계산 (line 1200+)

**결과**:
```
시뮬레이션 플로우:
1. decide_buy_sell() 실행
   - wants_to_buy[i] = 1 (갈아타기로 판단)
   - wants_to_sell[i] = ? (별도 계산)
2. wants_to_sell이 0이면?
   - wants_to_buy = 1인데 집을 팔지 않음
   → 2주택자가 됨 (갈아타기 아님!)
```

**개선 방안**:
```python
if has_lifecycle_reason:
    final_buy_decision = 1 if (buy_score + noise) > buy_threshold else 0
    if final_buy_decision == 1:
        self.wants_to_sell[i] = 1  # 갈아타기 = 동시 매도 강제
```

---

## 7. 정책 페널티의 점수 체계 모순

**위치**: `agents.py` lines 1104-1143

```python
policy_penalty = 0.0

if owned >= 2:
    policy_penalty = 1.5  # buy_threshold(0.4)보다 훨씬 큼
elif owned == 1:
    if self.wants_to_sell[i] == 0:
        policy_penalty = 0.6
    else:
        policy_penalty = 0.05
```

**문제**:
- 덧셈 기반 점수가 0~1 범위인데
- policy_penalty가 0.6, 1.5로 매우 큼
- **확률적 감소가 아닌 확정적 봉쇄**

**점수 계산 예시**:
```
buy_score = urgency + fomo + ... - policy_penalty

예:
- urgency = 0.15
- fomo_bonus = 0.1
- 합계 = 0.25
- policy_penalty = 0.6 (1주택 + 추가매수)
- 최종 = 0.25 - 0.6 = -0.35 < 0.4(threshold)
→ 매수 불가능 (확정)
```

**개선 방안**:
- policy_penalty 범위를 0-1로 정규화
- 또는 확률적 처리로 명확화:
```python
# buy_score 계산 후
if owned >= 2:
    policy_factor = 0.03  # 3% 확률만 통과
    if rng.random() > policy_factor:
        final_buy_decision = 0
```

---

## 8. 전망이론의 형식적 적용

**위치**: `agents.py` lines 9-50, 1020-1031

```python
"""전망이론 가치 함수 (Tversky & Kahneman, 1992)"""
# ... 정교한 함수 정의 ...

# 하지만 실제 사용:
pt_bonus = weighted_prob * pt_value * 0.05
pt_bonus = ti.min(pt_bonus, 0.15)  # 최대 0.15
```

**문제**:
- 전망이론을 인용했지만 실제 영향력은 **전체 점수의 5-10%**
- "학술적으로 보이게" 하려는 의도만 있고 실제 적용은 미미
- 손실 회피(loss aversion)가 매도 결정에는 강하게 적용되지만, 매수에는 약함

**개선 방안**:
- 두 가지 중 선택:
  1. **진정성 있는 전망이론 모델**: 모든 이득/손실을 prospect value로 계산
  2. **간단한 모델 + 명확한 주석**: "간단한 행동경제학 요소 추가" 명시

---

## 9. 하드코딩된 임계값 종합

| 파라미터 | 값 | 파일 | 근거 |
|---------|-----|------|------|
| 다주택 추가매수 확률 | 5%, 3%, 1% | agents.py | 없음 |
| 일자리 밀도 기준 | 0.5 | agents.py | 출처 불명확 |
| 고자산가 기준 | 10000만원 | agents.py | 상위 20%와 모순 |
| affordability 임계값 | 0.10~0.25 | agents.py | 혼재됨 |
| PIR 페널티 완화 | 50% | agents.py | 근거 없음 |
| prestige 최대값 | 0.225 | agents.py | job과 불균형 |
| FOMO 최대값 | 0.3 | agents.py | 생애주기와 동등 |
| 정책 페널티 | 0.6~1.5 | agents.py | 점수 체계 모순 |
| cascade 임계값 | 0.3 | config.py | 근거 없음 |

---

## 개선 권장사항

### 우선순위 1: 논리 모순 해결 (필수)
1. **정책을 명시적으로 모델링**: 취득세, 종부세를 "비용"으로 계산
2. **갈아타기와 wants_to_sell 연계**
3. **affordability 기준 통일**

### 우선순위 2: 근거 추가 (매우 중요)
1. 일자리 밀도: 실제 데이터 출처와 계산식 명시
2. 고자산가: 백분위수로 통일
3. 확률들: 실제 거래 데이터 기반으로 추정

### 우선순위 3: 체계 개선 (중요)
1. 정책 효과를 **비용 기반**으로 전환 (확률 X → 세금 O)
2. 가중치 재조정: job > prestige > FOMO
3. DSR 기반 affordability로 전환

### 우선순위 4: 모델 강화 (개선)
1. 고자산가 특화 행동 추가
2. 생애주기별 주택 선호 구체화
3. 세제 정책의 명시적 모델링

---

## 결론

현재는 **"시뮬레이션"보다 "목표된 결과를 만드는 시스템"**에 가깝습니다.

**올바른 접근**:
1. 정책(취득세, 종부세 등)을 **명시적으로 모델링**
2. 에이전트의 의사결정을 **일관된 효용최대화 로직**으로 통합
3. 규칙은 "코드에 심지 말고 경제학적으로 도출"
4. 그러면 자연스럽게 원하는 결과가 나타남

**현재 문제**:
- "한국은 다주택 규제가 강하니까 1주택 집중화" → 결론을 먼저 정함
- 그 다음 확률/가중치를 조정해서 결과를 강제함
- 학술적 모델링이 아닌 **의도적 조정**

---

*작성일: 2026-02-05*
*검토 대상: src/realestate/agents.py, config.py, market.py*
