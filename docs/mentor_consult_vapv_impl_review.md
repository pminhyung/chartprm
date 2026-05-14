# 멘토 자문 요청 — VAPV (Value-Anchored Process Verification) 구현 로직 검토

_프로젝트: ChartVCR (Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO)_
_본 페이지: VAPV reward 의 설계 의도, 실 구현, 평가 결과 정리. 옵션 비교 / 의견 미포함._

---

## 1. 배경

본 프로젝트는 chart-image 기반 numeric reasoning 에 대한 **process-level reward** 를 RL 학습 신호로 사용한다. Outcome-only (binary 0/1) 대비 reasoning trace 내부의 (a) 숫자 인용 정확성 (b) 산술 계산 정확성을 보상해 학습 신호를 풍부화하는 것이 의도이다.

설계 명칭은 **VAPV (Value-Anchored Process Verification)** 이며, 핵심 아이디어:
- _Value grounding_ — reasoning 내 모든 숫자 mention 중 CSV table 값과 매칭되는 비율
- _Anchored window_ — 숫자 양옆 ±20 토큰 윈도우 내 entity 토큰 (column header / row label) 출현을 요구해 indiscriminate 숫자 인용 게이밍 차단
- _Length scaling_ — 과도한 reasoning 길이에 대해 process score 감쇄

본 reward 의 4B Qwen3.5 정책 + LoRA r64/a128 + 5K-row v_hq 학습 코퍼스 (numeric 100%) 환경에서의 적용 결과, **outcome-only 대비 AVG5 평균 -1.40pp** 회귀가 관측되었다. 이에 reward 함수 자체의 설계·구현 정합성에 대한 검토를 요청한다.

---

## 2. 평가 결과 (4K max_tokens, standard scorer, think-on)

5 benchmarks: chartqa_human(H) / chartqa_augmented(A) / chartqa_pro(PRO) / charxiv_reasoning(CXR) / chartmuseum(CMU).
출처: `docs/standard_eval_comparison.md`. PRO 는 lmms-eval `relaxed_correctness` 점수 (`%` strip 미적용).

| Model | Algo | Reward | H | A | PRO | CXR | CMU | **AVG5** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| row_a_outcome | GRPO | outcome | 79.20 | 82.00 | 34.80 | 61.30 | 45.00 | **60.46** | base |
| row_b_vapv | GRPO | **VAPV v2** | 77.76 | 82.32 | 34.24 | 58.00 | 43.00 | 59.06 | **−1.40** |
| row3_dapo_outcome | DAPO | outcome | 73.84 | 79.60 | 32.75 | 56.80 | 40.70 | 56.74 | base |
| row4_dapo_vapv | DAPO | **VAPV v2** | 72.56 | 80.08 | 32.85 | 57.10 | 40.40 | 56.60 | **−0.14** |

ChartQA-Pro 공식 채점 (VLMEvalKit, ANLS + question-type-branched scorer):

| Model | Factoid | Conv | Hypoth | FactCheck | MultiCh | **Overall** | Δ |
|---|---|---|---|---|---|---|---|
| row_a_outcome | 51.44 | 63.08 | 48.71 | 61.89 | 44.83 | **53.26** | base |
| row_b_vapv | 47.32 | 56.54 | 46.40 | 62.30 | 42.93 | 49.84 | **−3.42** |
| row3_dapo_outcome | 46.26 | 50.93 | 46.79 | 57.79 | 44.44 | 48.21 | base |
| row4_dapo_vapv | 48.25 | 51.87 | 48.14 | 59.43 | 42.79 | 49.75 | +1.54 |

요점: GRPO 기반에서 VAPV 는 5 bench 중 4 개 (H/PRO/CXR/CMU) 에서 outcome 보다 낮음. DAPO 기반에서는 거의 무차이 (−0.14). 공식 VLMEK 채점에서는 GRPO+VAPV 가 −3.42pp 로 격차 확대, DAPO+VAPV 만 소폭 양 (+1.54).

학습 train-time reward log (Row 4 학습 19,576 reward calls 분석 — `docs/reward_v3_diagnosis.md` 측정):
- Wrong answer mean reward (v2 additive): 0.692 vs outcome 0.553
- Wrong answers reward ≥0.9 비율: 31.3% vs 0%
- Wrong answers reward = 1.0 비율: 21.1% vs 0%
- Correct−Wrong reward gap: 0.308 vs 0.447 (−31%)
- Within-group reward std: 0.104 vs 0.126 (−17.7%)
- Zero-variance group 비율: 38.9% vs 23.9% (+63%)

위 측정은 v2 additive 공식 (`min(1.0, r_acc + 0.3·r_proc)`) 시점. 현재 학습 사용 공식은 v2 multiplicative + length-scaled (아래 §3).

---

## 3. 구현 — Reward 함수 + Process Score

### 3.1 Reward wrapper (`train_grpo_dapo.py:319`, `reward_vapv_v2`)

분기 구조 (psuedo-flow):
```
for each completion:
  pred ← extract_answer_v2(response)           # post-think <answer> 추출

  if not _is_numeric_answer(gold):             # gold 가 텍스트
    r ← relaxed_text_match(pred, gold)         # binary 0/1
    return r                                   # process 미사용
  
  if no csv or not is_multi_step:              # CSV 없음 or 단일-fact 질문
    r ← cerm_accuracy(pred, gold)              # continuous [0,1]
    return 1.0 if r ≥ 0.95 else 0.0            # binary 변환
  
  r_acc ← cerm_accuracy(pred, gold)
  if r_acc ≥ 0.95:                             # 수치 정답
    return 1.0
  
  # 수치 + CSV + multi-step + 오답 → process 만 0.05 배 부여, floor 없음
  process, info ← compute_vapv_v2(response, csv, max_completion_length)
  return 0.05 · process
```

설계 의도 (코드 docstring 발췌):
- "No silent CSV fallback — no CSV → outcome-only"
- "Anchored grounding (denominator-aware, entity-windowed)"
- "Length scaling — process · (1 − (think_len/max_len)²) penalises bloat"
- "Wrong-answer floor removed: wrong + process → 0.0 + 0.05·process (was 0.15+0.30·)"

V1 (이전 사용 공식, deprecated): `min(1.0, r_acc + 0.3·r_proc)` — additive. wrong answer 가 1.0 도달 가능 → §2 train-time log 의 reward inflation 관측 → v2 로 변경.

### 3.2 Process score (`code/rewards/rule_verifier_fast.py:348`, `compute_vapv_v2`)

```python
def compute_vapv_v2(response, csv_path, max_completion_length=4096):
    grounding, arithmetic = compute_anchored_step_credit(response, csv_path)
    raw = 0.6 * grounding + 0.4 * arithmetic
    think_len = len(_split_reasoning(response).split())   # word-token approx.
    ratio = min(1.0, think_len / max(max_completion_length, 1))
    length_factor = max(0.0, 1.0 - ratio * ratio)
    process = raw * length_factor
    return process, info
```

- `grounding` ∈ [0, 1] — anchored value-matching ratio (아래 §3.3)
- `arithmetic` ∈ [0, 1] — `CALC_RE` 매치한 `a op b = c` 중 Python recompute 일치 비율 (5% tolerance)
- 가중치 0.6 / 0.4
- `length_factor` ∈ [0, 1] — think 길이 비율 r 에 대해 1 − r² (think_len=0 → 1.0, think_len=max_len → 0, 그 사이 quadratic)
- CSV 없음 → process 0.0 (outcome 분기로 회수)

### 3.3 Anchored grounding (`compute_anchored_step_credit`, `rule_verifier_fast.py:257`)

```python
def compute_anchored_step_credit(response, csv_path):
    table_values = get_table_values(csv_path)        # CSV 모든 cell 의 float set
    entities    = get_table_entities(csv_path)       # col header + row label tokens
                                                     # (소문자, len>1, digit 제외)
    reasoning   = _split_reasoning(response)         # <think>...</think> 본문

    tokens = re.findall(WORD_OR_NUM, reasoning)      # 단어/숫자 tokenize
    num_positions = [(i, parse(t)) for i,t in tokens 
                     if numeric(t) and not is_year(t) and abs(t)≥1e-10]
    
    matched = 0
    for i, n in num_positions:
        in_table = any(
            1.0/(1.0 + |n-tv|/max(|tv|,1e-10)) > 0.9   # ~10% tolerance
            for tv in table_values
        )
        if not in_table: continue
        if entities:
            window = tokens[max(0,i-20) : min(len, i+21)]
            if any(t.lower() in entities for t in window):
                matched += 1
        else:                                          # entity 추출 실패 시 fallback
            matched += 1
    grounding = matched / len(num_positions)

    arithmetic = recompute_CALC_RE(reasoning, tol=0.05)
    return grounding, arithmetic
```

핵심 매커니즘:
- denominator = **모든** non-zero non-year 숫자 mention. 매칭 안 되는 숫자가 늘면 점수 self-dilute.
- numerator = (값 매칭 ∧ entity window 매칭) 인 mention.
- entity = CSV col header + 1st-col row label 의 토큰 분해 결과 (예: `"GDP per capita"` → `{gdp, per, capita}`, 길이 1 또는 digit 제거).
- entity set 이 비면 entity 조건 미적용 (값 매칭만).
- 매칭 tolerance: `1.0/(1 + |Δ|/|tv|) > 0.9` ≈ ±10%.
- Year (1900–2030 integer) 는 numerator/denominator 양쪽에서 제거.

### 3.4 호출 환경 (numeric answer 비율)

| 데이터 | numeric % | text % |
|---|---:|---:|
| GRPO 학습 (v_hq baseline) | **100%** | 0% |
| ChartQA-H eval | 68% | 32% |
| ChartXiv-R eval | 45% | 55% |
| ChartMuseum eval | 27% | 73% |

학습 데이터 100% numeric → 모든 학습 샘플이 process 분기에 진입. 평가는 텍스트 비중 높은 bench (CXR/CMU) 에서 비중 감소.

### 3.5 보조 데이터 — train-time process 분포 (Row 4 학습 100 sample audit)

- `r_proc ≥ 0.95`: 79%
- `r_proc ∈ [0.5, 0.95)`: 16%
- `r_proc < 0.5`: 5%
- grounding accuracy 자체 평균: 97% — CSV 숫자가 reasoning 에 거의 다 등장
- length_factor 평균 (Row 4 학습 시 think median ≈ 400 words, max_len=4096): ≈ 0.99

→ process score 분산이 좁다. 0.05·process 의 wrong-answer signal 의 실 진폭 ≈ 0.05 × [0.7~1.0] ≈ 0.035~0.050.

### 3.6 V1 → V2 변경 요지 (히스토리)

| 항목 | V1 | V2 (현재) |
|---|---|---|
| Wrong-answer formula | `min(1.0, r_acc + 0.3·r_proc)` | `r_acc + 0.3·(1−r_acc)·r_proc_v2` (이론) / `0.05·process` (코드 실측 구현) |
| Grounding tolerance | > 0.8 (~20%) | > 0.9 (~10%) |
| Arithmetic default (no CALC match) | 0.5 | 0.0 |
| No-CSV fallback | 0.5 | 0.0 (outcome-only 분기) |
| Parse failure | correct += 1 | skip |
| Length scaling | 없음 | `(1 − ratio²)` quadratic |
| Anchored grounding (entity window) | 없음 | ±20 token window |

코드상 `reward_vapv_v2` 의 실 wrong-answer return 은 `0.05 · process` 이며, multiplicative `r_acc + 0.3·(1−r_acc)·r_proc` 는 doc/plan 문서에 명시된 표현. r_acc 가 0.95 미만이면 코드는 r_acc 를 reward 에 합산하지 않고 process 만 0.05 가중 (acc ≥ 0.95 분기는 r=1.0 별도 return).

---

## 4. 검토 요청 포인트

본 검토 요청은 위 구현이 **VAPV 의 설계 의도를 정확히 반영하는지**, 그리고 평가 회귀의 원인이 **(a) 설계 자체 문제** 인지 **(b) 구현 디테일 문제** 인지를 분리하기 위한 자문이다.

### 4.1 Process score 공식 (`compute_vapv_v2`)

- `raw = 0.6·grounding + 0.4·arithmetic` 의 가중치가 chart-numeric 도메인에서 적절한지.
- `length_factor = 1 − (think_len/max_len)²` 의 quadratic 형태와 cap (max_completion_length=4096) 의 결합 효과. 학습 시 think 길이 분포 (median ≈ 400 words ≈ 0.098 ratio → factor ≈ 0.99) 에서 실 차별화 진폭이 0.01 이하인 상황.
- think_len 측정이 word-split (whitespace tokenize) — Qwen tokenizer 의 BPE 단위와 비례하지 않을 가능성.

### 4.2 Anchored grounding (`compute_anchored_step_credit`)

- denominator = **모든** non-year non-zero numeric mention. 모델이 단순 산술 중간값 (예: `10 + 20 = 30` 중 `30`) 을 다수 인용 시 denominator 가 부풀어 numerator/denominator 가 낮아짐. 이 self-dilution 이 학습 signal 로서 의도된 동작인지 / 부작용인지.
- ±20 token window 의 width 선정 근거. chart-image 기반 reasoning 의 numeric mention 부근에서 entity 토큰이 항상 ±20 내 등장한다는 가정의 검증 여부.
- entity set 정의 = col header + 1st-col row label 토큰 분해. multi-column / wide-format CSV 에서 row label 만 entity 로 잡힘 (header 는 numeric value 컬럼 이름). 이 비대칭의 영향.
- entity set 이 비면 entity 조건을 떨어뜨리고 값 매칭만 채택 (fallback) — entity 추출 실패시 더 관대해지는 방향이 의도와 일치하는지.

### 4.3 Wrong-answer reward formula

- 코드: `return 0.05 · process` (no acc term). 설계 plan 문서 표기: `r_acc + 0.3·(1−r_acc)·r_proc_v2` (multiplicative).
- 두 표현이 다른 함수임. 어느 것이 의도된 공식인지 확인 필요. `0.05·process` 의 진폭 [0, 0.05] 가 outcome-only reward 의 [0, 1] 진폭 대비 학습 signal 로서 충분한지.
- "no floor" (V1 의 0.15 baseline 제거) 가 wrong-answer 그룹 내 variance 를 유지하기 위함인데, §3.5 의 좁은 process 분포 (95%+ 가 0.5 이상) 에서 실제 wrong-answer reward 분포가 0.03~0.05 의 좁은 대역에 집중되는지.

### 4.4 Train-eval coverage mismatch

- 학습 100% numeric → process 분기 100% 진입. 평가 일부 (CMU 73%, CXR 55%) 텍스트 → text branch (binary). text 답변에는 process signal 없음.
- VAPV 의 process supervision 이 numeric path 에서만 transfer 됨. text-heavy bench (CMU) 에서 VAPV 가 더 큰 회귀 (`row_b_vapv` H −1.44 vs CMU −2.00 vs outcome) 를 보이는 원인이 이 mismatch 와 일관되는지.

### 4.5 V1 train-log 의 reward inflation 진단 적용 후 잔존 효과

- V1 의 additive (`min(1.0, ...)`) 가 wrong-answer 의 21% 를 reward 1.0 에 몰아 zero-variance group 38.9% 발생 → V2 로 전환.
- V2 의 multiplicative + 0.05·process 가 이론적으로 wrong-answer = 1.0 도달 불가. 그러나 평가 결과는 여전히 outcome 대비 음. process signal 의 유효 진폭이 부족 / process 자체가 잘못된 dimension 을 보상 (예: 산술 중간값 인용 빈도 ≠ 정답 도달 능력) 가능성.

### 4.6 DAPO vs GRPO 비대칭

- GRPO 기반: VAPV AVG5 −1.40pp.
- DAPO 기반: VAPV AVG5 −0.14pp (거의 무차이), VLMEK Pro 에서는 +1.54pp.
- 두 algo 의 advantage 계산 방식 차이 (GRPO group-relative vs DAPO clip-higher + dynamic-sampling) 와 VAPV 의 좁은 분산 reward 사이의 상호작용. DAPO 의 dynamic-sampling 이 zero-variance group 을 거부함으로써 좁은 process signal 의 부정 영향을 완화하는지 / 무관한 우연인지.

---

## 5. Sample-level 회귀 분석 (row_a_outcome → row_b_vapv)

검토자가 §3 의 reward mechanism 별로 책임을 추정할 수 있도록, 동일 SFT base 에서 reward 만 (outcome vs VAPV) 다른 두 GRPO 모델의 회귀 샘플을 5 bench 전체 (총 6,448 sample) 에서 추출했다. **회귀 정의**: row_a_outcome score ≥ 0.5 ∧ row_b_vapv score < 0.5. 총 회귀 N=455 (improvement N=377, net −78).

### 5.1 Bench 별 회귀 분포 + 길이 시그널

| Bench | N | regr | impr | net | think_med chars (a→b) | content_med chars (a→b) | b_pred empty | numeric gold % |
|---|---:|---:|---:|---:|---|---|---:|---:|
| chartqa_human | 1250 | 65 | 47 | −18 | 2434 → 4310 (1.8×) | 6 → 5 | 0/65 | 82% |
| chartqa_augmented | 1250 | 51 | 55 | +4 | 2208 → 5644 (2.6×) | 7 → 7 | 0/51 | 92% |
| chartqa_pro | 1948 | 127 | 116 | −11 | 4086 → 10709 (2.6×) | 6 → 0 | 0/127 | 48% |
| charxiv_reasoning | 1000 | 106 | 73 | **−33** | 4980 → 11928 (2.4×) | 228 → 0 | **64/106 (60%)** | 41% |
| chartmuseum | 1000 | 106 | 86 | −20 | 5491 → **17220 (3.1×)** | 745 → 0 | 0/106 | 23% |

공통 패턴: row_b 의 think_len 이 모든 bench 에서 2.4–3.1× 증가, post-think content 가 hard bench (PRO/CXR/CMU) 에서 0 으로 붕괴, CXR 에서는 pred extraction 자체가 60% empty.

### 5.2 메커니즘별 측정 (455 회귀 sample, a vs b)

검토 포인트 §4 의 가설 별로 정량 측정한 결과:

| 가설 (§4 절 참조) | 메커니즘 | 판정 | a → b 측정값 |
|---|---|---|---|
| §4.1 arithmetic bonus | `a op b = c` 명시 산식 줄 학습 | partial | strict match 1.4–3.4× (chartqa_human 3.66→5.46, charxiv 0.55→0.92); 완화 calc-line CXR 14.3→29.3 (2.0×) |
| §4.2 denominator self-dilution | grounding 분모 = 모든 numeric mention → 재인용 안전 | **확인 (강)** | 중복 numeric mention 5 bench 모두 **≈2.0× 균일** (human 56.7→139.0, charxiv 92.5→202.1, CMU 89.1→191.3); 단일 값 최대 반복 charxiv 27→51 |
| §4.2 ±20 entity window | "Header: <num>" 패딩 학습 | 확인 | colon-label 패턴 1.2–3.4× (human 5.14→9.91, charxiv 1.16→3.91, CMU 3.99→7.69); quoted-header 1.7–3.0× |
| §4.1 length_factor | `1 − (think_words/4096)²` quadratic 페널티 | **반증** | 회귀 455건 중 think_words > 4096 = **0–1%** (CMU 1건, 그 외 0건). p95 word 2430–3767 → length_factor 학습 내내 ≈ 1.0, 사실상 미작동 |
| §4.3 wrong-reward 양 신호 | `0.05·process` (wrong 인데도 양 보상) → process-faking 시멘트 | 확인 (약) | process_score_proxy `log(1+nums)+log(1+arith)+log(1+colon_labels)` row_b 일관 **1.18× 상승** (5 bench 1.10–1.27×) |
| §4.4 train-eval gold-type mismatch | text-gold 도 bloat 전이 (policy-level cloning) | 확인 | text-gold subset bloat ratio numeric subset 보다 큼 — aug **11.6×** (vs num 2.44×), human 3.17× (vs 1.54×), CMU 3.40× (vs 2.57×) |
| eval-time truncation | 4–8K cap 안에서 `</think>` 미도달 | **확인 (dominant)** | row_b content empty: human 40%, aug 43%, PRO 57%, CXR **82%**, CMU 58%. CXR pred extraction fallback 60% empty |

### 5.3 대표 샘플 (3건)

**[denominator self-dilution + arithmetic bonus] `chartqa_human_667`** — gold=`19`, a_pred=`19`, b_pred=`19%`
Q: "What is the total of values below 10%?" / row_b think 3298 chars, 토큰 `19` 가 think 내 40회 재출현:
```text
chartqa_human_667 row_b think (middle excerpt)
    *   7 + 5 + 4 + 3
    *   7 + 5 = 12          (arith bonus 매치)
    *   12 + 4 = 16
    *   16 + 3 = 19
[… 25 lines: "19" vs "19%" 형식 자기-검토 루프 …]
- "19" is a word/number.
- "19%" might be considered a single word in casual speech but technically two tokens.
- Let's just give the number "19" or "19%". Given the context, "19%" makes most sense.
The answer is 19%.
```
think 11번째 줄에 이미 `19` 도달, 이후 41줄은 format self-debate. 최종 `19%` 출력 → `relaxed_text_match` fail.

**[eval-time truncation] `charxiv_1112`** — gold=`40`, a_pred=`40`, b_pred=`""`
Q: "What is the value of k where the highest peak in terms of magnitude is observed?" / row_a think 1286 chars (5번째 줄에 답 도달) / row_b think **11117 chars (8.6×)**:
```text
charxiv_1112 row_b think (last 8 lines, mid-think truncated)
    *   k=80: Peak ~50. Higher.
    *   k=100: Peak ~10. Lower.
    *   This is very erratic.
    *   Let's look at the plots again.
    *   k=20: Wide, low.
    *   k=40: Narrow, high.
    *   k=60: Medium width, medium height.
    *
```
think 내 답 확정 전에 cap 충돌 → `</think>` 미출현 → content / pred 모두 empty.

**[text-gold policy bloat 전이] `chartmuseum_707`** — gold=`Austin`, a_pred=`Austin`, b_pred=`Dallas`
Q: "Among High Growth Low Cost cities, which city has the second highest average technology salary?" / row_a 495 words / row_b **1566 words (3.2×)** — text-only gold (학습 시 process signal 비활성 분기) 인데도 bloat:
```text
chartmuseum_707 row_b think (tail)
    4.  Austin (~$98k)
    5.  Atlanta (~$95k)
    6.  Detroit (~$92k)
    7.  Montreal (~$82k)
    8.  Toronto (~$80k)
    Therefore, the second highest is Dallas.
```
순위 목록에서 Austin 4위로 정확히 적었으나 결론은 Dallas — long-context self-confusion.

### 5.4 회귀 인과 사슬 추정

측정 데이터가 일관되는 사슬:
1. 학습 시 think_words 분포 max 4K 미달 → `length_factor` 페널티 (§4.1) 미작동.
2. bloat 가 reward 상 안전 ∨ 유리: denominator self-dilution 으로 재인용 무손실 (§4.2), arithmetic 보상이 산식 줄 명시 유도 (§4.1), entity window ±20 이 header 인접 출력 유도 (§4.2), `0.05·process` 가 wrong-rollout 에서 약한 양 보상 (§4.3).
3. 100% numeric 학습 분포에서 굳어진 verbose 정책이 eval text-gold path 로 일반화 (§4.4).
4. eval cap (4–8K) 안에서 `</think>` 도달 실패 → content/pred empty → judge 0점. 회귀 −78 net 중 hard bench (PRO/CXR/CMU) 의 content-empty 만으로도 −40 이상 설명 가능.

→ §3 의 length 페널티 항이 학습 분포에서 실제로 발화하지 않는 점, 그리고 §3 의 denominator·anchored window·산술 보상 세 항이 모두 bloat 와 양립 가능한 점이 회귀 사슬의 기점인 상황.

---

## 6. 참고 문서

- `docs/reward_v3_diagnosis.md` — V1 → V2 전환 진단 (Row 4 19,576 reward call 분석).
- `docs/v8_reward_plan.md` — VAPV 적용 시점의 5개 구조적 한계 분석.
- `docs/standard_eval_comparison.md` — standard pipeline 평가 표.
- `docs/v8_distill_v2_chartqapro_vlmek_compare.md` — VLMEvalKit 공식 Pro 채점 표.

## 7. 코드 위치

- `code/rewards/rule_verifier_fast.py:222` — `get_table_entities` (entity 추출)
- `code/rewards/rule_verifier_fast.py:257` — `compute_anchored_step_credit` (anchored grounding)
- `code/rewards/rule_verifier_fast.py:348` — `compute_vapv_v2` (process score)
- `train_grpo_dapo.py:319` — `reward_vapv_v2` (reward wrapper, train-time 분기)

---

이상의 구현·평가·sample-level 분석을 두고 VAPV 의 설계 의도와 실 코드 간 정합성, 그리고 회귀 사슬의 기점이 §3 의 어느 메커니즘 (length 페널티 미작동 / denominator 정의 / anchored window / 산술 보상 / 0.05·process 양 신호) 에 귀속되는지에 대한 검토를 요청하는 상황.
