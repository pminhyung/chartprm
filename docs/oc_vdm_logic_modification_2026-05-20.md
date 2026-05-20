# OC-VDM Logic Modification — Bench-Agnostic Reward Construction

_Date: 2026-05-20_
_Track: research-chartprm_
_Predecessor: `oc_vdm_validation_report.md`_

---

## 0. TL;DR — 6 actionable modifications, ordered by ROI

| # | Modification | 효과 | 비용 |
|---|---|---|---|
| **M1** | Terminal step (k=n_steps-1) OC-VDM 측정 skip | chartmuseum 25/30 terminal=fake PatA 제거. charxiv PatC% 18%→21.4% | 0 (analyzer 1 line) |
| **M2** | chartqa_pro 의 open-ended gold → judge LLM scoring | chartqa_pro 의 fake PatA 27.5% 회복 → OC-VDM 적용 영역 확장 | 30min (scorer 분기) |
| **M3** | Pattern B threshold 강화 (`mc_w ≤ 0.3 AND vd ≤ -0.4`) | image-as-distractor false positive 제거 (chartmuseum 의 PatB 중 ~50% 가 prefix-already-correct 케이스) | 0 (분류 함수 수정) |
| **M4** | Claim-density step pre-filter | "Wait/Let's" 류 exploration step skip → 측정 compute -30%, 신호 SNR ↑ | 1h (regex/heuristic) |
| **M5** | Confidence-weighted modulation: `mod = 1 + λ·vd·min(mc_w, mc_wo+ε)` | marginal-noise PatB/C 약화, 양 branch 모두 비자명한 step 만 강조 | 0 (수식 1줄) |
| **M6** | Question-type-adaptive routing: T/F → outcome-only, multi-step quantitative → full OC-VDM, visual-ID → confidence-weighted OC-VDM | bench-agnostic working logic 달성 (chartqa_pro 도 step-poor 케이스 학습 가능) | 2h (orchestration) |

**M1+M2+M3 만으로 chartmuseum 의 informative% 17.3% → ~22%, charxiv 38% → ~42% 예상. M4-M6 는 mini-GRPO 학습 단계에서 본격 도입 권장.**

---

## 1. 실제 sample 직접 분석 결과 (no_cap 측정 데이터 기반)

### 1.1 Pattern A "false positive" — terminal step 측정 artifact

**chartmuseum_826 (gold="9"), step 5/6** (terminal):
- step_text: `The rest of the states are colored with other airlines. So only Alaska has Alaska Airlines as the cheapest. So the number is 1. </think><answer>1</answer>`
- with_preds: `['', '', '']` — **모든 sub-rollout 빈 문자열**
- without_preds: `['', '', '']` — 동일
- 원인: prefix 가 `</answer>` 로 종료 → continuation 생성 0 char → empty pred → outcome=0 → fake PatA

**chartmuseum_677 step 6/7** (terminal): 동일 패턴, empty preds.

**Quantification**:
- chartmuseum: 25/30 terminal = fake PatA (**83.3%**)
- charxiv: 9/30 terminal = fake PatA (30%)
- 즉, chartmuseum PatA 152/208 (73.1%) 중 25 는 측정 artifact

→ **M1: terminal step 측정 skip**.

### 1.2 Pattern A "true hard" — pre-terminal 검증

**chartmuseum_826 step 4/6** (pre-terminal, PatA):
- step_text: `The other states: looking at the key, Alaska is one color. The map shows that only Alaska is colored with Alaska Airlines.`
- with_preds: `["Wait, let's", "But wait, let's check the map again. The", 'So, the answer is 1.']`
- gold="9", model says "answer is 1" → **genuinely wrong**. 진짜 hard sample.

→ pre-terminal PatA 의 대부분은 정당. terminal-only skip 으로 충분.

### 1.3 Pattern C (image-critical) — 실제 signal 정당성

**charxiv_1284 step 1/4 (gold="model 1")**:
- mc_w=1.00 mc_wo=0.00
- with_preds: `['Final Answer: model 1', 'Final Answer: model 1', 'Answer: Model 1', '**Final Answer: model 1**']` (4/4 정답)
- without_preds: `['...look at the plots more carefully', 'Final Answer: M3', 'So the answer is: M10', 'But without seeing the image']`
- **Verdict: 완벽한 PatC — image 가 정답 단서, 가리면 무작위 추측**

**chartmuseum_604 step 4/8 (gold="LINK")**:
- mc_w=0.75 mc_wo=0.25
- with_preds: 2/4 "LINK", 1/4 "LINK w/ Reranker" (variant), 1/4 hedging
- without_preds: 1/4 "LINK", 3/4 wrong variants
- step_text 가 legend color 분석 (blue=LINK 식별)
- **Verdict: 정당한 PatC** — image 가 legend 색-이름 매핑 단서.

### 1.4 Pattern B 의 3 가지 conflated 현상 (CRITICAL)

**(a) True text leakage** — image 없이도 답 가능
- charxiv 류 일부: question 텍스트에 답 단서 (예: "what is the gold standard color?" 류 자기참조 질문). 드뭄.

**(b) Image-as-distractor** — prefix 가 이미 답 잠금, image 가 noise
- **chartmuseum_604 step 6/8 (gold="LINK")**: mc_w=0.25 mc_wo=0.75
  - step_text: `Wait, the caption says "..." when reranker is removed. So without reranker (blue), the distribution is centered...`
  - 모델이 caption 텍스트 다시 읽으며 혼란
  - with_preds: 1/4 "LINK" (correct), 3/4 wrong variants
  - without_preds: 2/4 "LINK" (correct) — **prefix 만으로 step 4 의 결론 유지**
  - **이는 진짜 leakage 가 아니라 "model 이 image 의 caption 을 추가로 읽으며 혼란"**.

**(c) Marginal noise** — vd ∈ [-0.3, -0.1]
- **charxiv_1524 step 1/3 (gold="1905")**: mc_w=0.50 mc_wo=0.75
  - 양 branch 비슷, statistical noise 수준.

→ **M3: Pattern B threshold 강화**: `mc_w ≤ 0.3 AND mc_wo ≥ 0.7 AND vd ≤ -0.4` — (a) 만 검출, (b)(c) 는 PatE 로 강등.

### 1.5 chartqa_pro thinking model — reasoning IS 존재, scoring 만 fail (F2 확정)

**chartqa_pro_1327 (gold="workers who switch jobs vs. those who stay")**:
- thinking model 이 2-step reasoning 생성
- with_preds (step 1): `'The answer is: workers who switch jobs vs. those w...'`, `'The graph compares the wage increase gap between *...'`
- **모든 with_pred 가 semantically 정답** but `relaxed_correctness` exact-match fail → mc=0.

**chartqa_pro_541 (gold="Unanswerable")**:
- with_preds: `"The graph doesn't show the exact share price chang"`, `'The answer is: Not available in the graph...'`
- without_preds: `'**Final Answer: \\boxed{1.17}**'` (hallucinate)
- **with-image 가 "graph 에 없음" 정답, without-image 가 hallucinate** — 자연스러운 PatC 인데 scoring fail 로 PatA.

→ **M2: chartqa_pro 도 judge LLM scoring** (또는 type detector → bool/numeric/open 별 routing).

### 1.6 chartmuseum 의 step granularity 문제

chartmuseum 8b_thinking 평균 n_steps=7. step 0~3 은 대체로 "let me look at the chart", "Wait, the map shows..." 류 exploration. step 4 이후에 claim 등장. 

**chartmuseum_677 step 1/7 (PatD)**:
- step_text: `Let's check each row: Row 1: Left graph. The red line is flat. Does it reach step 20?...`
- 구체적 entity ("Row 1", "step 20") 있음 → claim 있음
- mc_w=0.25 mc_wo=0.00 → 신호 약하지만 정당

**chartmuseum_677 step 5/7 (PatD)**:
- step_text: `Wait, but wait. Let's look at the graphs again. The first row's left graph: the red line is horizontal...`
- "Wait, but wait" — re-checking, 새 정보 없음
- 측정 noise 가능성 큼

→ **M4: claim-density pre-filter** — `Wait/Let me/Hmm` 류 + 구체적 entity (숫자, 고유명사) 부재 step 은 skip.

---

## 2. 도출된 reward logic modification 방향

### 2.1 핵심 원리: "measurement quality > metric tuning"

- 현재 PatC/B/D/E 분류 threshold 자체보다, **측정 자체의 noise 제거가 우선**.
- Terminal step pollution, chartqa_pro scoring, image-as-distractor → 모두 **측정 단계의 issue**.
- Threshold/modulation 미세 조정은 측정 noise 제거 후에야 의미 있음.

### 2.2 Implementation 우선순위

**Phase 1 — measurement noise 제거 (즉시 적용)**:

```python
# image_dep_mc_v2.py
def process_sample(s, ...):
    ...
    steps = split_steps(trace, max_steps=8)
    # M1: terminal step skip
    measurable_steps = steps[:-1] if len(steps) > 1 else steps
    
    for k, step_txt in enumerate(measurable_steps):
        # M4: claim density filter (optional, compute 절약)
        if is_exploration_step(step_txt):
            continue
        rec = await process_step(...)
        step_records.append(rec)

def is_exploration_step(text):
    \"\"\"Skip 'Wait, let me re-check' style steps with no specific claims.\"\"\"
    text_lower = text.lower()
    has_exploration_marker = any(m in text_lower for m in ['wait,', "let's check", "let me look", "hmm,"])
    has_specific_claim = bool(re.search(r'\d+|\b[A-Z][a-z]+\b|"[^"]+"', text))
    return has_exploration_marker and not has_specific_claim
```

```python
# scoring routing (M2)
async def score_subrollout(text, bench, gold, question, judge_client, judge_model, judge_sem):
    pred = extract_pred(text)
    if not pred.strip(): return 0, pred
    
    # M2: chartqa_pro question-type-adaptive
    if bench == "chartqa_pro":
        gold_norm = gold.strip().lower()
        if gold_norm in ('true','false'):
            return int(pred.strip().lower() == gold_norm), pred
        if re.match(r'^-?\d+\.?\d*%?$', gold.strip()):
            return relaxed_correctness(pred, gold), pred
        # open-ended → judge LLM
        prompt = CHARTQA_OPEN_J.format(q=question, a=gold, r=pred)
        return await judge_remote(judge_client, judge_model, prompt, judge_sem), pred
    ...
```

**Phase 2 — Pattern classification 강화 (즉시 적용, 분석만)**:

```python
def classify_v2(mc_w, mc_wo):
    if mc_w is None or mc_wo is None: return None
    vd = mc_w - mc_wo
    # A: hard-impossible (unchanged)
    if mc_w == 0 and mc_wo == 0: return 'A'
    # B: TRUE text leakage only — model strictly worse with image
    if mc_wo >= 0.7 and mc_w <= 0.3 and vd <= -0.4: return 'B'
    # C: clear image-critical (unchanged)
    if mc_w >= 0.5 and vd > 0.3: return 'C'
    # D: hard-perception (unchanged)
    if 0 < mc_w < 0.3 and mc_wo <= 0.1: return 'D'
    # E: marginal / image-as-distractor — no modulation
    return 'E'
```

**Phase 3 — Reward construction (mini-GRPO 학습 시)**:

```python
def oc_vdm_reward(step_records, base_traj_adv, lambda_vd=1.0):
    \"\"\"
    bench-agnostic step-level reward modulation.
    M5: confidence-weighted modulation
    M6: applies uniformly across benches (chartmuseum/charxiv/chartqa_pro)
    \"\"\"
    step_rewards = []
    for sr in step_records:
        mc_w, mc_wo = sr.get('mc_value'), sr.get('mc_without_value')
        if mc_w is None or mc_wo is None:
            step_rewards.append(base_traj_adv); continue
        pat = classify_v2(mc_w, mc_wo)
        vd = mc_w - mc_wo
        
        if pat == 'A':
            modulation = 0.0  # filter out (no contribution)
        elif pat == 'B':
            modulation = 0.3  # strong down-weight TRUE leakage
        elif pat == 'C':
            # M5: confidence-weighted — boost steps with consistent image-criticality
            confidence = mc_w  # high mc_w → high confidence
            modulation = max(1.0, 1.0 + lambda_vd * vd * confidence)
        elif pat == 'D':
            modulation = 1.1  # mild up-weight (model showing PROGRESS with image)
        else:  # E (marginal)
            modulation = 1.0
        
        # Only modulate positive trajectory advantage
        step_adv = base_traj_adv * modulation if base_traj_adv > 0 else base_traj_adv
        step_rewards.append(step_adv)
    return step_rewards
```

**Phase 4 — Question-type-adaptive routing (mini-GRPO 학습 시)**:

```python
# At trajectory level:
def select_reward_strategy(sample):
    \"\"\"M6: bench-agnostic via question-type routing.\"\"\"
    gold = (sample.get('gold') or '').strip().lower()
    n_steps = len(sample.get('step_records', []))
    
    if n_steps < 2:
        return 'outcome_only'  # T/F or single-line answer — no step structure
    if gold in ('true','false','yes','no'):
        return 'outcome_only'  # verification question
    if re.match(r'^-?\d+\.?\d*%?$', gold):
        return 'oc_vdm_full'  # quantitative — OC-VDM works well
    return 'oc_vdm_confidence_weighted'  # open-ended — use confidence-weighted modulation
```

### 2.3 예상 효과 (M1+M2+M3 적용 시, 기존 데이터 재분석)

| Combo | OLD informative% | M1 적용 | +M3 적용 |
|---|---:|---:|---:|
| chartmuseum 8b_thinking | 17.3% | 18.4% | ~22% |
| charxiv 8b_thinking | 38.0% | 41.4% | ~43% |
| chartqa_pro 8b_thinking | 0% | 0% (M2 미적용 시) | **~28%** (M2 적용 시) |

→ **bench 전반에 informative signal ≥ 20% 달성**.

---

## 3. Bench-agnostic working logic 의 최종 모습

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Measurement (image_dep_mc_v2.py)              │
│  - Terminal step skip (M1)                              │
│  - Question-type scoring routing (M2)                   │
│  - Claim-density pre-filter (M4, optional)              │
│  - max_tokens cap 없음 (이미 적용)                       │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: Classification (analyzer)                      │
│  - Pattern A/B/C/D/E with tightened B threshold (M3)    │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: Reward construction (training)                 │
│  - Trajectory-type routing (M6)                         │
│    - T/F: outcome-only                                  │
│    - Quantitative: oc_vdm_full                          │
│    - Open-ended: oc_vdm_confidence_weighted (M5)        │
│  - Sample-level filter: drop traj where max(mc_w)=0     │
└─────────────────────────────────────────────────────────┘
```

### 3.1 왜 universal 한가?

- **M1**: 모든 bench 에서 terminal-step artifact 제거 → 공통 측정 품질 보장
- **M2**: chartqa_pro 의 scoring fail 해결 → 모든 bench 에서 sub-rollout 평가 공정
- **M3**: PatB 가 진짜 leakage 만 검출 → bench-specific noise (chartmuseum 의 image-as-distractor) 제거
- **M5+M6**: question-type 별 reward strategy → bench 별 sweet spot 활용 (강제 universal 보다 strategic universal)

### 3.2 왜 step content parsing 필요 없는가?

이 modification 들은 **모두 outcome 만 사용**한다:
- Terminal step 식별: prefix 의 `</answer>` 또는 `n_steps - 1` index (구조적, content 비의존)
- Claim density (M4): regex 기반 heuristic, semantic parsing 아님
- Question-type routing (M6): gold answer format 기반, reasoning content 비의존

→ 이전에 폐기된 claim-parsing 방향 (verifier-based) 재도입 없이 universal 달성.

---

## 4. 권장 action (timeline)

| Phase | 작업 | 시간 | 효과 |
|---|---|---|---|
| **즉시** | M1+M3 analyzer 패치 (재측정 불필요) | 30min | 기존 데이터 재해석으로 informative% 상승 |
| **즉시** | M2 score_subrollout patch + chartqa_pro 30 sample 재측정 (no judge call 추가, judge 1 server 활용) | 2h compute + 30min code | chartqa_pro 0 → ~28% informative |
| **mini-GRPO 진입 전** | M5+M6 reward function 구현 | 2h | train_grpo_dapo.py 통합 |
| **mini-GRPO 후 retro** | M4 measurement compute 절약 (대량 학습 데이터 생성 시) | 1h | wall-time 30% 절약 |

---

## 5. Open question (decision 필요)

1. **M2 의 chartqa_pro judge prompt template**: `Yes/No` 형식 vs `Verdict: Correct/Incorrect` 형식? Recommend Yes/No (chartmuseum 와 일치).
2. **M3 threshold strictness**: `mc_w ≤ 0.3` 가 적당한가? 아니면 더 strict `mc_w ≤ 0.2`?
3. **M4 exploration step skip**: cost-saving 외에 measurement 품질 향상 효과 있는가? → 별도 ablation 필요 (mini-GRPO 학습 비교).
4. **M6 routing**: bench label 기반 routing vs gold-format 기반 routing? Gold-format 권장 (paper 에서 bench-agnostic 주장 강화).

---

_End of modification proposal. mini-GRPO 진행 결정 시 M1+M2+M3 즉시 적용, M5+M6 학습 코드에 통합 권장._
