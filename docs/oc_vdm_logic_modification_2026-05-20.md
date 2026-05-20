# OC-VDM Logic Modification — Bench-Agnostic Reward Construction

_Date: 2026-05-20_
_Track: research-chartprm_
_Predecessor: `oc_vdm_validation_report.md`_

---

## 0. TL;DR — patch 적용 후 measured outcome

| # | Modification | 적용 | 효과 (measured) |
|---|---|---|---|
| **M1** | Terminal step (k=n_steps-1) OC-VDM 측정 skip | ✓ 적용 (`oc_vdm_analyze_v2.py`) | chartmuseum thinking PatC 1.4%→5.0%, charxiv 18%→21.4% |
| **M2** | chartqa_pro open-ended gold → judge LLM scoring | ✓ 적용 (`rescore_chartqa_pro.py`, 15s total) | **chartqa_pro 0% → 20.2% mean** (qwen3vl_8b_thinking: 0→47.6%, chart_r1: 0→33.3%) |
| **M3** | Pattern B threshold 강화 (`mc_wo≥0.7 AND mc_w≤0.3 AND vd≤-0.4`) | ✓ 적용 (`classify_v2`) | image-as-distractor false positive 제거; PatC% 우선 보존 |
| ~~M4~~ | Claim-density pre-filter | SKIP — compute optimization only | — |
| ~~M5 revised~~ | PatC 보존 + PatB만 confidence-weighted dampen | TODO mini-GRPO 학습 코드에서 | — |
| ~~M6~~ | Question-type routing | SKIP — T/F 케이스 PatC 손실 위험 | — |

**Final measured result (12 combos mean)**: informative% **15.5% → 19.2%** (+3.7pp).

**Per-bench mean informative% (positives 유지 + negatives 일반 raise)**:
| Bench | OLD | NEW | Δ |
|---|---:|---:|---:|
| chartqa_pro | 0.0% | **20.2%** | **+20.2pp** ← 가장 일반적 raise |
| charxiv_reasoning | 23.4% | 24.4% | +1.0pp |
| chartmuseum (chartgemma n=4 제외) | 14.2% | 17.4% | +3.2pp |

**Positive 보존 검증 — PatC% 11/12 combo 증가 또는 동일** (chartgemma_chartmuseum n=4 outlier 외).

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

---

## 6. Final 적용 결과 (2026-05-20 실측)

### 6.1 12 combo 통합 표 (M1+M3 + M2 chartqa_pro only)

| Combo | OLD info% | NEW info% | Δ | OLD PatC% | NEW PatC% | Positive 보존 |
|---|---:|---:|---:|---:|---:|---|
| qwen3vl_8b_thinking__chartqa_pro | 0.0% | **47.6%** | **+47.6** | 0.0% | 38.1% | N/A → 신규 PatC |
| chart_r1__chartqa_pro | 0.0% | **33.3%** | +33.3 | 0.0% | 33.3% | N/A → 신규 PatC |
| qwen3vl_4b__chartqa_pro | 0.0% | 0.0% | 0 | — | — | reasoning trace 부재 (n_steps<2) |
| chartgemma__chartqa_pro | 0.0% | 0.0% | 0 | — | — | 동일 |
| qwen3vl_8b_thinking__charxiv | 34.0% | 32.9% | −1.1 | 18.0% | **21.4%** | ✓ PatC ↑ |
| chart_r1__charxiv | 21.7% | 17.9% | −3.8 | 11.4% | **13.8%** | ✓ PatC ↑ |
| qwen3vl_4b__charxiv | 21.7% | **26.7%** | +5.0 | 8.7% | **13.3%** | ✓ |
| chartgemma__charxiv | 16.1% | **20.0%** | +3.9 | 12.9% | **15.0%** | ✓ |
| qwen3vl_8b_thinking__chartmuseum | 8.9% | **16.2%** | **+7.3** | 1.4% | **5.0%** | ✓ |
| chart_r1__chartmuseum | 17.4% | 16.8% | −0.6 | 4.0% | **5.0%** | ✓ |
| qwen3vl_4b__chartmuseum | 16.2% | **19.3%** | +3.0 | 11.0% | **13.0%** | ✓ |
| chartgemma__chartmuseum | 50.0% | 0.0% | −50 | 25.0% | 0.0% | n=4 sample noise |

**12-combo mean: 15.5% → 19.2% (+3.7pp)**

### 6.2 Insight 1 — chartqa_pro 가 의외의 OC-VDM sweet spot

qwen3vl_8b_thinking_chartqa_pro 의 PatC% 38.1% 는 **charxiv 의 21.4% 보다 훨씬 높다**. 의미:
- chartqa_pro 의 open-ended question 은 image-grounding 이 강하게 요구되는 류 (예: "what graph compares between", "what is unanswerable from chart")
- thinking model 이 reasoning trace 를 생성 + judge LLM 이 semantic match 검출하면 **clean OC-VDM signal**.

→ mini-GRPO 학습에 chartqa_pro thinking-model trajectory 도 적극 포함 권장.

### 6.3 Insight 2 — Positives 보존 확인 (PatC% 단조 증가)

11/12 combo 에서 PatC% 증가 또는 동일. 예외: chartgemma_chartmuseum (n=4 sample noise).

|     | chartqa_pro | charxiv | chartmuseum |
|---|---|---|---|
| qwen3vl_8b_thinking | 0 → 38.1% | 18 → 21.4% | 1.4 → 5.0% |
| chart_r1 | 0 → 33.3% | 11.4 → 13.8% | 4.0 → 5.0% |
| qwen3vl_4b | 0 (no trace) | 8.7 → 13.3% | 11.0 → 13.0% |
| chartgemma | 0 (no trace) | 12.9 → 15.0% | 25 → 0 (n=4) |

### 6.4 Insight 3 — Negative target raised most generally on chartqa_pro

**chartqa_pro 가 가장 high-portion negative target** 이었음 (4 combo × 30 = 120 sample 의 100% 가 fake PatA).
- M2 (judge LLM scoring) 만으로 thinking model 2 combo (60 sample) 의 47.6% / 33.3% 가 informative 으로 회복.
- Instruct 모델 (60 sample) 은 reasoning trace 부재로 M2 만으로는 해결 못 함 — SFT-warmup 단계 필요.

### 6.5 Insight 4 — chartmuseum thinking 의 신호 회복

OLD informative 8.9% → NEW 16.2% (×1.8). PatC 1.4 → 5.0% (×3.6). 가장 어려웠던 combo 가장 큰 절대량 향상.

조합 효과: F1 (no_cap measurement) + M1 (terminal skip) + M3 (PatB tightening) 모두 기여:
- F1: hidden Pattern B 가시화 (이전 0.5% → 측정 후 3.4%, 그러나 M3 tighten 후 0%)
- M1: 25 fake PatA 제거
- M3: 잘못 분류된 PatB → PatE 강등

### 6.6 결론 — bench-agnostic working achieved (with caveats)

**달성된 것**:
- 3 bench 중 2 bench (charxiv + chartqa_pro thinking) 에서 informative% > 20% 달성
- chartmuseum thinking 의 sparse signal 도 PatC ×3.6 회복 (절대 5% 도달)
- Positives 보존 확인

**남은 caveat**:
- Instruct 모델 (qwen3vl_4b, chartgemma) 의 chartqa_pro 는 reasoning trace 부재로 적용 불가 — **SFT-warmup 으로 instruct → thinking-style 분포 부여 필수** (mini-GRPO 진행 시)
- chartgemma 의 chartmuseum 도 step-poor (1% only) — 동일 처방

### 6.7 mini-GRPO 진입 spec (Final)

1. **Policy**: SFT-warmed qwen3vl-4b-instruct (reasoning 분포 부여) 또는 qwen3vl-8b-thinking 직접 GRPO
2. **Training data routing**:
   - Quantitative gold (numeric) → outcome + OC-VDM full (Pattern A drop, B 0.3× dampen, C 1.5× up-weight)
   - Open-ended gold → 동일, 단 scoring 은 judge LLM
   - Bool gold → outcome-only (step structure 없으면 OC-VDM 비적용)
3. **Bench scope**: chartqa_pro + charxiv_reasoning + chartmuseum (3 bench 모두; M1+M2+M3 fix 후 모두 informative signal 존재)
4. **R1-R4 ablation**:
   - R1: outcome-only GRPO baseline
   - R2: + Math-Shepherd MC step value
   - R3: + OC-VDM modulation (M1+M3 classifier, M5 revised modulation)
   - R4: + sample filter (drop PatA-saturated trajectories)

---

_End of report. Patch 모두 적용 완료, mini-GRPO 진입 준비 완료._
