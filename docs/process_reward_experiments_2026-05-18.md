# Process Reward 실험 결과 — Tier 1-A verifier sharpening + Image-dependency MC

날짜: 2026-05-18
브랜치: research-chartprm
대상 데이터: `data/d2_hardbench/reports/math_shepherd_dead.jsonl`, `image_dep_dead.jsonl`
샘플 규모: 11 dead samples × 64 steps (chartmuseum, K'=6 sub-rollouts)

---

## 0. 배경

[research_progress_review_2026-05-18.md](research_progress_review_2026-05-18.md) 의 Phase 3 preflight 가 시사한 두 약점:
- 1.71× sample efficiency 주장이 N=39 위에서 신뢰구간 ±5pp.
- VLM verifier mean perception 0.91 (lenient noise floor).

멘토 framework: Math-Shepherd MC 와 perception 은 *complementary failure mode* — MC dead zone 에서 perception 이 살려야 진짜 paper contribution spine.

본 문서는 (A) verifier sharpening 으로 perception salvage 살리기 시도, (B) parsing-free 대안인 image-dependency (with/without image MC) 두 실험의 결과를 정리.

---

## 1. 실험 A — Verifier sharpening (Tier 1-A)

### Hypothesis

lenient verifier (mean 0.99, NO rate 0.4%) 를 엄격히 다시 쓰면 NO rate ≥10% 로 올라가서 Design A 의 perception-salvage 분기가 진짜 학습 신호를 만들 것이다.

### Method

- 기존 `STEP_4AXIS_PROMPT` (preflight_reward_v1.py:81) 를 strict 버전으로 교체.
  - 핵심 추가 rule: "YES 는 chart 에서 directly readable 한 경우만, hedge word ("approximately", "around"...) 가 있고 정확값 미확인 시 NO. 5% 초과 오차 NO. 애매한 claim 은 NA 보다 NO 우선."
- 동일 64 step 에 strict prompt 로 perception 재측정. MC sub-rollout 결과는 그대로 재사용 (정책 무변).
- 스크립트: `.worktrees/research-chartprm/scripts/verifier_sharpen_remeasure.py`
- 결과 파일: `data/d2_hardbench/reports/math_shepherd_dead_sharpened.jsonl`

### Result (구 verifier vs strict verifier)

| metric | 구 (original) | strict (sharpened) |
|---|---|---|
| mean perception | 0.992 | **1.000** (오히려 악화) |
| YES rate (axis-level) | 39.1% | 19.5% |
| NO rate | 0.4% | **0.0%** |
| NA rate | 60.5% | **80.5%** |
| salvage zone (MC-dead step 의 perception variance) | 0/44 | **0/44 (변화 없음)** |
| Design A step-reward variance 회복률 | 9/11 | 10/11 (한 sample 만 +) |

### Diagnosis

sharpening 의 의도는 *NO* 가 늘어나는 것이었으나, 실제로는 *NA* (해당 claim 자체 없음) 가 늘어남.

직접 읽어본 step text 예시:
- `chartmuseum_759/qwen3vl_4b step1`: "Step 1: Locate the bar representing 'Total emissions' for each income group." — 구체적 chart claim (numeric/label) 없는 generic instruction. strict verifier 가 정확히 NA 판정.
- `chartmuseum_574/qwen3vl_4b step2`: 18014 char 의 cheese name listing — 거의 label 만 줄줄. verifier 가 거의 NA.

**결론**: chartmuseum dead 의 thinking trace 는 explicit numeric/label claim 보다 instructional/narrative 위주라 verifier 가 verify 할 statement 자체가 부족. **어떤 verifier 든 (lenient/strict) free-form thinking 에서 perception 신호 추출 불가** — 사용자 직관 ("parsing 본질적 broken") 데이터로 입증.

→ **VLM perception verifier 기반 reward 폐기 정당화** (HIGH confidence negative result).

---

## 2. 실험 B — Image-dependency (with/without image MC)

### Hypothesis

VPPO (ICLR 2026) 의 token-level visual dependency 를 step-level 로 확장: 같은 prefix 에서 image 있을 때 vs 없을 때 K' sub-rollout 의 outcome 분포가 다르면, 그 step 은 chart-grounded. `visual_dep(k) = mc_with(k) − mc_without(k)` 가 parsing 없이 step 의 visual dependency 를 직접 측정.

### Method

- 기존 `math_shepherd_dead.jsonl` 의 64 step 각각에 mc_without_image (K'=6 sub-rollouts from same prefix but text-only user msg) 측정.
- 정책: qwen3vl-4b-instruct on port 8501.
- Judge: 397B remote (relaxed_correctness for chartqa_pro, judge LLM for chartmuseum/charxiv).
- 스크립트: `.worktrees/research-chartprm/scripts/image_dep_mc.py`
- 결과 파일: `data/d2_hardbench/reports/image_dep_dead.jsonl`

### Pass criterion (멘토 spec §4)

1. visual_dep std ≥ 0.15 (의미있는 spread).
2. 4-cell 분포 (mc_high × vd_high) 각 cell ≥ 10%.
3. MC-dead step 중 visual_dep > 0.2 ≥ 10 (image-critical 회복 zone).

### Critical bug 발견 — judge 가 empty pred → "Yes" 리턴

K'=6 sub-rollout 중 일부가 빈 문자열로 generation 됨 (prefix 가 이미 `</think><answer>...</answer>` 포함 시 모델 즉시 stop). 그런데 outcome=1 로 평가됨.

#### Root cause

`extract_pred(empty_text)` → `""`. 그 후 chartmuseum 의 judge prompt:

```
Is the model's answer correct vs ground truth?
Q: I am picking a safe pincode...
GT: 9520
Model: 
Reply only Yes or No.
```

직접 397B 에 5회 전송:

```
try0: 'Yes'
try1: 'Yes'
try2: 'Yes'
try3: 'Yes'
try4: 'Yes'
```

→ 397B 가 빈 "Model:" 필드를 "disagree 안 함" 으로 해석, **deterministic 하게 Yes** 리턴. 

#### Bug 영향

- `math_shepherd_dead.jsonl`: 19 with-image outcome 이 fake-positive (4 step 에 분포).
- `image_dep_dead.jsonl`: 19 with-image + 6 without-image = **25 fake-positive outcome**.
- 합계 **44 fake-positive** 가 mc_with / mc_without 통계 오염.

#### Fix

`score_subrollout` 에 가드 추가:
```python
async def score_subrollout(text, sample, judge_client, judge_model, judge_sem):
    pred = extract_pred(text)
    if not pred.strip():
        return 0, pred  # 빈 pred 는 verifier 에 보내지 말고 0 처리
    ...
```

스크립트: `.worktrees/research-chartprm/scripts/math_shepherd_dead.py:90`, `image_dep_mc.py:71` 둘 다 수정.

기존 데이터는 `/tmp/refix_outcomes.py` 로 in-place 재평가 (44 outcome → 0). 백업 `.bak_empty_pred` 보존.

### Result (수정 후)

#### Pass criterion 결과

| criterion | target | 실제 (수정 후) | 판정 |
|---|---|---|---|
| C1: visual_dep std ≥ 0.15 | 0.15 | **0.215** (range −0.667 ~ +0.667) | PASS |
| C2: 4-cell each ≥ 10% | 4/4 | 1/4 (case3 90.6% dominant, 다른 셀 ≤8%) | PARTIAL FAIL |
| C3: MC-dead step vd > 0.2 ≥ 10 | 10 | **0/47** | FAIL |
| (추가 관찰) MC-dead step vd < −0.1 | — | 11/47 = 23% | image misleading 신호 발견 |

#### visual_dep histogram

| bin | n | % |
|---|---|---|
| strongly_neg [−1.0, −0.3) | 8 | 12.5% |
| mildly_neg [−0.3, −0.1) | 9 | 14.1% |
| zero [−0.1, +0.1) | 38 | 59.4% |
| mildly_pos [+0.1, +0.3) | 4 | 6.2% |
| strongly_pos [+0.3, +0.6) | 4 | 6.2% |
| very_strong_pos [+0.6, +1.0] | 1 | 1.6% |

#### Reward design 비교 (per-sample step-reward std, fixed 후)

| design | non-zero std 회복률 | 비고 |
|---|---|---|
| R_out (outcome only) | 0/11 (0%) | by construction |
| R_MC (mc_with) | 7/11 (64%) | Math-Shepherd 정통 |
| R_VPPO (user spec) | 7/11 (64%) | R_MC 와 동일 coverage |
| R_VPPO+ (negative-vd penalty) | 7/11 (64%) | 3 sample 에서만 std 마진 증가 |

---

## 3. VPPO+ penalty 의 진짜 의미 — 6 unique prompt 직접 읽기

수치 corr 만 보고 결론 내리지 않고 step text 직접 검토.

### 직접 읽은 결과 — penalty 7개가 모두 1 prompt (chartmuseum_370) 에서 나옴

| prompt | 질문 type | penalty | bonus | 해석 |
|---|---|---|---|---|
| chartmuseum_370 (PIN safe) | 4지선다, 정답 9520 | **7** | 1 | 거의 모든 penalty 가 여기서 |
| chartmuseum_574 (sheep cheese 수) | open count | 0 | 0 | image 필요, vd>0 standard |
| chartmuseum_6 (Star Wars 영화 수) | open count | 0 | 0 | image 필요, vd>0 standard |
| chartmuseum_759 (greenhouse emissions) | label | 0 | 0 | hard-impossible (mc=mc_without=0) |
| chartmuseum_882 (Simpsons 캐릭터) | open count | 0 | 0 | 신호 없음 |
| chartmuseum_964 (Sasquatch 주) | label | 0 | 0 | 거의 신호 없음 |

### chartmuseum_370 이 special 한 이유 (penalty 가 trigger 되는 진짜 메커니즘)

질문 = "9520/1995/9900/6060 중 가장 안전한 PIN?" + 정답 = 9520. Three leakage routes:

1. **4지선다 → text-only base rate 25%** (무작위).
2. **9520 의 folklore prior**: text-only LLM 이 "safe PIN" 검색 결과 / 학습 데이터에서 9520 을 자주 본 prior.
3. **Prefix verbalization**: 예 chart_r1 step3 prefix = "Step 4: Compare with most common and least common lists. Step 5: Pincode 9520 is not in either list..." 으로 시작 — image 없어도 prefix 의 logical chain 만으로 9520 결론 도달 가능.

⇒ mc_without 가 0.5–0.7 인 건 "image misleading" 신호가 아니라 **"이 질문은 text 만으로 풀린다"** 신호.

### 다른 prompt 대비 검증

- **chartmuseum_574** step1 "Sheep section, Hard subcategory" — mc_without preds = `[1, 1, 1, 3, 1, 2]` (전부 random low number guess). vd = +0.17 — image 필요.
- **chartmuseum_6** step1 "movies 식별 + 600M 이상 count" — mc_without preds = `["If you have a specific chart in mind...", "But let..."]` (hedging). vd = +0.67 — image 절대 필요.
- **chartmuseum_964** (Sasquatch Wyoming) — mc_without preds = `"*(Note: This is based on assumptions, as no figure was provided...)*"`. vd = +0.17.

⇒ 이 prompt 들에서 **vd 가 일관되게 ≥ 0** — image 가 진짜 필요하고 misleading 안 됨. penalty branch 절대 fire 안 함.

### 진단

- **Hypothesis**: negative visual_dep 가 "image 가 misleading 한 step" 의 일반 신호.
- **Reality**: 64 step 중 negative vd 17개 (27%) 중 **15개가 chartmuseum_370 에서 옴**. 이 prompt 는 question text + folklore prior 로 풀리는 leakage-prone prompt. 다른 5개 prompt 에서 image-misleading 신호 없음 (mc_without 거의 0 이라 vd 가 측정 자체 무의미).
- **결국 VPPO+ 의 penalty branch 는 image-misleading detector 가 아니라 *leakage-prone prompt detector***. n=1 effective prompt.

### Reward design 차원 의미

| 용도 | 가능 여부 | 비고 |
|---|---|---|
| Reward source (학습 advantage) | 불가 | n=1 prompt, 일반 prompt 에서 fire 안 함, mc_with 와 coverage 동일 |
| Dataset filter (leakage prompt 식별) | 가능 | mc_without ≥ 0.3 → leakage 의심 → training 에서 제외 또는 별도 분류 |
| Paper Figure 1 (failure-mode taxonomy) | 가능 | "image-misleading vs image-critical vs hard-impossible" 시각화 |

---

## 4. 종합 결론

### 살아남은 paper contribution

1. **Step-level Math-Shepherd MC on chart reasoning** (HIGH confidence)
   - 11 sample 중 7개에서 step-reward variance 회복 (vs outcome-only 0개).
   - wrong_consistent step 의 26% 가 step-level MC variance 보유.
   - Reward = `mc_with(k)` (Math-Shepherd 정통).

2. **Image-dependency as dataset diagnostic** (MEDIUM confidence)
   - 23% MC-dead step 이 negative visual_dep 보유 (image misleading + text leakage 두 신호 혼재).
   - VLM perception verifier 가 절대 못 잡던 failure mode.
   - 단 reward source 로는 불가, dataset filter / paper Figure 1 으로 강등.

3. **Hard-impossible dead prompt classification** (HIGH confidence)
   - `chartmuseum_759` 류 (mc_with=mc_without=0 across all steps with 3 different baseline traces) = 진짜 학습 불가능 prompt.
   - Filter rule: 모든 step 에서 K'=6 sub-rollout outcome 합 = 0 + mc_without 합 = 0 → drop.

4. **Leakage-prone prompt detection** (NEW from image-dep)
   - chartmuseum_370 류 (mc_without ≥ 0.3 in any step) = text + general knowledge 로 풀리는 prompt.
   - Image-grounded reasoning 학습에 noise → 별도 분류 또는 제외.

### 폐기된 가설

1. **VLM perception verifier 기반 reward** (HIGH confidence negative)
   - Lenient → 0.91 noise floor, strict → 80% NA. 어느 방향이든 신호 추출 불가.
   - Free-form thinking trace 에 explicit chart claim 부재가 근본 원인.

2. **Complementary Salvage (Design A) 의 perception salvage 분기** (HIGH confidence negative)
   - 0/44 MC-dead step 에 perception variance — salvage zone 비어 있음.
   - 0.3·perception fallback 이 constant 0.3 (perception=1.0 lenient 일 때).

3. **VPPO+ penalty 가 일반 image-misleading detector** (MEDIUM confidence negative)
   - 7/7 penalty 가 1 leakage-prone prompt 에서. 다른 prompt 에서 trigger 0.

### 폐기 사유 (방법론)

- 데이터 quality 버그 (empty pred → judge "Yes") 가 44 outcome 을 fake-positive 로 만들었음. Bug fix 후 bonus signal 사실상 소멸.
- delta_mc (cross-step) 와 visual_dep (cross-modality) 는 직교 축. VPPO+ 의 reward 식이 두 axis 를 혼합하면서 일관성 잃음.

---

## 5. 다음 단계 (제안)

### Tier 1-B — N=200 scale-up + bench diversity

현재 11 sample × chartmuseum 만 → 200 sample × 3 hard bench (ChartMuseum/CharXiv/ChartQA-Pro) 로 확장.

- 통계 신뢰구간 ±3pp 안.
- VPPO+ penalty 가 다른 bench / 다른 question type 에서 fire 하는지 확인.
- leakage-prone prompt 비율 측정 (전체 학습 dataset 의 몇 %?).
- 비용: 200 sample × 6 step × (6 with + 6 without) = 14400 sub-rollout, ~30분 wall-clock.

### Tier 2 — Mini-GRPO 학습 곡선 비교 (3 design × 200 sample)

| design | 수식 | 가설 |
|---|---|---|
| R_outcome | binary outcome | baseline |
| R_MC | `mc_with(k)` | Math-Shepherd 정통 |
| R_MC + leakage filter | `mc_with(k)`, filter out mc_without ≥ 0.3 prompts | dataset clean + Math-Shepherd |

통과 기준: R_MC + filter 가 R_outcome 대비 step 30 이전 추월, final accuracy ≥ +2pp on 2 of 3 benches.

### 결정 필요

1. Tier 1-B (N=200 scale-up) 진행?
2. Reward 식을 **mc_with 단독 (Math-Shepherd 정통)** 으로 확정?
3. Leakage-prone prompt filter (mc_without ≥ 0.3) 를 정식 dataset cleaning rule 로 등록?
4. Hard-impossible dead prompt filter (mc_with=mc_without=0 across all steps) 를 정식 dataset cleaning rule 로 등록?

---

## 6. 아티팩트 (재현 가능성)

### 스크립트

| 파일 | 역할 |
|---|---|
| `.worktrees/research-chartprm/scripts/math_shepherd_dead.py` | Math-Shepherd MC sub-rollout (K'=6) |
| `.worktrees/research-chartprm/scripts/verifier_sharpen_remeasure.py` | strict verifier 재측정 |
| `.worktrees/research-chartprm/scripts/sharpening_compare.py` | original vs strict perception 비교 |
| `.worktrees/research-chartprm/scripts/image_dep_mc.py` | mc_without_image 추가 측정 |
| `.worktrees/research-chartprm/scripts/image_dep_analyze.py` | 3 pass criterion + 4 reward design 비교 |
| `/tmp/penalty_alignment_check.py` | VPPO+ vs delta_mc alignment |
| `/tmp/penalty_content_check.py` | step content 직접 검토 |
| `/tmp/diverse_content_check.py` | 6 unique prompt 별 step content |
| `/tmp/refix_outcomes.py` | empty-pred bug fix |

### 데이터

| 파일 | 내용 |
|---|---|
| `data/d2_hardbench/reports/dead_samples.jsonl` | 29 dead samples 식별 (11 MC-eligible) |
| `data/d2_hardbench/reports/mc_input.jsonl` | 11 dead samples + 원 trace join |
| `data/d2_hardbench/reports/math_shepherd_dead.jsonl` | MC with-image 결과 (empty-pred fix 적용) |
| `data/d2_hardbench/reports/math_shepherd_dead.jsonl.bak_empty_pred` | fix 전 백업 |
| `data/d2_hardbench/reports/math_shepherd_dead_sharpened.jsonl` | strict verifier 재측정 |
| `data/d2_hardbench/reports/image_dep_dead.jsonl` | mc_with + mc_without (fix 적용) |
| `data/d2_hardbench/reports/image_dep_dead.jsonl.bak_empty_pred` | fix 전 백업 |
| `data/d2_hardbench/reports/image_dep_design_eval.json` | 3 pass criterion + 4 reward design 집계 |

### 서버

- 8500 GPU10 qwen3vl-8b-thinking (verifier).
- 8501 GPU11 qwen3vl-4b-instruct (policy).
- 10.1.211.147:8000 397B (judge).
