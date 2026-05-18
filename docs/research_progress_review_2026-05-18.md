# ChartVCR Research Progress Review — 2026-05-18

> 리뷰/피드백 요청용 진척 보고서. 원래 연구계획 대비 현재까지의 시도·결과·방향 전환을 상세히 기록.

---

## 1. 원래 연구계획 (baseline)

**Title**: "Verification-Aware RL for Chart Reasoning: Co-Designing Data and Rewards for Process-Level Supervision" (v8 plan).

**주요 contributions (원래 계획)**:
- C1. **Process-augmented reward** — outcome 단독이 아닌 perception (grounding) + arithmetic 결합한 GRPO process reward.
- C2. **Verification-aware data design** — process reward 학습을 위해 설계된 SFT/GRPO 데이터셋 (chart diversity, multi-step reasoning, deterministic intermediates).
- C3. **GRPO 에서 process reward 작동 조건** 의 포괄 분석.

**Reward 함수 (원 설계)**:
```
r = r_outcome (binary) + 0.3 · r_process
r_process = ??? (perception verifier × MC rollout 결합, 미확정)
```

**핵심 가정**: chart reasoning trace 에 대해 perception verifier (chart-fact 검증) + Math-Shepherd MC rollout (step value 추정) 두 신호가 GRPO 의 dense advantage 를 만들어 outcome-only 대비 학습 가속.

**Position 명확화 (recent 확인)**: perception+MC 는 **GRPO inline reward** 계산용 (SFT 데이터 생성용 아님). v8 plan `compute_process_reward_fast` 함수 자리. PRM 별도 모델 학습 없이 매 batch step-wise reward 직접 산출.

---

## 2. Phase 1 — Easy bench validation (D1+D2 pilot)

### 2.1 시도
- ChartQA-train (69) + ReachQA-train (31) = 100 sample × 3 baselines (zeroshot_4b, chart_r1_7b, chartgemma_12b).
- D1: verifier 신뢰도 (CSV ground-truth 50 sample). InternVL2.5-26B agreement 90.1% PASS.
- D2: 100 sample MC rollout K=8, claim Jaccard, perception AUC, 4-cell distribution.

### 2.2 결과
- **Reliability PASS**: 90.1% (197 query, coverage 97.5%) — verifier 자체 신뢰성 검증 통과.
- **Alignment FAIL** (1/4 metrics PASS): perception_auc 0.481, mc_auc 0.930, r(p,outcome) = **‑0.068**, lift +0.001.
- **4-cell 분포** (per-source rescore 후):
  - zeroshot_4b: GroundedCorrect 48.5%, CarefulFlawed 30.4%
  - chart_r1_7b: GC 48.5%, CF 27.3%
  - chartgemma_12b: GC 24.2%, CF 35.4%

### 2.3 진단
- Easy bench 에서 perception 점수는 outcome 과 **decoupled** (r ≈ 0). Model 이 chart-reading 은 잘 함 (saturate) → drift signal 미작동.
- Pattern B (MC-only pivot) 결정 — perception 폐기, MC-only Math-Shepherd PRM 으로 narrative 변경 검토. 또는 hard chart bench 검증으로 판단 유보.

### 2.4 산출물
- `scripts/d2_claim_extract_v2.py`, `d2_perception_verify_v2.py`, `d2_alignment_v2.py`, `d2_4cell_distribution.py`, `d2_rescore_per_source.py`.
- `docs/d1d2_decision_report.md`, `data/d2_pilot/4cell_rescored.json`.

---

## 3. Phase 2 — Hard bench validation (12 combos)

### 3.1 시도
- "Hard chart bench 에서는 perception 신호 회복?" 가설 검증.
- **Benchmark**: ChartQA-Pro (LMMs-Eval deterministic) + CharXiv-Reasoning (CharXiv official judge) + ChartMuseum (ChartMuseum official judge).
- **Baselines**: Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Thinking / Chart-R1 7B / ChartGemma 12B.
- **Scale**: 4 baselines × 3 benches × 100 samples (seed 42) = 1200 inference traces.
- **Stack**: extractor Qwen3.6-27B (LLM-primary claim extraction), verifier InternVL2.5-26B (NOT_FOUND fallback), judge Qwen3.5-397B-A17B-FP8 (CharXiv/ChartMuseum official prompt).

### 3.2 결과 (final 12-cell, `docs/d2_hardbench_decision_report.md`)

| Bench | median pAUC | median r(p,o) | 평가 |
|---|---:|---:|---|
| Easy (Phase 1) | 0.481 | -0.068 | saturated, decoupled |
| ChartQA-Pro | 0.708 | +0.23~+0.45 | 부분 신호 |
| CharXiv-Reasoning | 0.559 | 평균 +0.10 | borderline |
| **ChartMuseum (가장 어려움)** | **0.449** | **모두 −0.05~−0.25** | **신호 부재, anti-correlated** |

- 초기 "pAUC 0.71-0.75 dramatic recovery" 주장은 ChartQA-Pro 3개 combo cherry-pick. 12 combos 전체로 보면 median 0.55 — easy 대비 marginal.
- **ChartMuseum 에서 perception 신호 가장 약함** (음의 r). "어려울수록 신호 회복" 가설 부정.

### 3.3 진단
- ChartGemma chartmuseum 1차 시도 100/100 inference error (max_tokens=8192 > max_model_len=4096 config bug). 재실행 (max_model_len=8192 native, max_tokens_override=4096) 후 정상 수집. 그러나 ChartGemma 자체가 chartmuseum 의 `<think>...</think><answer>` 포맷 미준수 → 92% unknown cell (단답만 emit, reasoning step 없음).
- ChartQA-Pro/qwen3vl_8b_thinking: outcome_pct 2% (thinking 분량 과대로 token budget 소진 → 답 미생성).

### 3.4 산출물
- `scripts/d2_hardbench_to_pilot.py`, `d2_hardbench_outcome.py`, `d2_hardbench_4cell.py`, `d2_hardbench_artifact_audit.py`, `d2_hardbench_synthesis.py`, `d2_hardbench_rejudge.py`, `d2_hardbench_4cell_rejudge.py`, `d2_hardbench_cf_taxonomy.py`.
- `docs/d2_hardbench_decision_report.md` (190줄, 12-cell 4cell + rejudge + cf_no_answer + artifact_audit + bench rollup).

---

## 4. Phase 2.5 — Root cause 재분석 (방향 전환)

### 4.1 시도 1 — cf taxonomy (perception ≥0.5 subset)
- cf cell (perception OK + outcome wrong) 426 sample 을 397B 로 6-class 분류.
- 결과: **CATEGORY_SELECT 46% (dominant)** / TRUNCATION 26.5% / ARITH 8.5% / Q_INTENT 6.3% / FMT 4.7% / OTHER 8.0%.
- 해석 (잠정): "perception OK 인데 답 틀림 → category-select reasoning step 오류 가 주범".

### 4.2 시도 2 — ALL wrong samples root cause (948 sample)
- 사용자 피드백: cf cohort 만 보지 말고 **모든 wrong sample** (948개) 의 thinking 내 first-error 분석.
- 397B 가 10-class root cause 분류.
- 결과 (substance perspective):

| Root cause | All wrong | chartqa_pro | charxiv | chartmuseum |
|---|---:|---:|---:|---:|
| **CHART_VALUE_MISREAD** | **50.6%** | 31% | 61% | **64%** |
| TRUNCATED | 21.4% | 17% | 24% | 24% |
| PREMATURE_CONCLUSION | 15.3% | 34% | 8% | <1% |
| 그 외 (Q_INTENT/FILTER/MAPPING/ARITH/FMT) | 12.7% | 18% | 7% | 12% |

- Cross-check: cf=CATEGORY_SELECT 라벨 160개가 root-cause=CHART_VALUE_MISREAD 로 재분류 → **claim-verifier (InternVL2.5-26B) 가 count/aggregate/spatial perception 진술 17% false-pass** (entity-claim only scope 한계).

### 4.3 시도 3 — 사용자 지적 후 직접 sample 읽기 (manual)
- "397B 위임 = 직접 분석 아님" 피드백.
- 15 wrong sample (3 bench × 5) 의 reasoning trace 를 직접 읽고 first-error step 수동 annotate.

직접 분류 결과 (N=15):

| Stage | 카운트 | % |
|---|---|---|
| **PERCEPTION (first chart-fact 진술 오류)** | 8 | 53% |
| TRUNCATED (signal 없음) | 3 | 20% |
| SHALLOW (chartgemma capability) | 2 | 13% |
| MAPPING (perception OK, wrong group) | 1 | 7% |
| Q_PARSING (Unanswerable/filter 미인지) | 1 | 7% |
| COMPUTATION | 0 | 0% |

verbatim 4-pattern:
- **VALUE**: charxiv_1838 — "red circle at d=70 ≈ 26.90%" (gold 40).
- **LABEL/LEGEND**: chartmuseum_604 — 잘못된 legend name 단언.
- **COUNT**: chartmuseum_6 — "4 movies above 600M" (gold 5).
- **STRUCTURE**: chartmuseum_759 — bar=area encoding 을 height-only 로 단정.

### 4.4 방향 전환 진단
- 초기 "first-step eager declarative commit + no self-check" 행동 패턴 framing 제안 → **사용자 지적**: GRPO 학습 중 self-check 행동은 K=8 rollout 모두에서 거의 안 나옴 → reward variance ~0 → policy gradient ~0 → 학습 실패.
- **Substance-stage framing 으로 수정**: 행동 변화 (self-check 유도) 가 아니라 substance accuracy (현재 thinking 스타일 유지한 채 step-별 chart-fact 정확도) 가 reward target.
- Perception-family coverable ≈ **60% wrong samples** (PERCEPTION 53% + MAPPING 7%). Substance-signal 가능 sample (TRUNCATED/SHALLOW 제외) 의 80% 가 perception target 으로 커버.

### 4.5 산출물
- `scripts/d2_hardbench_cf_taxonomy.py` (cf 426 sample, 6-class)
- `scripts/d2_root_cause.py` (all wrong 948 sample, 10-class)
- `data/d2_hardbench/reports/cf_taxonomy_summary.json`, `root_cause_summary.json`

---

## 5. Phase 3 — Reward 설계 preflight (현재 진행 — review 시점)

### 5.1 설계 가정
- 기존 perception verifier (InternVL2.5-26B, claim-level) 의 결함:
  - A. Entity-claim 위에 얹힌 quantifier/superlative ("lowest", "highest") 미검증.
  - B. 색·구조 semantic ("dark=least common") 미검증.
  - C. Axis-encoding 진술 (bar=height vs area) outside scope.
  - D. Sample-level aggregate score → step-level credit 손실.
  - E. Model 이 fact 아예 emit 안 하면 unknown 처리 (SHALLOW 무신호).
- 보완 방향: **4-axis multi-grained per-step VLM verifier** (VALUE / LABEL / COUNT / STRUCTURE).

### 5.2 구현
- `scripts/preflight_reward_v1.py`: 50 stratified wrong sample (anchor 15 + 35 stratified), step segmentation, 1-step → 1 VLM call (4 axis 통합 prompt), 397B judge 결합.
- VLM verifier: **qwen3vl-8b-thinking** (port 8500, GPU 10) — policy 4B 와 family-compatible, chart 강함, thinking 모드로 깔끔한 YES/NO 추출.
- Policy: **qwen3vl-4b-instruct** (port 8501, GPU 11) — MC rollout source.

### 5.3 시도 1 — Verifier-only preflight (50 sample, 기존 trace 사용)

목적: VLM verifier 의 4-axis 판별 정확도 검증.

결과:
- Per-axis verdict: VALUE PASS 73% / LABEL 89% / COUNT 82% / STRUCTURE 86%. 평균 perception 0.91.
- **68% wrong sample 이 perception ≥0.9** → verifier 가 chart-reading OK 로 판정 (실제 오류는 chart-reading 외 영역, 또는 verifier false-pos).
- 명확한 false-pos 사례: `charxiv_1855` — trace 에 "z-axis ≈ 1.5" (gold 0.5) 명시되었으나 verifier 가 VALUE=YES. VLM 자체 한계.
- 명확한 true-neg 사례 (low perception): chartgemma charxiv_172/1729/1838 — 모델이 chart 진술 자체 거의 안 함 → 0.0.

### 5.4 시도 2 — GRPO learning-signal preflight (50 prompt × K=4 fresh rollout)

목적: 실제 GRPO 학습 시 reward variance 가 발생하는지 진단.

설계:
- 각 prompt 에 대해 4B policy 가 K=4 fresh rollout 생성 (temperature 0.8, top_p 0.95, max_tokens 2048).
- 각 rollout 별: perception (per-step verifier 평균) + outcome (397B judge).
- 4 reward 함수 비교:
  - `R_out` = outcome (binary)
  - `R_per` = perception (continuous)
  - `R_mult` = perception × outcome (multiplicative)
  - `R_add@α` = α·perception + (1-α)·outcome (additive, α ∈ {0.3, 0.5, 0.7})

핵심 metric: **fzs (frac_zero_std)** — 한 prompt 의 K rollout 들이 모두 같은 reward 면 GRPO advantage = 0 → 학습 신호 없음. fzs 낮을수록 학습 신호 dense.

결과 (N=39 valid prompts):

| Bench | fzs_out | fzs_per | fzs_mult | fzs_add0.5 | <out> |
|---|---:|---:|---:|---:|---:|
| chartqa_pro (N=3) | 1.00 | 0.67 | 1.00 | **0.67** | 0.00 |
| charxiv (N=17) | 1.00 | 0.88 | 0.88 | **0.88** | 1.00 |
| chartmuseum (N=19) | 0.58 | 0.74 | 0.58 | **0.42** | 0.21 |
| **ALL (N=39)** | **0.79** | **0.79** | **0.74** | **0.64** | **0.54** |

핵심 발견:

1. **Multiplicative (perc × out) REJECTED**: chartmuseum 58% 가 outcome=[0,0,0,0] → mult=[0,0,0,0] → std=0. "perception 으로 outcome 의 0-variance 살리기" 불가능 (사용자 직관 검증 실패).
2. **Additive (α·p + (1-α)·o) ACCEPTED**: fzs 0.79 → 0.64 (ALL), chartmuseum 0.58 → 0.42, chartqa_pro 1.00 → 0.67.
3. **GRPO 효율 환산**: outcome-only 21% effective sample → additive 36% → **1.71× sample efficiency**.
4. **Step-level perception variance** (per-rollout 안에서): chartmuseum 0.203 (의미 있음), 다른 bench 0.000. → chartmuseum 은 per-step credit assignment (Math-Shepherd 식) 시 추가 신호 잠재력.
5. **잔여 dead 11 prompts** (모든 reward 설계 std=0): 정책이 정답 일관적 (charxiv_60/1661 — 학습할 필요 없음) 또는 K=4 rollout 들의 perception 이 모두 동일 (rollout diversity 부족).

신호 부활 사례 (additive 가 살린 prompts):
- `chartmuseum_574`: outs=[0,0,0,0], perc=[0.67,1.0,1.0,1.0] → add0.5 std=0.072. 모두 fail 인데 perception 다양 → policy 가 perception 1.0 방향 학습 유인.
- `chartqa_pro_1034 (chart_r1)`: outs=[0,_,0,0], perc=[1.0,_,0.0,0.0] → add0.5 std=0.236. 대 신호.

### 5.5 진단 (현재 시점)

가설: perception × MC multiplicative gating 이 GRPO 학습 신호 dense 화.
**현실**:
- Multiplicative 는 outcome=0 dominant case 자기 자신을 0 으로 만듦 → unviable.
- Additive combined `r = 0.5·perc + 0.5·out` 가 36% sample 에 signal 생성 (outcome-only 1.71×).
- ChartMuseum step-level variance 0.203 활용 시 per-step credit assignment 추가 효과 기대.
- VLM verifier (qwen3vl-8b-thinking) 가 평균 perception 0.91 으로 lenient → sharpening 필요.

### 5.6 산출물
- `scripts/preflight_reward_v1.py` — 4-axis VLM verifier + MC rollout (옵션).
- `scripts/preflight_grpo_signal.py` — K rollout × reward variance 측정.
- `scripts/preflight_grpo_signal_analyze.py`, `preflight_grpo_signal_analyze_v2.py` — fzs 분석.
- `data/d2_hardbench/reports/preflight_50.jsonl`, `preflight_reward.jsonl`, `preflight_grpo_signal.jsonl`, `preflight_grpo_signal_v2.json`.

---

## 6. 현재 결정 사항 + 미결 의문

### 6.1 채택된 설계 (current)
- **Reward**: `R = 0.5 · perception_avg + 0.5 · outcome` (additive, multiplicative 폐기).
- **Verifier**: qwen3vl-8b-thinking, 1-step → 1 VLM call, 4-axis (VALUE/LABEL/COUNT/STRUCTURE) 통합 prompt, thinking 허용 + max_tokens 1024.
- **Policy**: qwen3vl-4b-instruct, K=4 rollout, temperature 0.8.
- **Judge**: 397B-A17B-FP8 remote 4-host (bench 별 official prompt: relaxed_correctness/CharXiv-judge/ChartMuseum-judge).

### 6.2 알려진 한계
- VLM verifier mean perception 0.91 — over-lenient. Sharpening 필요.
- K=4 시 rollout diversity 일부 부족 (특히 chartqa_pro/8b_thinking 케이스 모두 동일 perception).
- ChartMuseum step-level variance 외에는 step-level credit 의 추가 가치 미확인.
- MC sub-rollout (Math-Shepherd 식 step value) 미구현 — outcome 1개당 K' 추가 sub-rollout 으로 step-별 P(correct) 추정 가능하나 비용 4-8×.

### 6.3 미결 의문 (멘토 피드백 요청)

1. **Reward 설계 — additive α 결정**: α=0.3/0.5/0.7 모두 fzs 동일 (0.64). 학습 중 어떤 α 가 policy convergence 더 빠를지 ablation 필요?
2. **Verifier upgrade**: qwen3vl-8b-thinking → qwen3vl-32b 또는 InternVL3-78B 시 mean perception sharper 가능? 비용-성능 trade-off.
3. **MC sub-rollout 추가 가치**: K' = 4-8 sub-rollout 으로 step value 추정 시 GRPO 신호 추가 향상? 비용 4-8× 정당화 가능?
4. **Step-level credit assignment**: GRPO 는 sequence-level reward 가 표준. Per-step credit 적용은 PPO + GAE 필요. ChartMuseum 만의 step variance (0.203) 가 별도 step-PPO 도입 정당화하기 충분한가?
5. **Bench-targeted training**: ChartMuseum (가장 어려운 bench) 에서 가장 큰 signal recovery (+16pp fzs) → ChartMuseum-anchored training data 가 가장 효과적? 또는 cross-bench 균형 batch?
6. **Paper framing**: 원 v8 plan ("Verification-Aware RL") → 현 결과 ("Additive perception reward gives 1.71× GRPO sample efficiency on hard chart bench, but multiplicative gating fails") 로 narrative 수정 정합한가?

### 6.4 다음 단계 (proposal — 검토 요청)

옵션 A (immediate): Verifier prompt sharpening (mean perception target 0.6-0.7) + K=8 rollout ablation. 1일.
옵션 B (parallel): GRPO 200-sample sanity training dispatch (4B policy × chartmuseum subset, additive reward) — 학습 곡선 4-8시간.
옵션 C (deep): MC sub-rollout 추가 + step-level credit (per-step reward 단계 진입). 3일.

**선호 (제안)**: A → B 순차. A 가 신호 dense 화 후 B 에서 실증.

---

## 7. 종합 self-assessment

- **원 가설**: "Perception + MC 결합 GRPO process reward 가 outcome-only 대비 우수." → **부분 검증** (additive 만, multiplicative 폐기).
- **방향 변경 횟수**: 3회 (Phase 1 perception decoupled → Phase 2 hard bench 검증 → Phase 2.5 substance-stage framing → Phase 3 additive reward).
- **현재 신뢰도 (높은 → 낮은)**:
  - HIGH: Hard chart bench wrong sample 의 50%+ 가 perception error 에서 시작 (948 sample 분석 + manual 15 검증).
  - HIGH: Multiplicative gating 폐기 정당 (outcome=0 dominant 시 자가 무력화).
  - MEDIUM: Additive combined 1.71× efficiency — 50 sample preflight 기반, 실제 GRPO 학습에서 재현 미검증.
  - MEDIUM: 4-axis verifier 의 axis-별 discriminative power — VALUE 가장 효과, COUNT/STRUCTURE 데이터 sparse.
  - LOW: ChartMuseum step-level variance 0.203 이 PPO 도입 정당화 충분한지.
- **리스크**: VLM verifier (qwen3vl-8b-thinking) 가 자체 chart 인식 한계로 false-pos. 결과적으로 perception reward 신뢰성 상한 존재. Stronger VLM 도입 또는 cross-VLM ensemble 검토 필요.

---

## 8. 핵심 산출물 인덱스

| 분류 | 경로 |
|---|---|
| Final 12-cell report | `docs/d2_hardbench_decision_report.md` |
| Phase 1 D1+D2 | `docs/d1d2_decision_report.md` |
| GRPO signal summary | `data/d2_hardbench/reports/preflight_grpo_signal_v2.json` |
| Root cause aggregate | `data/d2_hardbench/reports/root_cause_summary.json` |
| cf taxonomy | `data/d2_hardbench/reports/cf_taxonomy_summary.json` |
| Reward preflight scripts | `.worktrees/research-chartprm/scripts/preflight_reward_v1.py`, `preflight_grpo_signal.py`, `preflight_grpo_signal_analyze_v2.py` |
| Hard bench pipeline scripts | `.worktrees/research-chartprm/scripts/d2_hardbench_*.py`, `d2_root_cause.py` |
| Train script (target) | `train_grpo_dapo.py` (additive reward 통합 미완) |
