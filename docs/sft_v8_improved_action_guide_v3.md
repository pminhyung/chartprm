# ChartVR Action Guide v3 (2026-04-12)

> **Supersedes**: `docs/sft_v8_improved_plan.md` (v2 regression-targeted plan).
> **Scope**: Full execution path from SFT rebuild → GRPO core → ablation/scale/novelty → paper scaffold.
> **Status**: Approved by user on 2026-04-12. Execution plan mirror at `/home/poc/.claude/plans/wondrous-strolling-stearns.md`.

---

## 5 Strategic Revisions vs v2 Plan

1. **Filler quality gate** — minimum reasoning length raised 40 → **150 chars**. `<150` is hard-rejected; samples flagged with template prefixes (e.g. `Let me analyze`, `I'll analyze`, `Looking at`) and `<300` chars also rejected.
2. **Block 2 filler source limited** — pulls from `block_c` v9 only if `reasoning ≥ 200 chars`. If insufficient, shrink Block 2 target 8K → 7K and shift the 1K shortfall to Block 5.
3. **Scale flexible 28K–30K** — not a fixed 30K target. Quality over quantity. Low-quality filler → dataset stays smaller.
4. **RL algorithm ablation** — add Row 5: SFT → GRPO + conditional_v2, run in parallel with Row 4 (DAPO + conditional_v2). Paper's main method switches to whichever wins.
5. **Novelty framing work** — produce `docs/novelty_framing.md` with VAPV terminology (Value-Anchored Process Verification) and direct differentiation tables vs Chart-RVR (structural conformity) and R1-VL (step-wise reasoning reward).

---

## Part 1 — SFT Data Rebuild (Day 1, ~3h)

### 1.1 Pool loading and verification

Files verified on 2026-04-12:

| Path | Count | Purpose |
|---|---|---|
| `data/sft_hq_teacher.jsonl` | 25,382 | Teacher pool 1 |
| `data/sft_v8_improved_teacher_distill.jsonl` | 4,156 | Teacher pool 2 |
| `data/sft_30k.jsonl` | 30,000 | v8 base |
| `data/sft_v9.jsonl` | 41,940 | v9 supplement |
| `data/sft_hq_lite_clean.jsonl` | 43,242 | hq-lite extra |
| `data/charts_v2/chartvr_train_final.jsonl` | 3,916 | GRPO GT (csv_path present) |

### 1.2 `scripts/assemble_v8_improved_v2.py`

Reuses v1 helpers (`classify_answer`, `classify_question`, `dedup_key`) but replaces the filler selection. New functions:

- `passes_quality_gate(sample)`:
  - `reasoning_length(sample) >= 150` — hard minimum
  - template prefix (`Let me analyze`, `I'll analyze`, `Looking at`) + length < 300 → reject
  - require `<think>` tag in `assistant_text` OR `reasoning_steps`/`reasoning` field present

- `select_block1_chartqa(teacher, target=10_000)`:
  - Phase 1: text+which gold subset (~1,056) → all in
  - Phase 2: text-answer non-gold (~1,170) → all in
  - Phase 3: which-question non-gold (~327) → all in
  - Phase 4: question-type quota (multi_step 3000 / extreme 1500 / counting 1000 / comparison 1500 / other 447)
  - Phase 5: reasoning-length descending, top-`target` per bucket

- `select_block2_scientific(teacher, selfgen, filler, target=8000)` **(revision 2)**:
  - Phase 1: all `scientific_ext` teacher (1204)
  - Phase 2: all `scientific` teacher (2069)
  - Phase 3: all `scientific_ext` self-gen ≥300 (3040)
  - Phase 4: all `scientific` self-gen ≥300 (252)
  - → ~6565 accumulated
  - Phase 5 filler: `block_c` v9 with `reasoning_length ≥ 200` AND `passes_quality_gate`
    - If available_filler ≥ remaining → fill normally
    - Else: shrink target to current + available_filler, record `shortfall_for_block5 = target - actual_target`

- `select_block3_complex(teacher, selfgen, filler, target=6000)`:
  - Phase 1-4: plotly_complex teacher (1235) + kaggle_like teacher (2082) + additional teacher (1032) + plotly_complex self-gen ≥300 (716)
  - Phase 5 filler: block_d/block_f with text-answer priority, passes_quality_gate

- `select_block4_owid_synth_wb(teacher, gt_pool, target=4000)`:
  - owid 2000 + synthetic 1500 + worldbank 500 within teacher, text+which priority
  - dedup against `chartvr_train_final.jsonl`
  - backfill shortage from `chartvr_train_final.jsonl`

- `select_block5_balance(blocks_1_4, remaining_pool, target=2000 + shortfall_for_block5)`:
  - Pass 1: measure actual text% / which% / Y_N% / percentage% / average% from blocks 1-4
  - Pass 2: prioritize samples filling gaps in that order. All samples must pass `passes_quality_gate`.

- `verify_dataset(final)` — CHECK 1-12:

| # | Criterion | Threshold |
|---|---|---|
| 1 | size | ∈ [28_000, 30_000] |
| 2 | reasoning coverage | 100% (no template fallback) |
| 3 | reasoning median | ≥ 300 chars |
| 4 | template-shaped | < 5% |
| 5 | text answers | ≥ 25% (goal 30%) |
| 6 | Y/N answers | ≥ 5% |
| 7 | chartqa_train share | ≤ 35% |
| 8 | 50 random images exist on disk | 0 missing |
| 9 | answer relaxed-match | ≥ 98% |
| 10 | regression-targeted source coverage | kaggle_like ≥ 1500, additional ≥ 800 |
| 11 | source distribution bands | within ±20% of plan |
| 12 | reasoning < 150 | **0 samples** (hard); warn if <300 > 15% |

Run order: `--dry_run --stats` → inspect → non-dry → write `data/sft_v8_improved_v2.jsonl`.

### 1.3 `train_sft.py` fail-fast (already applied 2026-04-12)

At the template-fallback branch, raise `ValueError` when both `assistant_text` and any reasoning field are empty. Verify with 1-sample dry run.

---

## Part 2 — SFT Train + Eval + Decision Gate (Day 1-2, ~8h)

### 2.1 Training

```
accelerate launch --num_processes 8 \
    --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_sft.py \
    --model_path models/qwen3.5-4b \
    --data data/sft_v8_improved_v2.jsonl \
    --output_dir ckpt/sft_v8_improved_v2_4b \
    --use_lora --lora_rank 64 --lora_alpha 128
```

`gradient_checkpointing` reentrant, `flash_attention_2`, `liger_kernel`, ZeRO-3 no offload. Monitor loss < 1.0 by epoch midpoint. Expected ~5h on 8×A100.

### 2.2 Eval

LoRA merge → 8× vLLM TP=1 (one per GPU) → `eval_multi_server.py` on chartqa_human / chartqa_augmented / charxiv_reasoning / chartqa_pro / chartmuseum. Results to `results/sft_v8_improved_v2/`.

### 2.3 Decision gate (per-benchmark, not just AVG)

| Benchmark | minimum | goal | abort |
|---|---|---|---|
| CQA-Human | 83.0 | 85+ | < 80 |
| CQA-Aug | 89.0 | 91+ | < 86 |
| CharXiv | 45.0 | 48+ | < 43 |
| CQA-Pro | 36.0 | 39+ | < 35 |
| ChartMuseum | 29.0 | 33+ | < 27 |
| **AVG** | **56.4** | **59+** | **< 54** |

- All ≥ minimum AND AVG ≥ 56.4 → **proceed to Part 3**.
- AVG ∈ [54, 56.4) → run Block-removal ablation (no_block5, b2_teacher_only), one data-fix + re-train round allowed.
- Any metric below abort OR AVG < 54 → **stop, report to user**. Propose 25K shrink + Block 5 removal, or 397B teacher regen of Block 1 ($25-40).

---

## Part 3 — GRPO Core (Day 3-5, ~30h)

### 3.0 Pre-check

Verify `csv_path` resolvability on disk for `chartvr_train_final.jsonl` (3,916 rows). Missing > 10% blocks VAPV experiments.

### 3.1 Code change (required before any row)

Edit `train_grpo_dapo.py`:
- Add CLI flag `--algo {dapo,grpo}` (default `dapo` for back-compat).
- Replace hardcoded `loss_type="dapo"` at L631 with `loss_type=args.algo`.
- 1-step smoke run for both algos before launching full rows.

**Reward design is frozen**: `conditional_v2` stays. No reward-function redesign beyond the component-ablation masks in Part 4.

### 3.2 Rows

| Row | Base | Algo | Reward | Output |
|---|---|---|---|---|
| 3 | sft_v8_improved_v2_4b | DAPO | outcome_only_v2 | ckpt/row3_dapo_outcome |
| 4 | sft_v8_improved_v2_4b | DAPO | conditional_v2 (VAPV) | ckpt/row4_dapo_vapv |
| 5 | sft_v8_improved_v2_4b | GRPO | conditional_v2 (VAPV) | ckpt/row5_grpo_vapv |

Each ~10h on 8×A100. Sequential on a shared GPU pool; parallelize if pools allow.

### 3.3 Critical gate (all 3 required)

1. **Row 4 > Row 3** by ≥ 2pp AVG AND positive on ≥ 4/5 benchmarks → VAPV > outcome.
2. **max(Row 4, Row 5) > 56.91** (v8 DAPO+process) → new SFT base is worth it.
3. **max(Row 4, Row 5) ≥ 60** → matches/beats v7 Row B (59.9).

Decisions:
- Pass all three → Part 4. If Row 5 > Row 4 → paper's main method switches to GRPO; DAPO becomes an ablation.
- Condition 1 fails → run Step 3.4 diagnostic, one reward/data fix + rerun allowed, then escalate.

### 3.4 Process-reward diagnostic (only if condition 1 fails)

Dump reward distribution for 10 rollouts:
- `outcome` vs `process` score per sample
- CSV parse success rate
- reward variance across rollouts

Expected fixes:
- CSV parse fail > 10% → patch `code/rewards/rule_verifier_fast.py`
- Reward variance ≈ 0 → rescale reward
- `outcome=1 AND process<1` ratio → confirms process adds signal

---

## Part 4 — Ablations, 9B Scale-up, Novelty (Day 6-10)

### 4.1 ABL-7: reasoning-collapse base + VAPV

Train GRPO on `sft_hq_lite_clean_4b` (collapsed base, 51.59 AVG) with `conditional_v2`. Expected result: AVG ≈ 51.6. Proves VAPV needs healthy reasoning as precondition. Critical for paper narrative.

### 4.2 Component ablation

Add to `train_grpo_dapo.py`:
- `reward_conditional_v2_value_only` — mask arithmetic-step sub-rewards.
- `reward_conditional_v2_arith_only` — mask value-extraction sub-rewards.

Unit-test each against 5 hand-crafted rollouts (confirm only the intended sub-reward fires). Run each against v2 SFT base with the winning algo from Part 3.

**Expected pattern**: `value_only > arith_only` (text-entity errors dominate, 56% per error taxonomy). Both single-component < full VAPV → complementarity argument.

### 4.3 9B scale-up ⭐ (mandatory)

| Step | Time | Notes |
|---|---|---|
| Qwen3.5-9B SFT | ~8h | LoRA r=64, lr 1e-5, per-device bs 4 |
| GRPO outcome_only_v2 | ~15h | watch VRAM, may drop bs to 2 |
| GRPO VAPV | ~15h | same |

Success = process > outcome pattern reproduces at 9B. Otherwise → contingency C.

### 4.4 API baselines (only after Part 3 gate passes)

GPT-4o / Gemini 2.5 Pro / Claude 4 Sonnet via `eval_api.py` if present. **Do not spend API credit before the core contribution is verified.**

### 4.5 Novelty framing (`docs/novelty_framing.md`)

Three mandatory subsections:

1. **Chart-RVR vs ChartVR** diff table (8 dimensions: reward type, verification target, method, surrogate tasks, granularity, GT dependency, application time, scale).
2. **R1-VL StepGRPO vs ChartVR** diff table (domain, step reward, source, verifiability).
3. **Introduction narrative**: BigCharts-R1 / Chart-R1 / Chart-RVR gap statement, 56% wrong-entity + 31% partial-text observation, VAPV proposal paragraph.

Then global replace `"process reward"` → `"VAPV"` / `"value-anchored process verification"` / `"factual grounding reward"` across `docs/`. Historical entries may remain.

---

## Part 5 — Results + Paper Scaffold (Day 11-12)

Main Results Table (Table 1), Ablation Table (Table 2), and paper outline per v3 guide §5.1/§5.2.

Paper structure: Introduction → Related Work (differentiation from Chart-RVR / R1-VL / BigCharts-R1 / Chart-R1) → Method (VAPV formulation) → Experiments → Analysis (error-type improvement) → Conclusion.

---

## Prohibited Actions (updates)

1. No reasoning-less samples in SFT data.
2. No > 30K scale (range 28K-30K).
3. No `chartqa_train` > 35%.
4. No training start before CHECK 1-12 PASS.
5. No `conditional_v2` redesign — only component-masking variants allowed.
6. **No reasoning < 150 chars samples.**
7. **No API baseline spend before Part 3 condition 1 passes.**
8. **No 9B skip.**
9. **No generic `process reward` terminology in paper/docs.**
10. No hardcoded `loss_type` in `train_grpo_dapo.py` — `--algo` flag required.
11. No autonomous Findings reframing — user pre-approval required if 9B pattern fails.

---

## Contingency Protocols

| Scenario | Trigger | Response |
|---|---|---|
| A | SFT AVG < 54 | 25K shrink + Block 5 removal → retrain. Still fails → 397B Block 1 regen ($25-40), user approval required. |
| B | Process ≤ Outcome (Part 3 cond 1 fails) | Step 3.4 diagnostic → 1 reward/data fix + rerun → escalate. |
| C | 9B pattern does not reproduce | Reframe as "VAPV efficacy on small (4B) chart VLMs" → retarget ACL/EMNLP **Findings**. **User pre-approval required.** |
| D | GPU outage / schedule slip | Priority order: Row 3+4 > 9B > ablation > API. Drop component ablation + API before dropping 9B. |

---

## Timeline

```
Day 1   SFT data rebuild (3h) → SFT train start (5h)
Day 2   SFT eval (1h) → Part 2 gate decision
Day 3   Row 3 DAPO+outcome (10h)
Day 4   Row 4 DAPO+VAPV (10h)
Day 5   Row 5 GRPO+VAPV (10h) → 3-row eval (2h) → Part 3 gate ⭐
Day 6   ABL-7 collapsed+VAPV (10h)
Day 7   Component ablation value_only (10h)
Day 8   Component ablation arith_only (10h) + 9B SFT (8h parallel)
Day 9   9B outcome GRPO (15h)
Day 10  9B VAPV GRPO (15h)
Day 11  API baselines (3h) + novelty framing (3h)
Day 12  Results tables + paper scaffold + buffer
```

Critical path: Day 1-2 SFT → Day 3-5 GRPO → Day 5 gate. Day 6-8 ablation and Day 8-10 9B parallelize if GPU pools allow.
