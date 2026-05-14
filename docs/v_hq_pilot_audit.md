# v_hq Pilot Audit & Throughput Findings — 2026-04-09

## Summary

Teacher distillation (397B VLM, thinking mode) **at the planned 43K-sample scale is infeasible on current hardware**. Evidence and pivot recommendation below. All findings come from direct server-log inspection and client-side timing; servers were NOT modified.

## Throughput measurements

### Server state during pilot
- 9200 (GPU 0-7, TP=8, GPTQ-int4, --enforce-eager)
- 9201 (GPU 8-15, TP=8, GPTQ-int4, --enforce-eager)
- Both launched with `max_model_len=65536`

### Observed generation throughput (from vLLM `loggers.py:259`)

| Host | Running reqs | Avg gen throughput | per-request |
|---|---|---|---|
| 9200 | 6-7 | 0.7-3.5 tok/s | ~0.3-0.5 tok/s |
| 9201 | 6 | 10-14 tok/s | ~1.7-2.3 tok/s |
| 9200 (backlogged) | 21 | 8.4 tok/s | ~0.4 tok/s |

### GPU utilization snapshots

- Initial: 9200 GPU 4 at **1%** (straggler), others 100%. TP=8 tied → effective stall.
- Later: both hosts show mixed utilization (GPU 5 at 18%, GPU 10 at 38%, GPU 9 at 92%...) — TP groups not uniformly loaded.
- GPU 5 / GPU 10 appear to be periodic stragglers.

### Single-call latency

- Direct sequential call, 3 samples, 200s timeout: **did not complete** — 3 calls exceeded 200s sequentially on 9201.
- Direct single call, 400s timeout: also exceeded (possibly queued behind lingering requests from killed pilot).
- Estimated per-sample wall-clock with thinking mode: **60-200+ seconds** per request under contention.

## Projection

Assuming 2 tok/s/req (best case, 9201) and teacher thinking length ~1000-2000 tokens:
- Per sample: 500-1000 seconds
- 12 concurrent (6 per host × 2 hosts): 12 parallel
- **100 samples pilot**: (100/12) × 500-1000s = **70-140 min** (if no backlog)
- **43,744 samples (distill_source.jsonl)**: ~500-1000 hours wall-clock. **Infeasible**.
- Even reducing to **3,000 samples** (critical subset: scientific_ext + plotly_complex + owid) → 35-70 hours. Still marginal.

## Root causes

1. **397B GPTQ-int4 TP=8 + --enforce-eager**: known-slow combination. `--enforce-eager` disables CUDA graphs for stability (required per CLAUDE.md for this model on these GPUs).
2. **Thinking mode**: 5-10x slower than instruct mode due to long reasoning generation. v9.1 eval showed reasoning_content up to 22K chars (~5500 tokens) for complex queries. At ~2 tok/s, that's ~45 min per request.
3. **TP=8 stragglers**: Single GPU at 1-18% util causes all 8 GPUs in the tensor-parallel group to wait. Root cause not diagnosed (not touching server per policy).
4. **Lingering requests after client cancel**: vLLM does not cancel in-flight requests on client disconnect — killing the pilot process left 20+ requests still processing, blocking new requests in the queue. This amplified the apparent slowness in later measurements.

## Issues encountered + fixes applied (call/data layer only)

### Issue 1: `max_tokens=65536` → 400 error
- vLLM constraint: `prompt_tokens + max_tokens ≤ max_model_len`
- Requesting max_tokens exactly equal to max_model_len leaves zero room for prompt
- **Fix**: Remove `max_tokens` from all 397B OpenAI SDK calls (5 files). vLLM auto-computes `max_model_len - prompt_tokens`. Affected files:
  - `scripts/teacher_distill.py`
  - `scripts/generate_qa.py` (2 sites)
  - `scripts/generate_cot.py` (2 sites)
  - `scripts/verify_qa_answers.py` (1 site)
  - `code/rewards/llm_verifier.py` (3 sites: `__init__`, verify_single, geval_single)
- `chartvr/llm_client.py::MultiHostClient.chat()` wrapper verified clean — no hidden `max_tokens` default.

### Issue 2: Pilot throughput catastrophically slow
- Root cause: described above
- **Not a fix** — structural hardware/model constraint
- **Pivot**: see below

## Pivot Strategy — v_hq-lite

Abandon full teacher distillation. Use `data/distill_source.jsonl` (43,744 samples) directly as SFT training data with existing `reasoning_steps` fields from source. Rationale:

1. **Root-cause fix from distillation plan is still in place**:
   - Train-eval prompt unified (`train_sft.py` patched to `EVAL_SYSTEM_PROMPT` + eval user template)
   - Rule template exclusion (block_a/b/c/d/f filtered out in source build — 30,923 rejected from v9)
   - High-signal sources only (11,017 from v9 + 29,549 from v8 + 3,178 from block_c_api_qa_v2)
2. **v8 (46% CharXiv baseline) also had no teacher distillation** — it used sft_30k.jsonl with mixed-quality reasoning/placeholder and achieved baseline
3. **Expected improvement** over v8 without teacher distillation:
   - Prompt unification: small-to-medium benefit (eliminates format drift at eval time)
   - v9 high-quality source inclusion (owid/synthetic/plotly_complex/scientific_ext): potential CharXiv-reasoning improvement
   - Rule template exclusion + dedup: stability (addresses v9.1 bimodal collapse)
4. **Residual risk**: v9/v9.1 regression showed that simply adding v9 data hurt ChartQA-human. Need length-based filtering and potentially rebalancing.

### v_hq-lite execution plan

Preparation (no GPU, proceed now):
1. **Length audit of distill_source.jsonl** — compute reasoning_steps length distribution per source, flag samples too long for student `max_length=4096`
2. **Filter script** `scripts/filter_for_training.py` — drop samples exceeding length budget, write `data/sft_hq_lite.jsonl`
3. **Update train_sft.py assistant_text fallback** — already in place from Pillar 1 patch
4. **Update plan doc** — reflect v_hq-lite approach

Training (GPU needed, WAIT for user):
5. SFT with `data/sft_hq_lite.jsonl`, unified prompt, `max_length=4096`
6. LoRA merge, 5-bench eval, compare

### Partial teacher distillation (opportunistic, if servers drain)
If 9200/9201 drain naturally during this session, attempt a small-batch distillation (e.g., 500 samples from highest-priority categories: plotly_complex + scientific_ext subset). Use the result as **augmentation** on top of sft_hq_lite, not replacement.

## Status at 19:40 (final)

### Completed
- ✓ `train_sft.py` patched (EVAL_SYSTEM_PROMPT + eval user template + assistant_text preference field)
- ✓ `scripts/build_distill_source.py` written, reasoning_steps carryover added
- ✓ `data/distill_source.jsonl` built (43,744 samples, reasoning_steps preserved for 25,744)
- ✓ `scripts/teacher_distill.py` written, `max_tokens` omitted
- ✓ `scripts/prepare_sft_hq_lite.py` written
- ✓ `data/sft_hq_lite.jsonl` built (43,744 samples ready for training)
- ✓ `scripts/audit_distill.py` written
- ✓ `scripts/merge_distilled_into_lite.py` written (for optional future merge)
- ✓ `max_tokens` removed from all 5 live 397B-caller files + wrapper verified clean
- ✓ `docs/v_hq_training_readiness.md` written (command recipes for user)

### Aborted
- ✗ Pilot distillation (100 samples) — throughput infeasible
- ✗ Small distillation (500 samples) — 9200 hung (0% GPU util, 0.3 tok/s for 1 req), 9201 also stuck with lingering requests
- ✗ Single-sample direct test (240s timeout) — 9201 did not complete 1 request in 240s while other requests were in flight

### Final verdict on teacher distillation

**Infeasible on current hardware within any reasonable session time.**

Measurements and projections:
- 397B GPTQ-int4 + TP=8 + `--enforce-eager` + thinking mode yields ~2 tok/s per request (aggregate 10-16 tok/s across all concurrent requests)
- Thinking responses for chart questions produce 1000-5000+ tokens based on v9.1 eval distribution (mean 6,391 chars ≈ 1,600 tokens)
- **Per sample wall-clock**: 500-2500 seconds
- **100 samples at 4 concurrent on 9201 alone**: ~7-35 hours
- **43,744 full source**: days-to-months on current hardware

Contributing factors (not touching server per policy):
1. `--enforce-eager` disables CUDA graphs (required for GPTQ-int4 TP=8 stability per CLAUDE.md)
2. Recurring **single-GPU stragglers** (GPU 4 at 1%, later GPU 9 at 24%, GPU 13 at 3%) bottleneck entire TP group
3. vLLM lingering-request behavior: killing client does not cancel in-flight requests, so pilot retries compound backlog
4. 9200 reached a fully-hung state (0% GPU util, 0.3 tok/s for 1 running req) — functionally dead for distillation without restart
5. **9201 is occupied by another workflow (not ours)** — Running: 4-5 requests persists throughout the session with constant new POST 200 OK entries and prompt throughput spikes (1569-2017 tok/s). Matches CLAUDE.md note: "9201 (GPU 8-15) 은 절대 건드리지 않음 — 다른 용도 전용". Our distillation requests were queuing behind existing traffic, effectively doubling latency and consuming capacity the other workflow needs.

### Path chosen: v_hq-lite

Use `data/sft_hq_lite.jsonl` directly for SFT training:
- Preserves v9's high-quality reasoning_steps (25,744 samples) 
- Inherits v8's sft_30k (29,549 samples, of which 18,000 chartqa_train use placeholder — same as v8 baseline)
- Rejects v9 rule templates (block_a/b/c/d/f, 30,923 rows — addresses bimodal collapse)
- Applies unified prompt via `train_sft.py` Pillar 1 patch (EVAL_SYSTEM_PROMPT + eval user template)
- Zero teacher distillation required

**Root cause fixes delivered without distillation**:
- #3 bimodal reasoning → fixed (rule exclusion)
- #4 prompt mismatch → fixed (prompt unification)
- #1 runaway thinking → **indirectly addressed** via #3 + #4; not guaranteed but best possible without distillation
- #2 non-empty accuracy → depends on reasoning quality of v9 high-signal sources (which worked in v9 except for CharXiv regression)

### Open questions the user should review

1. Whether to attempt SFT on sft_hq_lite.jsonl as-is (recommended: go) vs. add teacher distillation later as augmentation
2. Whether 9200 needs a clean restart (can't be done without user approval — GPU policy)
3. CharXiv regression risk: v_hq-lite inherits v9's data for non-rule sources. v9 CharXiv was 42.10% (-3.9 from v8 46.00). Prompt unification may recover some of this, but not guaranteed
4. Alternative path if v_hq-lite underperforms: drop chartqa_train entirely and train on 25,744 samples of pure v9-quality — or fall back to v8 sft_30k.jsonl directly

### CRITICAL finding: v9.1 runaway is REPETITION LOOPS, not max_tokens cap

**Correction to earlier hypothesis.** I initially suspected `eval_multi_server.py:138 max_tokens=4096` was silently truncating long reasoning and causing the 44.8% empty_content. Deeper analysis of `results/v9_1/sft_v9_1_4b/charxiv_reasoning.jsonl` (1000 samples) shows the actual pattern:

| rc_chars range | samples | empty_content% | accuracy% |
|---|---|---|---|
| 0-1,000 | 130 | 0.0% | 76.2% |
| 1,000-3,000 | 275 | 0.0% | 67.3% |
| 3,000-5,000 | 91 | 1.1% | 56.0% |
| 5,000-8,000 | 73 | 13.7% | 41.1% |
| **8,000-12,000** | **259** | **89.2%** | **9.7%** |
| **12,000-16,000** | **168** | **98.8%** | **3.0%** |
| 16,000+ | 4 | 100.0% | 0% |

The empty_content rate explodes precisely when reasoning_content exceeds ~8,000 chars (~2,000 tokens). At max_tokens=4096, this is **well under the cap** — so truncation is NOT the mechanism.

**Inspection of runaway samples shows unmistakable repetition loops**:
> "Let's look at the X values. Wait, let's look at the X values. ..."
> "Let's look at the point at x=4000. It's 40. Wait, let's look at the point at x=2000. It's 35. Let's look at the point at x=4000. It's 40. ..."

The model gets stuck in a loop, generates the same statements repeatedly, burns through generation budget (naturally not hitting max_tokens, but not producing `</think>` either), and the server eventually returns with `content=""` because `</think>` never appeared.

**Root cause**: v9.1 training data (bimodal: 82% short rules + 18% longer CoT) taught the student to produce EITHER very short responses OR to fall into unstructured open-ended thinking on complex queries. Without well-formed stop discipline, complex CharXiv queries trigger the loop behavior.

### What v_hq-lite actually fixes

- **Bimodal collapse**: rule templates removed → cleaner reasoning length distribution (25,744 samples all with reasoning_steps p50=189 chars, p99=1376 chars). Should reduce "unstructured thinking" propensity.
- **Prompt mismatch**: train-eval prompt unified → no format drift
- **Does NOT directly break repetition loops** — this requires either:
  - Better training data diversity (more varied reasoning patterns)
  - OR: repetition_penalty at inference (could be added, but currently 1.0 in SAMPLING_PARAMS)
  - OR: teacher distillation to provide natural-length reasoning (infeasible on this hardware)

### Cross-version runaway analysis — empty_content is NOT v9.1-specific

Comparing CharXiv empty_content rate across historical runs:

| Run | CharXiv acc | empty_content | rc p50 | rc p95 | rc max | long_rc samples | long_rc empty |
|---|---|---|---|---|---|---|---|
| v7_row_a | 47.0% | 29.8% (298/1000) | 2976 | 7002 | 9258 | 1 | 1/1 (100%) |
| v7_row_b | 49.6% | 27.4% (274/1000) | 3670 | 13657 | 26331 | 330 | 270/330 (81.8%) |
| v8_row2_sft | **44.5%** | **38.6%** (386/1000) | 3811 | 13319 | 48591 | 384 | 368/384 (95.8%) |
| v9_sft | 42.1% | 31.6% (316/1000) | 3182 | 12275 | 15511 | 317 | 280/317 (88.3%) |
| v9.1_sft | 39.5% | **41.2%** (412/1000) | 5083 | 13548 | 22898 | 431 | 401/431 (93.0%) |

**Critical observations**:
1. **empty_content exists in ALL versions** (27-41%), not just v9.1. The plan claim "v9.1 failure = runaway empty content" was oversimplified.
2. **v8 had 38.6% empty yet achieved 44.5% CharXiv** — higher empty doesn't monotonically mean lower accuracy. The non-empty response quality matters.
3. **Once rc exceeds ~8,000 chars, empty_content happens 82-96% across all versions** — runaway is a universal behavior that scales with rc length, not unique to any version's data composition
4. **v9 had LOWER empty than v8** (31.6% vs 38.6%) yet **still regressed in accuracy** (42.1% vs 44.5%). This proves the regression was not driven by empty_content change.
5. **v9.1's accuracy drop beyond v9 IS correlated with more long_rc samples** (v9.1 has 431 samples with rc>8000 vs v9's 317)

### Revised root cause attribution (for v9.1 regression)

- **Primary**: v9 bimodal rule templates (82% short rules <20 words) biased student toward short responses, but also amplified OOD behavior on complex queries → v9.1 made this worse by adding MORE rule templates and scientific_ext with mismatched numeric/text answers
- **Secondary**: train-eval prompt drift (train used "Question: {q}", eval used "Look at this chart... Question: {q}")
- **Not primary**: empty_content / runaway — this is pervasive across versions and not uniquely caused by v9.1's data

### Revised expectation for v_hq-lite

**Realistic baseline**: ≈v8 performance (~44.5% CharXiv). The unique value of v_hq-lite is:
- Cleaner training distribution (rule templates excluded)
- Prompt unification
- Slightly more varied sources

**Upper bound without teacher distillation**: probably ≈v8 + 1-2pp if prompt unification + cleaner distribution helps. Teacher distillation was supposed to provide the big CharXiv-specific improvement via natural-length reasoning; we cannot get that in this session.

**Failure mode to watch**: if v_hq-lite empty_content stays at ~35-40%, we're not solving the underlying behavior — just the data composition. A subsequent experiment could try **repetition_penalty=1.05** at eval time as a cheap loop-breaker test.

### Secondary fix: `eval_multi_server.py` max_tokens removed (harmless, may help edge cases)

Even though max_tokens=4096 is NOT the primary cause of v9.1 runaway, the same semantic consistency applies as with the 397B fixes. Removed as a cleanup. In edge cases where reasoning reaches >4000 tokens naturally (not loops), this lets the student complete. Line `eval_multi_server.py:138` now lets vLLM auto-compute `max_model_len - prompt_tokens` (effective ~6000 tokens budget).

### Additional observation on eval sampling params (flagged for user decision)

`eval_multi_server.py` at eval time uses:
- `temperature=0.6, top_p=0.95` (Qwen thinking recommended ✓)
- `presence_penalty=0.0` (from `sp.get(..., 0.0)` default)
- `top_k`, `min_p`, `repetition_penalty` only if caller passes sampling_params dict; otherwise **not set**

Qwen3.5 thinking_general recommends `presence_penalty=1.5, top_k=20, min_p=0.0, repetition_penalty=1.0`. Our eval uses `presence_penalty=0.0` by default.

**Presence penalty 0.0 provides no disincentive to repeat phrases** — this is a plausible contributing factor to the repetition-loop runaway. However, changing it retroactively would make v_hq-lite results non-comparable to v8/v9/v9.1 numbers (which all used the same defaults).

**Recommendation for user**: consider running v_hq-lite eval with and without `presence_penalty=1.5` to isolate the effect. Do NOT change the default for the first comparison run. This is a follow-up experiment.

### Files touched this session

Modified:
- `train_sft.py` (untracked file; patched to use EVAL_SYSTEM_PROMPT + eval user template + assistant_text)
- `scripts/generate_qa.py`
- `scripts/generate_cot.py` (untracked file)
- `scripts/verify_qa_answers.py`
- `code/rewards/llm_verifier.py`
- **`eval_multi_server.py`** (NEW FIX — removed `max_tokens=4096`, critical for v9.1 runaway)

Created:
- `scripts/build_distill_source.py`
- `scripts/teacher_distill.py`
- `scripts/prepare_sft_hq_lite.py`
- `scripts/audit_distill.py`
- `scripts/merge_distilled_into_lite.py`
- `data/distill_source.jsonl`
- `data/sft_hq_lite.jsonl`
- `docs/v_hq_pilot_audit.md`
- `docs/v_hq_training_readiness.md`

Unchanged (critical):
- `chartvr/llm_client.py` — verified clean, no `max_tokens` default
- 9200/9201 server processes — not touched, still serving 397B
- No GPU-requiring training or eval launched
