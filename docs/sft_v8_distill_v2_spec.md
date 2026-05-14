# SFT v8 Distill v2 Spec

**Goal**: rebuild the v8 SFT corpus by re-distilling all 28,594 source records through Qwen3.6-27B VLM teacher with **per-source bench-format prompts**, fixing the format misalignment that crashed `sft_v8_standard` (CMU −12.6pp).

**Why distill instead of convert**: v1 conversion preserved 397B teacher prose 200-450 chars, which violated ChartMuseum publisher template. Re-distilling under the right per-bench prompt yields teacher CoT *and* answer that match the eval format from the start.

## Family Map

Source × answer_type matrix (counts from `data/sft_v8_improved_v2.jsonl`):

| Source | answer_type | n | Family | User-prompt template | Assistant target shape | Eval bench |
|---|---|---|---|---|---|---|
| `chartqa_train` | (any) | 9,500 | ChartQA | `{Q}\nAnswer the question with a single word.` | `<think>…</think>\n{bare}` | ChartQA · CQA-aug · CQA-pro |
| `scientific(_ext)` | `numeric` ∪ digits-only | 1,853 | CharXiv cat 4 | `{Q}\n{num_inst}` (decimal-place hint when gold has fractional) | `<think>…</think>\n{bare}` | CharXiv-reasoning |
| `scientific(_ext)` | `text` ∪ `yesno` ∪ rest | 4,692 | CharXiv cat 2 | `{Q}\n* options/instructions/short phrase` | `<think>…</think>\n{bare}` | CharXiv-reasoning |
| `plotly_complex` | `multichoice` | 1,107 | ChartMuseum + value add-on | `QA_PROMPT[Q ← Q\n"Answer with the value or name of the chosen option."]` | `<think>…</think>\n<answer>{value}</answer>` | ChartMuseum |
| `plotly_complex` | `numeric` ∪ `text` ∪ `comparison` | 1,675 | ChartMuseum | `QA_PROMPT.replace([QUESTION], Q)` | `<think>…</think>\n<answer>{bare}</answer>` | ChartMuseum |
| `owid` · `kaggle_like` · `synthetic` · `additional` · `worldbank` | (any) | 9,767 | ChartMuseum | `QA_PROMPT.replace([QUESTION], Q)` | `<think>…</think>\n<answer>{bare}</answer>` | ChartMuseum |

Templates pulled verbatim from:
- `third_party/lmms-eval/lmms_eval/tasks/chartqa/chartqa.yaml` (default `post_prompt`)
- `third_party/CharXiv/src/constants.py:REASONING_RESP_INST`
- `third_party/ChartMuseum/prompt.py:QA_PROMPT`

### Multichoice add-on rule

`answer_type == "multichoice"` exists **only in `plotly_complex` (1,107 rows)**.
ChartQA / CharXiv / other ChartMuseum sources never carry it (verified
2026-05-09 over `data/sft_v8_improved_v2.jsonl`).

The 1,107 source rows have mixed gold form (751 letter / 356 value), but
**all 5 eval benches (ChartQA / CQA-aug / CQA-pro / CharXiv / ChartMuseum)
use value-form gold for multichoice questions** (CharXiv 3/3 mc-pattern
val rows are value golds; ChartMuseum essentially has none). Letter form
is therefore an annotation artifact — not a target the student should
learn to emit.

The add-on says "Answer with the value or name of the chosen option." —
which steers the teacher toward value-form regardless of how the source
row's gold is annotated. `_teacher_match` against letter golds may show
False, but the assistant_text the student trains on still carries the
value-form (matching what the eval scorers expect).

CharXiv cat 2 already states "If there are options in the question, your
final answer must conform to one of the options," so no extra hint is
needed there even if a future row carries multichoice.

## CharXiv inst_category Derivation

We do not have native `inst_category` on synthetic CharXiv-like data. Rule:

```
if answer_type == "numeric"     → 4 (number-in-general, with decimal-place hint)
elif answer is digits-only      → 4
elif answer_type in {"text","yesno","multichoice","comparison"} → 2 (text-in-general)
else                            → 2 (default)
```

### Why cat 1/3 are not used

CharXiv's `REASONING_RESP_INST[1]` requires the answer to be **grounded to
text explicitly written in the chart** (e.g., a legend label like
"Group A", a subplot title like "Density Plot"). `[3]` is the numeric
analog (e.g., a value printed on top of a bar, an axis tick label). The
official CharXiv corpus annotates this in-chart grounding, but our
synthetic `scientific(_ext)` rows are CSV-derived and have no metadata
indicating which strings/numbers are actually rendered onto the chart.
Using cat 1/3 without that guarantee would mis-instruct the teacher and
cause spurious grounding-failure mismatches. We therefore restrict to
cat 2/4, which target reasoning-derived (out-of-chart) answers.

## Teacher Sampling

`chartvr/config.py:SAMPLING_PARAMS["27b"]["thinking"]`:
- temperature 0.6, top_p 0.95, top_k 20, min_p 0.0
- presence_penalty 0.0, repetition_penalty 1.0
- `chat_template_kwargs.enable_thinking = True`

Server flags (`scripts/launch_27b_vllm.sh`):
- `--reasoning-parser qwen3 --generation-config vllm --enforce-eager`
- `--max-model-len 131072` (long CoT room)
- TP=4, 3 hosts on GPU 4-7 / 8-11 / 12-15

## Distill Output Schema (`data/sft_v8_distill_v2.jsonl`)

Each row preserves all v8 source fields and adds:

| Field | Definition |
|---|---|
| `user_text` | Final user-turn content (image + text per family template) |
| `assistant_text` | `<think>{teacher_reasoning}</think>\n{family_answer}` |
| `_format` | `distill_v2_chartqa` / `distill_v2_charxiv` / `distill_v2_chartmuseum` |
| `_charxiv_inst_cat` | 2 or 4 (only for CharXiv rows) |
| `_teacher_pred` | Raw teacher answer string (post-`</think>`, pre-format) |
| `_teacher_match` | bool — relaxed match against `answer` gold |
| `_teacher_finish_reason` | `stop` / `length` |
| `_teacher_reasoning_len` | char count of `reasoning` (or `reasoning_content`) |
| `_teacher_completion_tok` | usage.completion_tokens |

## Drop Policy (raw-capture mode)

The distill step (`distill_sft_v2.py`) only drops on **hard failures** and
writes those rows to `data/sft_v8_distill_v2_failed.jsonl`:
1. `image_missing` (image path not on disk)
2. HTTP error (timeout, connection refused, 5xx) — eligible for retry
3. Empty teacher output (`reasoning` and `content` both empty)

`finish_reason == "length"` and `_teacher_match == False` are **kept** —
neither relates directly to the eval-bench scorers' criteria, and
discarding rows on those signals would punish:
- ambiguous golds (e.g., letter vs value form)
- teacher rewordings that pass LLM judge in CharXiv/ChartMuseum
- complete CoTs that simply happen to be long

`max_tokens` is **NOT passed** to the chat completion call — the server
uses `max_model_len - prompt_tokens` automatically (27B = 131072,
397B = 262144). This eliminated the 16.7% length-cap rate observed at the
former 8192 cap during preflight.

Relaxed match (computed for diagnostic `_teacher_match` only — never used
as a drop criterion):
- Multichoice with single A-E gold: letter-equality + `\b[A-E]\b` extraction
- Numeric (gold parses as float): `abs(p - g) / max(1, abs(g))` ≤ 0.05
- Yes/No: normalize {y,yes,true,1} ↔ {n,no,false,0}
- Text: case-insensitive, strip whitespace + punctuation, contains-or-equals

## SFT Build Step (post-processing)

`scripts/build_distill_v2_sft.py` reads the raw distill JSONL, applies
per-family cleanup, and writes SFT-ready rows with `assistant_text` to
`data/sft_v8_distill_v2_sft.jsonl`. Format-compliance is the **only**
filter — rows that fail it go to `data/sft_v8_distill_v2_sft_dropped.jsonl`.

Compliance criterion derived from each bench's scorer (eval-bench-targeted):

| family | scorer mechanism | compliance check |
|---|---|---|
| chartqa | `relaxed_correctness` on last non-empty line | content has ≥ 1 non-empty line |
| charxiv | last non-empty line + LLM judge | content has ≥ 1 non-empty line |
| chartmuseum | `<answer>(.*?)</answer>` regex + LLM judge | content has ≥ 1 closed `<answer>…</answer>` pair with non-empty body |

Cleanup rules applied to compliant rows:
- chartmuseum: `assistant_text = <think>{rsn}</think>\n<answer>{LAST_<answer>_BODY}</answer>` — keeps only the last (most decisive) `<answer>` block, drops any trailing prose
- chartqa / charxiv: `assistant_text = <think>{rsn}</think>\n{LAST_LINE}` — strips trailing punctuation only

Diagnostic fields (`_teacher_match`, `_teacher_finish_reason`,
`_teacher_pred`, `_teacher_reasoning_len`, `_teacher_completion_tok`)
are preserved on the cleaned rows for downstream analysis but are **not**
filter inputs.

## Train-time Use

`train_sft.py` patch:
- If `item["user_text"]` present → use it verbatim as user-turn content (do not append `CHARTQA_POST`)
- Else fallback to existing `q + CHARTQA_POST` behavior

## Preflight (P3, ≥50/family = 150 minimum)

Audit script (`scripts/audit_distill_v2.py`):
1. **Format**: `user_text` byte-equal publisher template (mod question + image content) ✓
2. **Reasoning preserved**: `<think>` opens, `</think>` closes, body length ≥ 50 chars ✓
3. **Family target**: ChartQA/CharXiv = no `<answer>` tag, ChartMuseum = trailing `<answer>X</answer>` ✓
4. **Round-trip**: applying `score_standard.py:extract_post_think` to `assistant_text` recovers the gold under target bench scorer ✓
5. **Match rate**: `_teacher_match=True` ratio per family ≥ 60% (else escalate)
6. **Throughput**: aggregate tok/s × 28,594 → ETA estimate

## Resume Semantics

- Read existing JSONL → build set of completed `sample_id` (= `f"{source}_{i}"`)
- Skip already-completed rows in input loop
- Append new rows incrementally; one fsync per row not required (OS-buffered append safe under crash recovery)

## Bulk Run

After preflight PASS:
```
PYTHONPATH=. python scripts/distill_sft_v2.py \
  --input data/sft_v8_improved_v2.jsonl \
  --output data/sft_v8_distill_v2.jsonl \
  --failed data/sft_v8_distill_v2_failed.jsonl \
  --hosts http://localhost:9101/v1,http://localhost:9102/v1,http://localhost:9103/v1 \
  --max_concurrent_per_host 6 \
  --max_tokens 8192
```

Bulk ETA = (28,594 × avg_completion_tokens) / aggregate_tok_s. Computed from preflight measurements.
