"""Assemble v8_improved_v2: regression-targeted 28K-30K SFT dataset.

Follows docs/sft_v8_improved_action_guide_v3.md:
  5 revisions over v1/v2 plan:
   1. Filler min reasoning 40 -> 150 chars
   2. Block 2 block_c filler requires reasoning >= 200; shrink 8K->7K on shortfall
   3. Scale flexible 28K-30K (quality > fixed 30K)
   4. Self-gen source from hq_lite_clean reasoning_steps pool
   5. CHECK 1-12 regression coverage verification

Pools:
  teacher   = sft_hq_teacher.jsonl (25382 AT) + sft_v8_improved_teacher_distill.jsonl (4156 AT chartqa)
  selfgen   = sft_hq_lite_clean.jsonl (rows with reasoning_steps, >=300 chars)
  v9_filler = sft_v9.jsonl blocks (a/b/c/d/f + sci_ext/plotly/etc.)
  v8_rest   = sft_30k.jsonl (non-chartqa rows, kaggle_like / additional / etc.)
  gt_pool   = charts_v2/chartvr_train_final.jsonl (Block 4 backfill)

Usage:
  python scripts/assemble_v8_improved_v2.py --dry_run --stats
  python scripts/assemble_v8_improved_v2.py --output data/sft_v8_improved_v2.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

random.seed(42)

TEMPLATE_PREFIX_RE = re.compile(
    r"^\s*(let me analyze|i'?ll analyze|looking at|let's look at|to answer)",
    re.IGNORECASE,
)

# ------------------------------------------------------------------
# Field helpers
# ------------------------------------------------------------------
def get_reasoning_text(s: dict) -> str:
    """Return best-effort reasoning string (assistant_text preferred)."""
    at = s.get("assistant_text")
    if at:
        return at
    r = s.get("reasoning_steps") or s.get("reasoning") or ""
    if isinstance(r, list):
        r = "\n".join(str(x) for x in r)
    return str(r)


def reasoning_length(s: dict) -> int:
    return len(get_reasoning_text(s))


def has_think_tag(s: dict) -> bool:
    return "<think>" in (s.get("assistant_text") or "")


def has_reasoning(s: dict) -> bool:
    """Non-trivial reasoning present (>= 50 chars, not N/A)."""
    text = get_reasoning_text(s).strip()
    return bool(text) and text not in ("N/A", "None") and len(text) >= 50


# ------------------------------------------------------------------
# Quality gate (Revision 1: min 150 chars)
# ------------------------------------------------------------------
def passes_quality_gate(s: dict) -> bool:
    """Strict gate: reasoning >= 150 chars, not template-shaped if short,
    and must have <think> tag or reasoning_steps field."""
    text = get_reasoning_text(s).strip()
    if not text or text in ("N/A", "None"):
        return False
    n = len(text)
    if n < 150:
        return False
    # Template-prefix samples are only allowed if long enough to carry content
    stripped = text.lstrip("<>think\n ")
    if TEMPLATE_PREFIX_RE.match(stripped) and n < 300:
        return False
    # Must have an actual reasoning field
    if not has_think_tag(s) and not (s.get("reasoning_steps") or s.get("reasoning")):
        return False
    return True


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------
def classify_answer(ans) -> str:
    a = str(ans).strip()
    if a.lower() in ("yes", "no"):
        return "yesno"
    try:
        float(a.replace(",", "").replace("%", "").replace("$", ""))
        return "numeric"
    except ValueError:
        return "text"


def is_text(ans) -> bool:
    return classify_answer(ans) == "text"


def is_yesno(ans) -> bool:
    return classify_answer(ans) == "yesno"


def classify_question(q: str) -> str:
    q = (q or "").lower().strip()
    if any(w in q for w in ["difference", "average", "mean", "ratio", "sum of",
                            "product", "median", "percent", "percentage of",
                            "total", "overall", "combined"]):
        return "multi_step"
    if any(w in q for w in ["more than", "less than", "higher", "lower", "greater",
                            "compared", "between", "most", "least", "smallest", "largest",
                            "maximum", "minimum", "longest", "shortest", "highest", "lowest"]):
        return "comparison"
    if q.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ", "has ", "have ")):
        return "yesno"
    if q.startswith(("which ", "who ", "what color")):
        return "which_entity"
    if any(w in q for w in ["how many", "how much"]):
        return "counting"
    if any(w in q for w in ["what is the value", "what was the value"]):
        return "simple_value"
    return "other"


def is_which(q: str) -> bool:
    q = (q or "").lower().strip()
    return q.startswith(("which ", "who ")) or "which of" in q


def is_percentage(q: str) -> bool:
    q = (q or "").lower()
    return "percent" in q or "percentage" in q or "%" in q


def is_average(q: str) -> bool:
    q = (q or "").lower()
    return "average" in q or "mean" in q or "median" in q


# ------------------------------------------------------------------
# Dedup key (normalize absolute path)
# ------------------------------------------------------------------
def dedup_key(s: dict) -> str:
    img = s.get("image_path", "") or ""
    img = img.replace("/ex_disk2/mhpark/poc/chartvr/", "")
    q = (s.get("question") or "")[:80]
    return f"{img}|{q}"


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        print(f"  WARN: {path} missing", file=sys.stderr)
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_teacher_map(pool_files: list) -> dict:
    """Merge teacher pools keyed by dedup_key; assistant_text preserved."""
    m = {}
    for p in pool_files:
        for s in load_jsonl(p):
            k = dedup_key(s)
            if k not in m:
                m[k] = s
    return m


def build_selfgen_map(hq_lite: list, min_chars: int = 300) -> dict:
    """Self-gen reasoning_steps >= min_chars, keyed by dedup_key."""
    m = {}
    for s in hq_lite:
        if reasoning_length(s) < min_chars:
            continue
        if not passes_quality_gate(s):
            continue
        k = dedup_key(s)
        if k not in m:
            m[k] = s
    return m


# ------------------------------------------------------------------
# Block 1: ChartQA teacher 10K
# ------------------------------------------------------------------
def select_block1_chartqa(teacher_map: dict, target: int = 10_000) -> list:
    pool = [s for s in teacher_map.values()
            if s.get("source") == "chartqa_train" and passes_quality_gate(s)]
    print(f"  [B1] chartqa teacher pool (QG pass): {len(pool)}")

    # Phase 1: text+which gold
    gold = [s for s in pool if is_text(s["answer"]) and is_which(s["question"])]
    text_only = [s for s in pool if is_text(s["answer"]) and not is_which(s["question"])]
    which_only = [s for s in pool if is_which(s["question"]) and not is_text(s["answer"])]
    rest = [s for s in pool
            if not is_text(s["answer"]) and not is_which(s["question"])]

    selected = list(gold) + list(text_only) + list(which_only)
    print(f"  [B1] gold(text+which)={len(gold)}  text_only={len(text_only)}  which_only={len(which_only)}")

    # Phase 4: fill remainder with question-type quota, reasoning-length sort
    remainder = target - len(selected)
    if remainder > 0:
        buckets = defaultdict(list)
        for s in rest:
            buckets[classify_question(s["question"])].append(s)
        for k in buckets:
            buckets[k].sort(key=lambda s: reasoning_length(s), reverse=True)

        quota = {
            "multi_step": int(remainder * 0.40),
            "comparison": int(remainder * 0.20),
            "counting": int(remainder * 0.14),
            "yesno": int(remainder * 0.07),
            "simple_value": int(remainder * 0.04),
            "other": int(remainder * 0.15),
        }
        # ensure quota sums <= remainder
        assigned = 0
        fill = []
        for qt, q in quota.items():
            take = min(q, len(buckets.get(qt, [])))
            fill.extend(buckets[qt][:take])
            assigned += take
        # backfill any deficit from any remaining rest, longest-reasoning first
        if assigned < remainder:
            used = {dedup_key(s) for s in fill}
            leftover = [s for s in rest if dedup_key(s) not in used]
            leftover.sort(key=lambda s: reasoning_length(s), reverse=True)
            fill.extend(leftover[: remainder - assigned])
        selected.extend(fill)

    # Truncate if overshoot
    selected = selected[:target]
    return selected


# ------------------------------------------------------------------
# Block 2: Scientific (Revision 2: filler >= 200)
# ------------------------------------------------------------------
def select_block2_scientific(
    teacher_map: dict,
    selfgen_map: dict,
    v9_pool: list,
    target: int = 8000,
) -> tuple:
    """Returns (selected, shortfall_for_block5)."""
    selected = []
    used = set()

    def add(s):
        k = dedup_key(s)
        if k in used:
            return
        used.add(k)
        selected.append(s)

    # Phase 1-2: teacher scientific_ext + scientific
    for src in ("scientific_ext", "scientific"):
        for s in teacher_map.values():
            if s.get("source") == src and passes_quality_gate(s):
                add(s)
    tph = len(selected)
    print(f"  [B2] teacher sci_ext+scientific: {tph}")

    # Phase 3-4: self-gen >=300 for same sources (text-answer priority)
    sg_candidates = [
        s for s in selfgen_map.values()
        if s.get("source") in ("scientific_ext", "scientific") and dedup_key(s) not in used
    ]
    sg_candidates.sort(
        key=lambda s: (is_text(s["answer"]), is_which(s["question"]), reasoning_length(s)),
        reverse=True,
    )
    for s in sg_candidates:
        add(s)
    sgh = len(selected) - tph
    print(f"  [B2] self-gen sci >=300: {sgh}  accum={len(selected)}")

    # Phase 5: filler from v9 block_c (reasoning >= 200) — Revision 2
    remaining = target - len(selected)
    shortfall = 0
    if remaining > 0:
        filler = [
            s for s in v9_pool
            if s.get("source") == "block_c"
            and dedup_key(s) not in used
            and reasoning_length(s) >= 200
            and passes_quality_gate(s)
        ]
        # prioritize text/which, then reasoning length
        filler.sort(
            key=lambda s: (is_text(s["answer"]), is_which(s["question"]), reasoning_length(s)),
            reverse=True,
        )
        available = len(filler)
        print(f"  [B2] block_c filler (reasoning>=200, QG): {available}  need={remaining}")
        if available >= remaining:
            for s in filler[:remaining]:
                add(s)
        else:
            for s in filler:
                add(s)
            shortfall = remaining - available
            print(f"  [B2] ⚠️ shortfall {shortfall} → shift to Block 5")

    return selected, shortfall


# ------------------------------------------------------------------
# Block 3: Complex / Diverse (plotly + kaggle + additional)
# ------------------------------------------------------------------
def select_block3_complex(
    teacher_map: dict,
    selfgen_map: dict,
    v9_pool: list,
    v8_rest: list,
    target: int = 6000,
) -> list:
    selected = []
    used = set()

    def add(s):
        k = dedup_key(s)
        if k in used:
            return
        used.add(k)
        selected.append(s)

    # Phase 1-3: teacher plotly_complex + kaggle_like + additional
    for src in ("plotly_complex", "kaggle_like", "additional"):
        for s in teacher_map.values():
            if s.get("source") == src and passes_quality_gate(s):
                add(s)
    tph = len(selected)
    print(f"  [B3] teacher plotly+kaggle+additional: {tph}")

    # Phase 4: self-gen plotly_complex >=300, text-priority
    sg = [
        s for s in selfgen_map.values()
        if s.get("source") == "plotly_complex" and dedup_key(s) not in used
    ]
    sg.sort(key=lambda s: (is_text(s["answer"]), reasoning_length(s)), reverse=True)
    for s in sg:
        add(s)
    sgh = len(selected) - tph
    print(f"  [B3] self-gen plotly >=300: {sgh}  accum={len(selected)}")

    # Phase 5 filler: block_d / block_f from v9 (text-rich) — needs QG pass
    remaining = target - len(selected)
    if remaining > 0:
        filler = [
            s for s in v9_pool
            if s.get("source") in ("block_d", "block_f")
            and dedup_key(s) not in used
            and reasoning_length(s) >= 150
            and passes_quality_gate(s)
        ]
        filler.sort(
            key=lambda s: (is_text(s["answer"]), is_which(s["question"]), reasoning_length(s)),
            reverse=True,
        )
        print(f"  [B3] v9 block_d/f filler pool: {len(filler)}  need={remaining}")
        for s in filler:
            if len(selected) >= target:
                break
            add(s)

    # Final backfill from v8_rest kaggle_like/additional with reasoning (if still short)
    if len(selected) < target:
        extra = [
            s for s in v8_rest
            if s.get("source") in ("kaggle_like", "additional", "plotly_complex")
            and dedup_key(s) not in used
            and passes_quality_gate(s)
        ]
        extra.sort(key=lambda s: reasoning_length(s), reverse=True)
        for s in extra:
            if len(selected) >= target:
                break
            add(s)

    return selected[:target]


# ------------------------------------------------------------------
# Block 4: OWID / Synth / WB (teacher + chartvr_train_final)
# ------------------------------------------------------------------
def select_block4_owid(
    teacher_map: dict,
    gt_pool: list,
    used_keys: set,
    target: int = 4000,
) -> list:
    quota = {"owid": 2000, "synthetic": 1500, "worldbank": 500}
    selected = []
    used = set(used_keys)

    def add(s):
        k = dedup_key(s)
        if k in used:
            return False
        used.add(k)
        selected.append(s)
        return True

    for src, q in quota.items():
        pool = [s for s in teacher_map.values()
                if s.get("source") == src and passes_quality_gate(s)
                and dedup_key(s) not in used]
        pool.sort(
            key=lambda s: (is_text(s["answer"]), is_which(s["question"]), reasoning_length(s)),
            reverse=True,
        )
        for s in pool[:q]:
            add(s)

    # Backfill from chartvr_train_final (already-verified GT)
    if len(selected) < target:
        gt = [s for s in gt_pool
              if dedup_key(s) not in used and passes_quality_gate(s)]
        gt.sort(key=lambda s: reasoning_length(s), reverse=True)
        for s in gt:
            if len(selected) >= target:
                break
            add(s)

    return selected[:target]


# ------------------------------------------------------------------
# Block 5: Balance / regression targeting (2K + shortfall)
# ------------------------------------------------------------------
def select_block5_balance(
    blocks_1_4: list,
    teacher_map: dict,
    selfgen_map: dict,
    v9_pool: list,
    hq_lite_all: list,
    v8_rest: list,
    target: int = 2000,
) -> list:
    used = {dedup_key(s) for s in blocks_1_4}
    total = len(blocks_1_4)

    text_have = sum(1 for s in blocks_1_4 if is_text(s["answer"]))
    which_have = sum(1 for s in blocks_1_4 if is_which(s["question"]))
    yn_have = sum(1 for s in blocks_1_4 if is_yesno(s["answer"]))
    pct_have = sum(1 for s in blocks_1_4 if is_percentage(s["question"]))
    avg_have = sum(1 for s in blocks_1_4 if is_average(s["question"]))

    final_n = total + target
    text_gap = max(0, int(final_n * 0.30) - text_have)
    which_gap = max(0, int(final_n * 0.15) - which_have)
    yn_gap = max(0, int(final_n * 0.05) - yn_have)
    pct_gap = max(0, int(final_n * 0.08) - pct_have)
    avg_gap = max(0, int(final_n * 0.05) - avg_have)

    print(f"  [B5] gaps: text={text_gap} which={which_gap} yn={yn_gap} pct={pct_gap} avg={avg_gap}")

    # Candidate pool: teacher + selfgen + v9 block_a/b (passing QG), excluding used.
    # HARD EXCLUDE chartqa_train here — Block 1 already caps chartqa at 10K and
    # the CHECK 7 ceiling (35%) forbids inflating chartqa share in balance pass.
    cand = []
    for s in teacher_map.values():
        if s.get("source") == "chartqa_train":
            continue
        if dedup_key(s) not in used and passes_quality_gate(s):
            cand.append(s)
    for s in selfgen_map.values():
        if s.get("source") == "chartqa_train":
            continue
        if dedup_key(s) not in used and passes_quality_gate(s):
            cand.append(s)
    for s in v9_pool:
        if s.get("source") in ("block_a", "block_b") and dedup_key(s) not in used \
                and reasoning_length(s) >= 150 and passes_quality_gate(s):
            cand.append(s)
    # Backup pool: hq_lite rows with 150<=reasoning<300 (below selfgen_map gate)
    for s in hq_lite_all:
        if s.get("source") == "chartqa_train":
            continue
        if dedup_key(s) in used:
            continue
        if 150 <= reasoning_length(s) < 300 and passes_quality_gate(s):
            cand.append(s)
    # Backup pool: v8_rest non-chartqa with reasoning passing QG
    for s in v8_rest:
        if s.get("source") == "chartqa_train":
            continue
        if dedup_key(s) in used:
            continue
        if passes_quality_gate(s):
            cand.append(s)
    # dedupe cand
    seen = set()
    uniq = []
    for s in cand:
        k = dedup_key(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    cand = uniq

    # Bucket-based filling to guarantee text_gap is met.
    text_bucket = [s for s in cand if is_text(s["answer"])]
    yn_bucket = [s for s in cand if is_yesno(s["answer"]) and not is_text(s["answer"])]
    other_bucket = [s for s in cand
                    if not is_text(s["answer"]) and not is_yesno(s["answer"])]

    def score_text(s):
        k = 0
        if is_which(s["question"]):
            k += 50
        if is_percentage(s["question"]):
            k += 20
        if is_average(s["question"]):
            k += 15
        return (k, reasoning_length(s))

    text_bucket.sort(key=score_text, reverse=True)
    yn_bucket.sort(key=lambda s: reasoning_length(s), reverse=True)
    other_bucket.sort(key=lambda s: (is_which(s["question"]),
                                     is_percentage(s["question"]),
                                     is_average(s["question"]),
                                     reasoning_length(s)), reverse=True)

    print(f"  [B5] cand pool: text={len(text_bucket)} yn={len(yn_bucket)} other={len(other_bucket)}")

    selected = []
    text_take = min(text_gap, len(text_bucket), target)
    selected.extend(text_bucket[:text_take])
    remaining = target - len(selected)

    yn_take = min(yn_gap, len(yn_bucket), remaining)
    selected.extend(yn_bucket[:yn_take])
    remaining = target - len(selected)

    # Fill rest with other bucket, then leftover text/yn
    leftover_others = other_bucket[:remaining]
    selected.extend(leftover_others)
    remaining = target - len(selected)
    if remaining > 0:
        # backfill with unused text/yn
        extras = text_bucket[text_take:] + yn_bucket[yn_take:]
        extras.sort(key=lambda s: reasoning_length(s), reverse=True)
        selected.extend(extras[:remaining])

    return selected[:target]


# ------------------------------------------------------------------
# Verification: CHECK 1-12
# ------------------------------------------------------------------
def verify_dataset(final: list, strict: bool = True) -> dict:
    errors = []
    warnings = []
    stats = {}

    n = len(final)
    stats["size"] = n
    if not (28000 <= n <= 30000):
        errors.append(f"CHECK 1 size {n} not in [28000, 30000]")

    # CHECK 2: reasoning coverage
    no_reason = [s for s in final if not has_reasoning(s)]
    stats["no_reasoning"] = len(no_reason)
    if no_reason:
        errors.append(f"CHECK 2 reasoning-missing samples: {len(no_reason)}")

    lengths = [reasoning_length(s) for s in final]
    lengths_sorted = sorted(lengths)
    med = lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else 0
    stats["reasoning_median"] = med
    if med < 300:
        errors.append(f"CHECK 3 reasoning median {med} < 300")

    template_n = sum(
        1 for s in final
        if TEMPLATE_PREFIX_RE.match(get_reasoning_text(s).lstrip("<>think\n "))
    )
    stats["template_pct"] = template_n / n if n else 0
    if n and template_n / n >= 0.05:
        errors.append(f"CHECK 4 template-shaped {template_n}/{n} ({100 * template_n / n:.1f}%) >= 5%")

    text_n = sum(1 for s in final if is_text(s["answer"]))
    stats["text_pct"] = text_n / n if n else 0
    if n and text_n / n < 0.25:
        errors.append(f"CHECK 5 text answers {100 * text_n / n:.1f}% < 25%")

    yn_n = sum(1 for s in final if is_yesno(s["answer"]))
    stats["yn_pct"] = yn_n / n if n else 0
    if n and yn_n / n < 0.05:
        warnings.append(f"CHECK 6 Y/N {100 * yn_n / n:.1f}% < 5% (soft)")

    cqa_n = sum(1 for s in final if s.get("source") == "chartqa_train")
    stats["chartqa_pct"] = cqa_n / n if n else 0
    if n and cqa_n / n > 0.35:
        errors.append(f"CHECK 7 chartqa_train {100 * cqa_n / n:.1f}% > 35%")

    # CHECK 8: 50 random image existence
    random.seed(1337)
    sample50 = random.sample(final, min(50, n))
    missing_img = [s["image_path"] for s in sample50
                   if not os.path.exists(s.get("image_path", ""))]
    stats["missing_image_sample"] = len(missing_img)
    if missing_img:
        errors.append(f"CHECK 8 missing images in sample: {len(missing_img)} e.g. {missing_img[:3]}")

    # CHECK 9: answer relaxed match (lightweight — not using extraction module here)
    # Skipping strict check; warn-only.
    stats["check9"] = "skipped"

    # CHECK 10: regression-target source coverage
    src_cnt = Counter(s.get("source") for s in final)
    stats["sources"] = dict(src_cnt)
    if src_cnt.get("kaggle_like", 0) < 1500:
        warnings.append(f"CHECK 10 kaggle_like {src_cnt.get('kaggle_like', 0)} < 1500 (soft)")
    if src_cnt.get("additional", 0) < 800:
        warnings.append(f"CHECK 10 additional {src_cnt.get('additional', 0)} < 800 (soft)")

    # CHECK 11: source distribution sanity (soft)
    # CHECK 12: reasoning < 150 count
    under_150 = sum(1 for l in lengths if l < 150)
    stats["under_150"] = under_150
    if under_150 > 0:
        errors.append(f"CHECK 12 reasoning < 150 chars: {under_150} samples (must be 0)")

    under_300 = sum(1 for l in lengths if l < 300)
    stats["under_300_pct"] = under_300 / n if n else 0
    if n and under_300 / n > 0.15:
        warnings.append(
            f"CHECK 12 reasoning < 300 chars: {100 * under_300 / n:.1f}% > 15% (soft)"
        )

    return {"errors": errors, "warnings": warnings, "stats": stats}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="data/sft_hq_teacher.jsonl")
    ap.add_argument("--teacher_extra", default="data/sft_v8_improved_teacher_distill.jsonl")
    ap.add_argument("--v8_rest_src", default="data/sft_30k.jsonl")
    ap.add_argument("--v9", default="data/sft_v9.jsonl")
    ap.add_argument("--hq_lite", default="data/sft_hq_lite_clean.jsonl")
    ap.add_argument("--gt_pool", default="data/charts_v2/chartvr_train_final.jsonl")
    ap.add_argument("--output", default="data/sft_v8_improved_v2.jsonl")
    ap.add_argument("--target_b1", type=int, default=9500)
    ap.add_argument("--target_b2", type=int, default=8000)
    ap.add_argument("--target_b3", type=int, default=6000)
    ap.add_argument("--target_b4", type=int, default=4000)
    ap.add_argument("--target_b5", type=int, default=2000)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("assemble_v8_improved_v2 — regression-targeted 28K-30K SFT")
    print("=" * 70)

    # Load pools
    print("\n[load]")
    teacher_map = build_teacher_map([args.teacher, args.teacher_extra])
    print(f"  teacher_map (dedup): {len(teacher_map)}")

    hq_lite = load_jsonl(args.hq_lite)
    selfgen_map = build_selfgen_map(hq_lite, min_chars=300)
    print(f"  selfgen_map (>=300, QG): {len(selfgen_map)}")

    v9_all = load_jsonl(args.v9)
    print(f"  v9 pool: {len(v9_all)}")

    v8_rest_all = load_jsonl(args.v8_rest_src)
    v8_rest = [s for s in v8_rest_all if s.get("source") != "chartqa_train"]
    print(f"  v8_rest (non-chartqa): {len(v8_rest)}")

    gt_pool = load_jsonl(args.gt_pool)
    print(f"  gt_pool: {len(gt_pool)}")

    # Block 1
    print("\n[Block 1 — ChartQA teacher 10K]")
    b1 = select_block1_chartqa(teacher_map, args.target_b1)
    print(f"  -> {len(b1)}")

    # Block 2
    print("\n[Block 2 — Scientific 8K]")
    b2, shortfall = select_block2_scientific(
        teacher_map, selfgen_map, v9_all, args.target_b2
    )
    print(f"  -> {len(b2)} (shortfall {shortfall})")

    # Block 3
    print("\n[Block 3 — Complex/Diverse 6K]")
    b3 = select_block3_complex(
        teacher_map, selfgen_map, v9_all, v8_rest, args.target_b3
    )
    print(f"  -> {len(b3)}")

    # Block 4
    print("\n[Block 4 — OWID/Synth/WB 4K]")
    used_keys = {dedup_key(s) for s in b1 + b2 + b3}
    b4 = select_block4_owid(teacher_map, gt_pool, used_keys, args.target_b4)
    print(f"  -> {len(b4)}")

    # Block 5 (absorb shortfall)
    print("\n[Block 5 — Balance + regression targeting]")
    b5_target = args.target_b5 + shortfall
    b5 = select_block5_balance(
        b1 + b2 + b3 + b4, teacher_map, selfgen_map, v9_all,
        hq_lite, v8_rest, b5_target
    )
    print(f"  -> {len(b5)} (target {b5_target})")

    # Combine + final dedup
    combined = b1 + b2 + b3 + b4 + b5
    seen = set()
    final = []
    for s in combined:
        k = dedup_key(s)
        if k in seen:
            continue
        seen.add(k)
        final.append(s)

    random.seed(42)
    random.shuffle(final)

    # Block-level stats
    print("\n[block sizes post-dedup]")
    block_id = {}
    for s in b1:
        block_id[dedup_key(s)] = "B1"
    for s in b2:
        block_id.setdefault(dedup_key(s), "B2")
    for s in b3:
        block_id.setdefault(dedup_key(s), "B3")
    for s in b4:
        block_id.setdefault(dedup_key(s), "B4")
    for s in b5:
        block_id.setdefault(dedup_key(s), "B5")
    bc = Counter(block_id[dedup_key(s)] for s in final)
    for k in ("B1", "B2", "B3", "B4", "B5"):
        print(f"  {k}: {bc.get(k, 0)}")

    # Verify
    print("\n[verify — CHECK 1-12]")
    rep = verify_dataset(final)
    for w in rep["warnings"]:
        print(f"  WARN: {w}")
    if rep["errors"]:
        print("  FAIL:")
        for e in rep["errors"]:
            print(f"    - {e}")
    else:
        print("  ✅ All hard checks passed.")

    # Stats dump
    if args.stats or rep["errors"]:
        st = rep["stats"]
        print("\n[stats]")
        print(f"  N = {st['size']}")
        print(f"  reasoning median = {st['reasoning_median']}")
        print(f"  template-shaped  = {100 * st['template_pct']:.2f}%")
        print(f"  text answers     = {100 * st['text_pct']:.2f}%")
        print(f"  Y/N answers      = {100 * st['yn_pct']:.2f}%")
        print(f"  chartqa share    = {100 * st['chartqa_pct']:.2f}%")
        print(f"  reasoning <150   = {st['under_150']}")
        print(f"  reasoning <300   = {100 * st['under_300_pct']:.2f}%")
        print("  source distribution:")
        for src, cnt in Counter(s.get("source") for s in final).most_common():
            print(f"    {src:>22}: {cnt:>5} ({100 * cnt / len(final):.1f}%)")

    if rep["errors"] and not args.dry_run:
        print("\n❌ CHECK failures — not writing output. Use --dry_run --stats to inspect.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN — no file written]")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"\n✅ Wrote {len(final)} samples → {args.output}")


if __name__ == "__main__":
    main()
