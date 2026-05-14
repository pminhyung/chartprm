"""Quality filter for SFT corpus, aligned with eval-bench scorer expectations.

Reads `data/sft_v8_distill_v2_sft.jsonl` (post-format-compliance build) and
writes a stricter `*_filtered.jsonl` for actual training. Rows that fail the
filter are dumped to `*_filtered_dropped.jsonl` with `_drop_reason`.

The unfiltered file is preserved so the original distill corpus stays
recoverable and the filter can be re-tuned without re-distilling.

Filter rules (drop if ANY trigger):

  R1. `len(_teacher_reasoning) < 150` — shallow reasoning gives weak SFT signal.

  R2. ChartMuseum family AND `len(_clean_answer) > 200` — verbose answer body.
       Eval regex captures whatever's inside `<answer>...</answer>`; a full
       sentence in there means the student would learn to dump prose.

  R3. ChartMuseum family AND newline inside `_clean_answer` — multi-line
       answer body, same concern.

  R4. `_teacher_match == False` AND it represents a real teacher error (not
       a form-mismatch artifact):
         - answer_type == "numeric" (5% tolerance already; mismatch = bad)
         - multichoice with VALUE-form gold (teacher should emit value)
       Letter-form multichoice mismatches are KEPT — teacher emits value
       per the value-form prompt, so `_teacher_match=False` is expected.

Text / yesno / non-multichoice text mismatches are kept; the eval LLM
judge tolerates lexical variation (e.g., "decreasing" vs "decreases as X").
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

LETTER_RE = re.compile(r"^[A-Ea-e]$")


def quality_check(row: dict, *, min_reasoning_chars: int, max_cm_answer_chars: int) -> tuple[bool, str]:
    """Return (passes, drop_reason_or_empty)."""
    rsn = row.get("_teacher_reasoning") or ""
    if len(rsn) < min_reasoning_chars:
        return False, "R1_reasoning_too_short"

    family = row.get("_format_family", "")
    clean = row.get("_clean_answer", "") or ""

    if family == "chartmuseum":
        if len(clean) > max_cm_answer_chars:
            return False, "R2_cm_answer_body_long"
        if "\n" in clean:
            return False, "R3_cm_answer_body_multiline"

    teacher_match = bool(row.get("_teacher_match", True))
    if not teacher_match:
        answer_type = row.get("answer_type", "") or ""
        is_mc = bool(row.get("_is_multichoice"))
        gold = str(row.get("answer", "")).strip()
        gold_is_letter = bool(LETTER_RE.match(gold))

        if answer_type == "numeric":
            return False, "R4a_numeric_mismatch"
        if is_mc and not gold_is_letter:
            return False, "R4b_mc_value_form_mismatch"
        # Letter-form multichoice or text/yesno mismatch → keep (form artifact
        # or LLM-judge-tolerable lexical variation).

    return True, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="SFT-ready JSONL (output of build_distill_v2_sft.py).")
    p.add_argument("--output", required=True, help="Filtered SFT JSONL.")
    p.add_argument("--dropped", default=None, help="Optional path for filtered-out rows.")
    p.add_argument("--min_reasoning_chars", type=int, default=150)
    p.add_argument("--max_cm_answer_chars", type=int, default=200)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drop_f = open(args.dropped, "w") if args.dropped else None
    out_f = open(out_path, "w")

    n_in = n_kept = 0
    drop_reasons: Counter = Counter()
    fam_kept: Counter = Counter()

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            ok, reason = quality_check(
                d,
                min_reasoning_chars=args.min_reasoning_chars,
                max_cm_answer_chars=args.max_cm_answer_chars,
            )
            if not ok:
                drop_reasons[reason] += 1
                if drop_f is not None:
                    drop_f.write(json.dumps({**d, "_drop_reason": reason}, ensure_ascii=False) + "\n")
                continue
            out_f.write(line + "\n")
            n_kept += 1
            fam_kept[d.get("_format_family", "?")] += 1

    out_f.close()
    if drop_f is not None:
        drop_f.close()

    print("=== filter_sft_quality ===")
    print(f"  in={n_in}  kept={n_kept}  dropped={n_in - n_kept}  ({(n_in-n_kept)*100/max(1,n_in):.1f}%)")
    for k, v in sorted(fam_kept.items()):
        print(f"  kept_{k:18s} {v}")
    if drop_reasons:
        print("  drop reasons:")
        for k, v in sorted(drop_reasons.items()):
            print(f"    {k:36s} {v}")


if __name__ == "__main__":
    main()
