"""Build SFT-ready data from raw distill_v2 output.

The distill step (`distill_sft_v2.py`) captures raw teacher output. This
script applies per-family cleanup and the only filter that matters for
SFT quality: **format-compliance with the eval-bench scorer**.

Compliance criteria (one per family):
  - chartmuseum  : content must contain a closed `<answer>…</answer>`
                   pair with a non-empty body
  - chartqa      : content must have ≥ 1 non-empty line
  - charxiv      : content must have ≥ 1 non-empty line

Cleanup applied to compliant rows:
  - chartmuseum  : keep only the LAST `<answer>` body (drop any
                   pre/post `</answer>` prose, drop earlier stale tags)
  - chartqa      : last non-empty line, strip trailing punctuation
  - charxiv      : last non-empty line, strip trailing punctuation

`_teacher_match` and `_teacher_finish_reason` are preserved as
diagnostic metadata but are NEVER drop criteria — they don't relate to
eval-bench scorer behavior.

Output schema (per row):
  - all source v8 fields (image_path, question, answer, source, ...)
  - user_text                : the per-family prompt (verbatim from distill)
  - assistant_text           : `<think>{rsn}</think>\\n{family_answer}`
  - _clean_answer            : the bare answer string after cleanup
  - _format_family           : chartqa | charxiv | chartmuseum
  - _teacher_*               : preserved diagnostics
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TRAILING_PUNCT_RE = re.compile(r"[\s.!?,;:]+$")


def clean_chartmuseum(content: str) -> str | None:
    """Return the LAST `<answer>{X}</answer>` body, trimmed.
    None if no closed pair exists or the last body is empty."""
    matches = list(ANSWER_RE.finditer(content or ""))
    if not matches:
        return None
    body = matches[-1].group(1).strip()
    return body or None


def clean_lastline(content: str) -> str | None:
    """Return last non-empty line with trailing punctuation stripped.
    None if content has no non-empty line."""
    lines = [l.strip() for l in (content or "").splitlines() if l.strip()]
    if not lines:
        return None
    return TRAILING_PUNCT_RE.sub("", lines[-1]) or None


def build_assistant_text(rsn: str, family: str, answer: str) -> str:
    rsn = (rsn or "").strip()
    if family == "chartmuseum":
        return f"<think>\n{rsn}\n</think>\n<answer>{answer}</answer>"
    return f"<think>\n{rsn}\n</think>\n{answer}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Raw distill_v2 JSONL (output of distill_sft_v2.py).")
    p.add_argument("--output", required=True,
                   help="SFT-ready JSONL with assistant_text built.")
    p.add_argument("--dropped", default=None,
                   help="Optional JSONL for rows that fail format-compliance.")
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
            family = d.get("_format_family", "")
            rsn = d.get("_teacher_reasoning") or ""
            ct = d.get("_teacher_content") or ""

            if family == "chartmuseum":
                ans = clean_chartmuseum(ct)
                drop_key = "chartmuseum_no_answer_tag"
            elif family in ("chartqa", "charxiv"):
                ans = clean_lastline(ct)
                drop_key = f"{family}_empty_lastline"
            else:
                ans = None
                drop_key = "unknown_family"

            if ans is None:
                drop_reasons[drop_key] += 1
                if drop_f is not None:
                    drop_f.write(json.dumps({**d, "_drop_reason": drop_key}, ensure_ascii=False) + "\n")
                continue

            assistant_text = build_assistant_text(rsn, family, ans)
            out = dict(d)
            out["assistant_text"] = assistant_text
            out["_clean_answer"] = ans
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_kept += 1
            fam_kept[family] += 1

    out_f.close()
    if drop_f is not None:
        drop_f.close()

    print(f"=== build_distill_v2_sft ===")
    print(f"  in={n_in}  kept={n_kept}  dropped={n_in - n_kept}")
    for k, v in sorted(fam_kept.items()):
        print(f"  kept_{k:18s} {v}")
    if drop_reasons:
        print("  drop reasons:")
        for k, v in sorted(drop_reasons.items()):
            print(f"    {k:36s} {v}")


if __name__ == "__main__":
    main()
