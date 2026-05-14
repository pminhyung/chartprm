"""Convert v8_improved_v2 SFT data to standard-eval-aligned format.

Source: data/sft_v8_improved_v2.jsonl
  - 28,594 rows, teacher-distilled CoT in `assistant_text` (22K with full
    `<think>...</think><answer>X</answer>` tags) or fallback via `reasoning`/
    `reasoning_steps` (6.4K rows).

Target: data/sft_v8_standard.jsonl
  - Same row count, same fields preserved.
  - `assistant_text` rewritten so the answer is plain text after `</think>`,
    matching standard eval (ChartQA/CharXiv) which has NO `<answer>` tag.
  - The `<think>...</think>` reasoning block is preserved verbatim.

Format change examples:

  before:  <think>step 1\nstep 2</think><answer>83.8%</answer>
  after :  <think>step 1\nstep 2</think>\n83.8%

  before (fallback):  reasoning="step 1; step 2", answer="83.8"
  after :             <think>step 1; step 2</think>\n83.8

The user-turn / system-message changes are handled in train_sft.py separately;
this script only normalizes the assistant target string.

Usage:
  python scripts/convert_v8_to_standard_format.py \
      --input data/sft_v8_improved_v2.jsonl \
      --output data/sft_v8_standard.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)
TRAILING_ANSWER_RE = re.compile(r"\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)
THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)


def _strip_answer_tags(text: str, fallback_answer: str) -> str:
    """Convert assistant_text from `<think>...</think><answer>X</answer>` to
    `<think>...</think>\n<X>`. If `<answer>` is missing, append fallback.

    Also handles cases where `<answer>` appears without `<think>` (rare).
    """
    if not text:
        return ""
    s = text.strip()
    m = TRAILING_ANSWER_RE.search(s)
    if m:
        ans = m.group(1).strip()
        # remove the entire trailing <answer>...</answer> block
        head = s[: m.start()].rstrip()
        # ensure single newline between </think> and answer
        if head.endswith("</think>"):
            return head + "\n" + ans
        return head + "\n" + ans if head else ans
    # No <answer> tag — keep as is, but if it ends with </think>, append answer
    if s.endswith("</think>"):
        return s + "\n" + fallback_answer.strip()
    return s


def _build_from_reasoning(rec: dict) -> str | None:
    """Fallback: build `<think>{reasoning}</think>\n{answer}` from
    `reasoning`/`reasoning_steps` field. Returns None if no reasoning.
    """
    reasoning = rec.get("reasoning", rec.get("reasoning_steps", ""))
    if isinstance(reasoning, list):
        reasoning = "\n".join(str(s) for s in reasoning)
    reasoning = (reasoning or "").strip()
    if not reasoning:
        return None
    answer = str(rec.get("answer", "")).strip()
    return f"<think>\n{reasoning}\n</think>\n{answer}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sft_v8_improved_v2.jsonl")
    p.add_argument("--output", default="data/sft_v8_standard.jsonl")
    args = p.parse_args()

    stats = Counter()
    src_dist = Counter()
    out_rows = []

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["total"] += 1
            src_dist[rec.get("source", "?")] += 1

            assistant_text = (rec.get("assistant_text") or "").strip()
            answer = str(rec.get("answer", "")).strip()

            if assistant_text:
                if "<answer>" in assistant_text:
                    new_text = _strip_answer_tags(assistant_text, answer)
                    stats["had_answer_tag"] += 1
                else:
                    new_text = assistant_text
                    stats["no_answer_tag_kept"] += 1
            else:
                fallback = _build_from_reasoning(rec)
                if fallback is None:
                    stats["dropped_no_reasoning"] += 1
                    continue
                new_text = fallback
                stats["from_fallback"] += 1

            # Sanity: must contain <think> open
            if "<think>" not in new_text:
                stats["missing_think_tag"] += 1
                # still write — train_sft.py will accept; format mismatch will hurt
            # Sanity: must NOT contain residual <answer> tag
            if "<answer>" in new_text or "</answer>" in new_text:
                stats["residual_answer_tag"] += 1
                # remove any stragglers
                new_text = re.sub(r"</?answer>", "", new_text)

            out_rec = dict(rec)
            out_rec["assistant_text"] = new_text
            out_rec["_format"] = "standard_v1"
            out_rows.append(out_rec)
            stats["written"] += 1

    with open(args.output, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== Conversion complete ===")
    print(f"input : {args.input}")
    print(f"output: {args.output}")
    print(f"\nCounts:")
    for k in ("total", "had_answer_tag", "no_answer_tag_kept", "from_fallback",
              "dropped_no_reasoning", "missing_think_tag", "residual_answer_tag", "written"):
        print(f"  {k:<28} {stats[k]:>6}")
    print(f"\nSource distribution (preserved):")
    for k, v in sorted(src_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v:>6}")


if __name__ == "__main__":
    main()
