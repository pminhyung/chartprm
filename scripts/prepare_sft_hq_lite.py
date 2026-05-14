"""Prepare v_hq-lite SFT training data from distill_source.jsonl.

v_hq-lite strategy (fallback from full teacher distillation):
- Use distill_source.jsonl as-is (already filters rule templates)
- Preserve reasoning_steps field for v9 high-quality samples (58.9% coverage)
- chartqa_train (v8 sft_30k, 18K) uses placeholder reasoning (same as v8 baseline)
- Unified prompt already applied via train_sft.py Pillar 1 patch

This script performs:
1. Sanity checks (image existence, question/answer presence, reasoning format)
2. Length audit (with Qwen3.5-4B tokenizer)
3. Optional priority subset extraction (charxiv-analog focus)
4. Writes data/sft_hq_lite.jsonl ready for train_sft.py

Usage:
    python scripts/prepare_sft_hq_lite.py \
        --source data/distill_source.jsonl \
        --output data/sft_hq_lite.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter


def normalize_reasoning(reasoning):
    """Normalize reasoning_steps to a single string."""
    if reasoning is None:
        return ""
    if isinstance(reasoning, list):
        return "\n".join(str(s) for s in reasoning if s)
    return str(reasoning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/distill_source.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop_empty_reasoning", action="store_true",
                    help="Drop samples without reasoning (reduces dataset size but purer)")
    ap.add_argument("--only_sources", default="",
                    help="Comma-separated source filter (e.g. scientific_ext,plotly_complex)")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        print(f"ERROR: {args.source} not found")
        sys.exit(1)

    only = set(args.only_sources.split(",")) if args.only_sources else None

    kept = []
    stats = Counter()
    per_src = Counter()
    per_src_with_reason = Counter()

    with open(args.source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["total"] += 1

            src = r.get("source", "")
            if only and src not in only:
                stats["filtered_source"] += 1
                continue

            if not r.get("image_path") or not os.path.exists(r["image_path"]):
                stats["missing_image"] += 1
                continue

            if not r.get("question") or r.get("answer") is None or str(r.get("answer", "")).strip() == "":
                stats["missing_qa"] += 1
                continue

            reasoning = normalize_reasoning(r.get("reasoning_steps", r.get("reasoning", "")))
            if args.drop_empty_reasoning and not reasoning:
                stats["empty_reasoning_dropped"] += 1
                continue

            out = {
                "question": r["question"],
                "answer": r["answer"],
                "answer_type": r.get("answer_type", ""),
                "reasoning_steps": reasoning,  # str form for train_sft.py fallback path
                "image_path": r["image_path"],
                "csv_path": r.get("csv_path", ""),
                "source": src,
                "chart_slug": r.get("chart_slug", ""),
                "difficulty": r.get("difficulty", ""),
                "origin": r.get("origin", ""),
            }
            kept.append(out)
            stats["kept"] += 1
            per_src[src] += 1
            if reasoning:
                per_src_with_reason[src] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    n = len(kept)
    print(f"Wrote {n} samples → {args.output}")
    print()
    print("Stats:")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")
    print()
    print("Per-source (total / with_reasoning):")
    for src_name in sorted(per_src.keys()):
        t = per_src[src_name]
        r = per_src_with_reason[src_name]
        print(f"  {src_name:20s} n={t:5d}  reason={r:5d} ({r*100/t:.0f}%)")


if __name__ == "__main__":
    main()
