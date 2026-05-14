"""Merge teacher-distilled samples into sft_hq_lite.jsonl.

For samples present in distilled output (with full `assistant_text`), use the
distilled version (higher quality, natural-length reasoning). For samples not
distilled, keep the lite fallback (reasoning_steps or placeholder).

Usage:
    python scripts/merge_distilled_into_lite.py \
        --lite data/sft_hq_lite.jsonl \
        --distilled data/sft_hq_partial.jsonl \
        --output data/sft_hq_augmented.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter


def dedup_key(sample):
    img = sample.get("image_path", "")
    q = (sample.get("question") or "")[:80]
    return f"{img}|{q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", required=True, help="Baseline lite jsonl (fallback source)")
    ap.add_argument("--distilled", required=True, help="Distilled jsonl (higher-quality override)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.lite):
        print(f"ERROR: {args.lite} not found")
        sys.exit(1)
    if not os.path.exists(args.distilled):
        print(f"ERROR: {args.distilled} not found — nothing to merge")
        sys.exit(1)

    # Load distilled into map by dedup key
    distilled_by_key = {}
    with open(args.distilled) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            distilled_by_key[dedup_key(r)] = r

    stats = Counter()
    out_lines = []
    with open(args.lite) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lite_rec = json.loads(line)
            key = dedup_key(lite_rec)
            if key in distilled_by_key:
                d = distilled_by_key[key]
                merged = dict(lite_rec)
                # Preserve assistant_text from distilled
                merged["assistant_text"] = d.get("assistant_text", "")
                merged["distilled_from"] = d.get("distilled_from", "")
                merged["teacher_reasoning_len"] = d.get("teacher_reasoning_len", 0)
                merged["teacher_content_len"] = d.get("teacher_content_len", 0)
                merged["teacher_finish_reason"] = d.get("teacher_finish_reason", "")
                out_lines.append(merged)
                stats["distilled_override"] += 1
            else:
                out_lines.append(lite_rec)
                stats["lite_fallback"] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for rec in out_lines:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"Merged {len(out_lines)} samples → {args.output}")
    print(f"  distilled_override: {stats['distilled_override']}")
    print(f"  lite_fallback:      {stats['lite_fallback']}")


if __name__ == "__main__":
    main()
