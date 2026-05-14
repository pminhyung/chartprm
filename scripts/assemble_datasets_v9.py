#!/usr/bin/env python
"""Assemble v9 SFT and GRPO datasets from all data sources.

SFT (~40K): All blocks with CoT reasoning
GRPO (~6.4K): Existing 3.9K + filtered new data with CSV

Usage:
  python scripts/assemble_datasets_v9.py \
      --sft_output data/sft_v9.jsonl \
      --grpo_output data/grpo_v9.jsonl \
      --sft_limit 40000
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.extraction import _is_numeric_answer

BASE = "/ex_disk2/mhpark/poc/chartvr"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def validate_sample(rec):
    """Basic validation: required fields + files exist."""
    for r in ["question", "answer", "image_path"]:
        if r not in rec or not rec[r]:
            return False
    if not os.path.exists(rec["image_path"]):
        return False
    return True


def validate_cot(rec):
    """Validate CoT quality. Returns True if passes quality gates."""
    reasoning = rec.get("reasoning", rec.get("reasoning_steps", ""))
    if isinstance(reasoning, list):
        reasoning = "\n".join(str(s) for s in reasoning)
    reasoning = str(reasoning)

    # Gate 1: Non-empty
    if not reasoning or reasoning.strip() == "":
        return False

    # Gate 2: Not just a placeholder
    if reasoning.strip() in ("", "N/A", "None"):
        return False

    # Gate 3: Length check (50-2000 tokens, approximate by words)
    word_count = len(reasoning.split())
    if word_count < 10:  # Very short — likely placeholder
        return False
    if word_count > 2000:  # Too long — truncation risk
        return False

    return True


def validate_grpo(rec):
    """Validate GRPO eligibility."""
    if not validate_sample(rec):
        return False
    csv_path = rec.get("csv_path", "")
    if not csv_path or not os.path.exists(csv_path):
        return False
    if not _is_numeric_answer(str(rec.get("answer", ""))):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Assemble v9 SFT + GRPO datasets")
    parser.add_argument("--sft_output", default="data/sft_v9.jsonl")
    parser.add_argument("--grpo_output", default="data/grpo_v9.jsonl")
    parser.add_argument("--sft_limit", type=int, default=None, help="Max SFT samples (stratified)")
    parser.add_argument("--block_c_cap", type=int, default=None,
                        help="Optional cap on Block C samples before merging (stratified by source file)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ═══════════════════════════════════════
    # SFT Dataset
    # ═══════════════════════════════════════
    print("=== SFT Dataset ===")

    sft_sources = {
        "block_a": [
            f"{BASE}/data/charts_v9/rule_qa_block_a_v2.jsonl",
        ],
        "block_b": [
            f"{BASE}/data/charts_v9/rule_qa_block_b_v2.jsonl",
        ],
        "block_c": [
            f"{BASE}/data/charts_v9/block_c_rule_qa_v3.jsonl",    # v9.1: per-type dispatcher
            f"{BASE}/data/charts_v9/block_c_api_qa_v2.jsonl",     # v9.1: sci_* sub-prompts, 1 QA/chart
        ],
        "block_d": [
            f"{BASE}/data/charts_v9/block_d_rule_qa_v2.jsonl",
            f"{BASE}/data/charts_v9/block_d_api_qa.jsonl",
        ],
        "block_e": [
            f"{BASE}/data/charts_v2/chartvr_train_final.jsonl",
        ],
        "block_f": [
            f"{BASE}/data/charts_v9/rule_qa_block_f_v2.jsonl",
        ],
        "chartqa_train": [
            f"{BASE}/data/charts_v9/chartqa_with_cot.jsonl",
            f"{BASE}/data/charts_v9/chartqa_cot_api.jsonl",
        ],
        "rule_qa_with_cot": [
            f"{BASE}/data/charts_v9/rule_qa_with_cot_v2.jsonl",
        ],
    }

    sft_by_source = {}
    seen_sft = set()

    for source_name, paths in sft_sources.items():
        records = []
        for p in paths:
            raw = load_jsonl(p)
            print(f"  {p}: {len(raw)} records")
            records.extend(raw)

        # Validate and dedup
        valid = []
        for rec in records:
            if not validate_sample(rec):
                continue

            key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:80]}"
            if key in seen_sft:
                continue
            seen_sft.add(key)

            # Ensure source field
            if "source" not in rec or not rec["source"]:
                rec["source"] = source_name

            valid.append(rec)

        # Block C cap (applied before merging into sft_all)
        if source_name == "block_c" and args.block_c_cap and len(valid) > args.block_c_cap:
            print(f"  → {source_name}: capping from {len(valid)} to {args.block_c_cap}")
            valid = random.sample(valid, args.block_c_cap)

        sft_by_source[source_name] = valid
        print(f"  → {source_name}: {len(valid)} valid (deduped)")

    # Merge all
    sft_all = []
    for source_name, records in sft_by_source.items():
        sft_all.extend(records)

    print(f"\n  SFT total (before limit): {len(sft_all)}")

    # Stratified sampling if limit set
    if args.sft_limit and len(sft_all) > args.sft_limit:
        # Keep proportion by source
        ratio = args.sft_limit / len(sft_all)
        sampled = []
        for source_name, records in sft_by_source.items():
            n = max(1, int(len(records) * ratio))
            sampled.extend(random.sample(records, min(n, len(records))))
        # Fill remainder randomly
        remaining = [r for r in sft_all if r not in sampled]
        random.shuffle(remaining)
        while len(sampled) < args.sft_limit and remaining:
            sampled.append(remaining.pop())
        sft_all = sampled
        print(f"  SFT after stratified sampling: {len(sft_all)}")

    random.shuffle(sft_all)

    # Stats
    text_count = sum(1 for r in sft_all if not _is_numeric_answer(str(r.get("answer", ""))))
    has_reasoning = sum(1 for r in sft_all if validate_cot(r))
    print(f"\n  SFT final: {len(sft_all)}")
    print(f"  With reasoning: {has_reasoning} ({has_reasoning/max(len(sft_all),1)*100:.1f}%)")
    print(f"  Text answers: {text_count} ({text_count/max(len(sft_all),1)*100:.1f}%)")

    # Source distribution
    sources = {}
    for r in sft_all:
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    print(f"\n  Source distribution:")
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c} ({c/max(len(sft_all),1)*100:.1f}%)")

    # Answer type distribution
    answer_types = {}
    for r in sft_all:
        at = r.get("answer_type", "numeric" if _is_numeric_answer(str(r.get("answer", ""))) else "text")
        answer_types[at] = answer_types.get(at, 0) + 1
    print(f"\n  Answer type distribution:")
    for at, c in sorted(answer_types.items(), key=lambda x: -x[1]):
        print(f"    {at}: {c} ({c/max(len(sft_all),1)*100:.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.sft_output) or ".", exist_ok=True)
    with open(args.sft_output, "w") as f:
        for rec in sft_all:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"\n  Saved to {args.sft_output}")

    # ═══════════════════════════════════════
    # GRPO Dataset
    # ═══════════════════════════════════════
    print("\n=== GRPO Dataset ===")

    grpo_sources = [
        f"{BASE}/data/charts_v2/chartvr_train_final.jsonl",
        f"{BASE}/data/charts_v9/grpo_filtered_v9.jsonl",
    ]

    grpo_all = []
    seen_grpo = set()

    for p in grpo_sources:
        raw = load_jsonl(p)
        valid = 0
        for rec in raw:
            if not validate_grpo(rec):
                continue
            key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:80]}"
            if key in seen_grpo:
                continue
            seen_grpo.add(key)
            grpo_all.append(rec)
            valid += 1
        print(f"  {p}: {len(raw)} raw → {valid} valid")

    random.shuffle(grpo_all)
    print(f"\n  GRPO total: {len(grpo_all)}")

    # Save
    os.makedirs(os.path.dirname(args.grpo_output) or ".", exist_ok=True)
    with open(args.grpo_output, "w") as f:
        for rec in grpo_all:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved to {args.grpo_output}")


if __name__ == "__main__":
    main()
