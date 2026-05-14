#!/usr/bin/env python
"""Assemble GRPO and SFT datasets from generated QA pairs.

GRPO (~8K): CSV + numeric answer + difficulty filtered (30-70%)
SFT (~30K): GRPO all + ChartQA train + easy/hard rejects

Usage:
  python scripts/assemble_datasets.py \
      --rule_qa data/charts_v2/rule_based_qa.jsonl \
      --llm_qa data/charts_v2/llm_qa.jsonl \
      --difficulty_filtered data/charts_v2/difficulty_filtered_v2.jsonl \
      --chartqa_train data/chartqa/train/ \
      --grpo_output data/grpo_v2.jsonl \
      --sft_output data/sft_30k.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

from chartvr.extraction import _is_numeric_answer


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def validate_sample(rec):
    """Basic validation: has required fields and files exist."""
    required = ["question", "answer", "image_path"]
    for r in required:
        if r not in rec or not rec[r]:
            return False
    if not os.path.exists(rec["image_path"]):
        return False
    return True


def load_chartqa_train(chartqa_dir):
    """Load ChartQA train split as SFT samples."""
    samples = []
    train_json = os.path.join(chartqa_dir, "train_human.json")
    train_aug = os.path.join(chartqa_dir, "train_augmented.json")

    for path in [train_json, train_aug]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            img_name = item.get("imgname", "")
            img_path = os.path.join(chartqa_dir, "png", img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(chartqa_dir, img_name)
            samples.append({
                "question": item.get("query", ""),
                "answer": str(item.get("label", "")),
                "image_path": img_path,
                "csv_path": "",
                "source": "chartqa_train",
                "chart_slug": f"cqa_{img_name}",
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Assemble GRPO and SFT datasets")
    parser.add_argument("--rule_qa", default="data/charts_v2/rule_based_qa.jsonl")
    parser.add_argument("--llm_qa", default="data/charts_v2/llm_qa.jsonl")
    parser.add_argument("--difficulty_filtered", default="data/charts_v2/difficulty_filtered_v2.jsonl")
    parser.add_argument("--existing_grpo", default="data/charts_v2/chartvr_train_final.jsonl",
                        help="Existing v7 GRPO data to include")
    parser.add_argument("--chartqa_train", default="data/chartqa/train/",
                        help="ChartQA train directory (contains train_human.json, train_augmented.json, png/)")
    parser.add_argument("--grpo_output", default="data/grpo_v2.jsonl")
    parser.add_argument("--sft_output", default="data/sft_30k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── GRPO Dataset ──
    print("=== GRPO Dataset ===")

    # 1. Difficulty-filtered QAs (primary GRPO source)
    grpo_samples = []
    if os.path.exists(args.difficulty_filtered):
        filtered = load_jsonl(args.difficulty_filtered)
        for rec in filtered:
            if validate_sample(rec) and rec.get("csv_path") and os.path.exists(rec.get("csv_path", "")):
                if _is_numeric_answer(str(rec["answer"])):
                    grpo_samples.append(rec)
        print(f"  Difficulty filtered (numeric + CSV): {len(grpo_samples)}")

    # 2. Existing v7 GRPO data
    existing = load_jsonl(args.existing_grpo)
    existing_valid = [r for r in existing if validate_sample(r)]
    print(f"  Existing v7 GRPO data: {len(existing_valid)}")

    # Merge, dedup by chart_slug + question
    seen = set()
    grpo_final = []
    for rec in grpo_samples + existing_valid:
        key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:80]}"
        if key not in seen:
            seen.add(key)
            grpo_final.append(rec)

    random.shuffle(grpo_final)
    print(f"  GRPO total (deduped): {len(grpo_final)}")

    os.makedirs(os.path.dirname(args.grpo_output) or ".", exist_ok=True)
    with open(args.grpo_output, "w") as f:
        for rec in grpo_final:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved to {args.grpo_output}")

    # ── SFT Dataset ──
    print("\n=== SFT Dataset ===")

    sft_samples = []

    # 1. All GRPO data
    sft_samples.extend(grpo_final)
    print(f"  From GRPO: {len(grpo_final)}")

    # 2. Rule-based QA (including non-filtered)
    rule_qa = load_jsonl(args.rule_qa)
    rule_valid = [r for r in rule_qa if validate_sample(r)]
    print(f"  Rule-based QA: {len(rule_valid)}")
    sft_samples.extend(rule_valid)

    # 3. LLM-generated QA
    llm_qa = load_jsonl(args.llm_qa)
    llm_valid = [r for r in llm_qa if validate_sample(r)]
    print(f"  LLM-generated QA: {len(llm_valid)}")
    sft_samples.extend(llm_valid)

    # 4. ChartQA train
    cqa_samples = load_chartqa_train(args.chartqa_train)
    cqa_valid = [r for r in cqa_samples if validate_sample(r)]
    print(f"  ChartQA train: {len(cqa_valid)}")
    sft_samples.extend(cqa_valid)

    # Dedup
    seen_sft = set()
    sft_final = []
    for rec in sft_samples:
        key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:80]}"
        if key not in seen_sft:
            seen_sft.add(key)
            sft_final.append(rec)

    random.shuffle(sft_final)
    print(f"  SFT total (deduped): {len(sft_final)}")

    # Stats
    text_count = sum(1 for r in sft_final if not _is_numeric_answer(str(r.get("answer", ""))))
    print(f"  Text answers: {text_count} ({text_count/max(len(sft_final),1)*100:.1f}%)")

    sources = {}
    for r in sft_final:
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    with open(args.sft_output, "w") as f:
        for rec in sft_final:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"  Saved to {args.sft_output}")


if __name__ == "__main__":
    main()
