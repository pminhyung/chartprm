"""
Phase 0.6: Convert all training datasets to unified GRPO format.
Each sample: {"problem": str, "solution": str, "image": str, "csv_path": str}

CRITICAL: csv_path must point to a real, parseable CSV file.
"""
import json
import os
import random
import pandas as pd

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")


def load_chartqa_train():
    """Load ChartQA train split with CSV paths."""
    samples = []
    csv_valid_cache = {}  # Cache CSV validation to avoid redundant parsing
    skip_reasons = {"missing_field": 0, "no_csv": 0, "no_image": 0, "bad_csv": 0}

    for split_file in ["train_human.json", "train_augmented.json"]:
        path = os.path.join(BASE, f"data/chartqa/train/{split_file}")
        if not os.path.exists(path):
            print(f"  Warning: {split_file} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        for item in data:
            # ChartQA uses: query, label, imgname
            question = item.get("query") or item.get("question")
            raw_answer = item.get("label") if item.get("label") is not None else item.get("answer")
            image = item.get("imgname") or item.get("image")

            if not question or raw_answer is None or not image:
                skip_reasons["missing_field"] += 1
                continue

            answer = str(raw_answer)

            # Derive CSV path from image name
            img_base = os.path.splitext(image)[0]
            csv_path = os.path.join(BASE, f"data/chartqa/train/tables/{img_base}.csv")
            image_path = os.path.join(BASE, f"data/chartqa/train/png/{image}")

            if not os.path.exists(image_path):
                skip_reasons["no_image"] += 1
                continue

            if not os.path.exists(csv_path):
                skip_reasons["no_csv"] += 1
                continue

            # Validate CSV (cached)
            if csv_path not in csv_valid_cache:
                try:
                    df = pd.read_csv(csv_path)
                    # Charts need at least 2 columns (axis + values)
                    csv_valid_cache[csv_path] = len(df) > 0 and len(df.columns) >= 2
                except Exception:
                    csv_valid_cache[csv_path] = False

            if not csv_valid_cache[csv_path]:
                skip_reasons["bad_csv"] += 1
                continue

            samples.append({
                "problem": question,
                "solution": answer,
                "image": image_path,
                "csv_path": csv_path,
                "source": "chartqa",
                "split_file": split_file
            })

    print(f"ChartQA train: {len(samples)} samples with valid CSV")
    print(f"  Skipped: {skip_reasons}")
    print(f"  Unique CSVs validated: {len(csv_valid_cache)}")
    return samples


def prepare_grpo_data():
    """Create unified GRPO training data."""
    all_samples = []

    # ChartQA (primary)
    chartqa = load_chartqa_train()
    all_samples.extend(chartqa)

    # PlotQA, DVQA, FigureQA — add loaders here if downloaded
    # For now ChartQA alone is sufficient per Chart-RL finding

    if not all_samples:
        print("ERROR: No valid samples found. Check data paths.")
        return []

    random.seed(42)
    random.shuffle(all_samples)

    output_path = os.path.join(BASE, "data/grpo_train.json")
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"Total GRPO data: {len(all_samples)} → {output_path}")

    # Print statistics
    sources = {}
    for s in all_samples:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"By source: {sources}")

    # Verify a random sample
    sample = random.choice(all_samples)
    print(f"\nRandom sample check:")
    print(f"  problem: {sample['problem'][:100]}")
    print(f"  solution: {sample['solution']}")
    print(f"  image exists: {os.path.exists(sample['image'])}")
    print(f"  csv exists: {os.path.exists(sample['csv_path'])}")

    return all_samples


if __name__ == "__main__":
    prepare_grpo_data()
