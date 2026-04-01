"""Data integrity audit for chartvr_train_final.jsonl (v7)."""
import json
import os
import sys
from collections import Counter

import pandas as pd
from PIL import Image

DATA_PATH = "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/chartvr_train_final.jsonl"
REQUIRED_KEYS = {"question", "answer", "csv_path", "image_path"}

def main():
    with open(DATA_PATH) as f:
        samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(samples)}")

    # Key presence
    missing_keys = []
    for i, s in enumerate(samples):
        miss = REQUIRED_KEYS - set(s.keys())
        if miss:
            missing_keys.append((i, miss))
    if missing_keys:
        print(f"FAIL: {len(missing_keys)} samples missing keys: {missing_keys[:5]}")
    else:
        print("PASS: All required keys present")

    # Source distribution
    sources = Counter(s.get("source", "unknown") for s in samples)
    print(f"Source distribution: {dict(sources)}")

    # Answer check
    null_answers = sum(1 for s in samples if s.get("answer") is None or str(s["answer"]).strip() == "")
    print(f"Null/empty answers: {null_answers}")

    # CSV existence + readability
    csv_missing = 0
    csv_unreadable = 0
    csv_checked = 0
    for s in samples:
        p = s.get("csv_path", "")
        if not p:
            csv_missing += 1
            continue
        if not os.path.exists(p):
            csv_missing += 1
            continue
        csv_checked += 1
        if csv_checked <= len(samples):  # check all
            try:
                df = pd.read_csv(p)
                if df.empty:
                    csv_unreadable += 1
            except Exception:
                csv_unreadable += 1
    print(f"CSV: {csv_missing} missing, {csv_unreadable} unreadable, {len(samples) - csv_missing - csv_unreadable} OK")

    # Image existence + loadability (spot check 100)
    img_missing = 0
    img_unloadable = 0
    for i, s in enumerate(samples):
        p = s.get("image_path", "")
        if not p or not os.path.exists(p):
            img_missing += 1
            continue
        if i < 100:  # spot check first 100
            try:
                img = Image.open(p)
                img.verify()
            except Exception:
                img_unloadable += 1
    print(f"Images: {img_missing} missing, {img_unloadable} unloadable (spot-checked 100)")

    # Difficulty distribution
    diffs = Counter(s.get("difficulty", "unknown") for s in samples)
    print(f"Difficulty distribution: {dict(diffs)}")

    # Chart type distribution (if available)
    chart_types = Counter(s.get("chart_type", s.get("chart_slug", "unknown").split("_")[0]) for s in samples)
    print(f"Chart types (top 10): {dict(chart_types.most_common(10))}")

    # Summary
    issues = csv_missing + csv_unreadable + img_missing + img_unloadable + null_answers + len(missing_keys)
    if issues == 0:
        print("\n✅ ALL CHECKS PASSED")
    else:
        print(f"\n⚠️ {issues} total issues found")

if __name__ == "__main__":
    main()
