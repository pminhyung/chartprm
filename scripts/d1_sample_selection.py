"""D1 sample selection — 100 ChartQA-train + 100 ReachQA-train.

Anti-contamination: NEVER touch test benches (ChartQA-Pro / ChartMuseum / CharXiv-R).
Training-side data only. pHash overlap check against test bench images = 0.

Output:
  data/d1_pilot/samples.jsonl   — 200 sample manifest
  data/d1_pilot/images/         — extracted ReachQA images as PNG
  data/d1_pilot/contamination_report.json — pHash overlap stats

Usage: python scripts/d1_sample_selection.py
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import imagehash
import pyarrow.parquet as pq
from PIL import Image

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
DATA = BASE / "data"

# Training-side sources
CHARTQA_TRAIN_JSON = DATA / "chartqa/train/train_human.json"
CHARTQA_TRAIN_IMG = DATA / "chartqa/train/png"
REACHQA_PARQUET_GLOB = sorted((DATA / "reachqa/data").glob("train-*.parquet"))

# Test benches (input X — pHash check only)
TEST_BENCH_IMG_DIRS = [
    DATA / "chartqa_pro/images",
    DATA / "chartmuseum/images",
    DATA / "charxiv/images",
]

# Output
OUT_DIR = DATA / "d1_pilot"
OUT_JSONL = OUT_DIR / "samples.jsonl"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_REPORT = OUT_DIR / "contamination_report.json"

SEED = 42
N_CHARTQA = 100
N_REACHQA = 100


def build_test_bench_hashes() -> set:
    """pHash every test bench image. Used to pre-filter contaminated train samples."""
    hashes = set()
    for test_dir in TEST_BENCH_IMG_DIRS:
        if not test_dir.exists():
            print(f"[phash-prefilter] WARN test dir missing: {test_dir}", file=sys.stderr)
            continue
        n = 0
        for img_path in test_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                with Image.open(img_path) as ti:
                    ti.load()
                    hashes.add(imagehash.phash(ti))
                n += 1
            except Exception:
                continue
        print(f"[phash-prefilter] {test_dir.name}: hashed {n}")
    print(f"[phash-prefilter] total unique hashes: {len(hashes)}")
    return hashes


def is_contaminated(img_hash, test_hashes: set, threshold: int = 5) -> bool:
    """True if img_hash is within Hamming threshold of any test bench hash."""
    for th in test_hashes:
        if abs(img_hash - th) < threshold:
            return True
    return False


def select_chartqa_train(rng: random.Random, test_hashes: set) -> list[dict]:
    """Filter ChartQA train: word count >= 8 + no pHash overlap with test benches."""
    with open(CHARTQA_TRAIN_JSON) as f:
        raw = json.load(f)
    pool = []
    n_word_filtered = 0
    n_contaminated = 0
    for i, r in enumerate(raw):
        q = r.get("query", "").strip()
        if len(q.split()) < 8:
            n_word_filtered += 1
            continue
        img_path = CHARTQA_TRAIN_IMG / r["imgname"]
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as ti:
                ti.load()
                ih = imagehash.phash(ti)
        except Exception:
            continue
        if is_contaminated(ih, test_hashes):
            n_contaminated += 1
            continue
        pool.append({
            "id": f"chartqa_train_{i}",
            "source": "chartqa_train",
            "image_path": str(img_path),
            "question": q,
            "gold_answer": str(r.get("label", "")),
            "qa_type": "factoid_or_compositional",
            "chart_type": None,
        })
    print(f"[chartqa] raw={len(raw)} word_filtered={n_word_filtered} contaminated={n_contaminated} clean_pool={len(pool)}")
    return rng.sample(pool, N_CHARTQA)


def select_reachqa_train(rng: random.Random, test_hashes: set) -> tuple[list[dict], list[Image.Image]]:
    """Filter ReachQA: qa_type=Reasoning + no pHash overlap. Extract images to disk."""
    pool_meta = []  # (shard_idx, row_idx, row_dict, img_obj)
    n_contaminated = 0
    for shard_idx, parquet_path in enumerate(REACHQA_PARQUET_GLOB):
        t = pq.read_table(parquet_path)
        for row_idx, r in enumerate(t.to_pylist()):
            if r["qa_type"] != "Reasoning":
                continue
            try:
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                ih = imagehash.phash(img)
            except Exception:
                continue
            if is_contaminated(ih, test_hashes):
                n_contaminated += 1
                continue
            pool_meta.append((shard_idx, row_idx, r, img))
    print(f"[reachqa] pool size (qa_type=Reasoning, clean): {len(pool_meta)} contaminated={n_contaminated}")
    picked = rng.sample(pool_meta, N_REACHQA)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    images_for_phash = []
    for shard_idx, row_idx, r, img in picked:
        img_name = f"reachqa_s{shard_idx}_r{row_idx}.png"
        img_path = OUT_IMG_DIR / img_name
        img.save(img_path, format="PNG")
        samples.append({
            "id": f"reachqa_train_s{shard_idx}_r{row_idx}",
            "source": "reachqa_train",
            "image_path": str(img_path),
            "question": r["question"],
            "gold_answer": r["answer"],
            "qa_type": "Reasoning",
            "chart_type": r["chart_type"],
        })
        images_for_phash.append(img)
    return samples, images_for_phash


def compute_phash_overlap(train_images: list[Image.Image]) -> dict:
    """Hash all test-bench images, check overlap with our training samples."""
    train_hashes = []
    for img in train_images:
        try:
            train_hashes.append(imagehash.phash(img))
        except Exception as e:
            print(f"[phash] train img skipped: {e}", file=sys.stderr)
    print(f"[phash] train hashes: {len(train_hashes)}")

    test_count = 0
    overlap_pairs = []
    for test_dir in TEST_BENCH_IMG_DIRS:
        if not test_dir.exists():
            print(f"[phash] WARN test dir missing: {test_dir}", file=sys.stderr)
            continue
        for img_path in test_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            test_count += 1
            try:
                with Image.open(img_path) as ti:
                    ti.load()
                    th = imagehash.phash(ti)
            except Exception:
                continue
            for ih, h in enumerate(train_hashes):
                if abs(h - th) < 5:
                    overlap_pairs.append({
                        "train_idx": ih,
                        "test_path": str(img_path),
                        "hamming": int(abs(h - th)),
                    })
    print(f"[phash] test images hashed: {test_count}")
    print(f"[phash] overlapping pairs (Hamming<5): {len(overlap_pairs)}")
    return {
        "n_train_hashed": len(train_hashes),
        "n_test_hashed": test_count,
        "overlap_pairs": overlap_pairs,
    }


def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"D1 sample selection (seed={SEED})")
    print(f"  ChartQA-train target: {N_CHARTQA}")
    print(f"  ReachQA-train target: {N_REACHQA}")
    print("=" * 60)

    print("\n[step 1/3] Hashing test bench images (chartqa_pro + chartmuseum + charxiv)")
    test_hashes = build_test_bench_hashes()

    print("\n[step 2/3] Selecting ChartQA-train (clean)")
    chartqa = select_chartqa_train(rng, test_hashes)

    print("\n[step 3/3] Selecting ReachQA-train (clean)")
    reachqa, reachqa_images = select_reachqa_train(rng, test_hashes)
    samples = chartqa + reachqa

    # Source distribution
    src_counts = Counter(s["source"] for s in samples)
    print(f"\n[summary] source distribution: {dict(src_counts)}")

    # Write manifest
    with open(OUT_JSONL, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[output] wrote {len(samples)} samples -> {OUT_JSONL}")

    # pHash overlap check (use Image objects from disk for both pools)
    print("\n" + "=" * 60)
    print("Anti-contamination pHash check")
    print("=" * 60)
    chartqa_imgs = []
    for s in chartqa:
        try:
            chartqa_imgs.append(Image.open(s["image_path"]).convert("RGB"))
        except Exception:
            pass
    train_images_all = chartqa_imgs + reachqa_images
    report = compute_phash_overlap(train_images_all)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[output] contamination report -> {OUT_REPORT}")

    if report["overlap_pairs"]:
        print(f"\n⚠️  WARNING: {len(report['overlap_pairs'])} overlap(s) detected. Review {OUT_REPORT}", file=sys.stderr)
        return 1
    print("\n✅ Anti-contamination: 0 overlap with test benches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
