"""
Phase 0.5: Verify ChartQA data tables exist and are parseable.
CHECKPOINT: All assertions must pass before proceeding to GRPO data prep.
"""
import os
import json
import glob
import pandas as pd


def verify_chartqa():
    """Verify ChartQA CSV tables and PNG images exist and align."""
    base = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")
    tables_dir = os.path.join(base, "data/chartqa/train/tables")
    png_dir = os.path.join(base, "data/chartqa/train/png")

    csvs = glob.glob(os.path.join(tables_dir, "*.csv"))
    pngs = glob.glob(os.path.join(png_dir, "*.png"))

    assert len(csvs) > 4000, f"Expected >4000 CSVs, got {len(csvs)}"
    assert len(pngs) > 4000, f"Expected >4000 PNGs, got {len(pngs)}"

    # Verify CSV-PNG alignment
    csv_names = {os.path.splitext(os.path.basename(f))[0] for f in csvs}
    png_names = {os.path.splitext(os.path.basename(f))[0] for f in pngs}
    overlap = csv_names & png_names
    assert len(overlap) > 3500, f"CSV-PNG overlap only {len(overlap)}"

    # Verify CSV is parseable
    sample_csv = csvs[0]
    df = pd.read_csv(sample_csv)
    assert len(df) > 0, f"Empty CSV: {sample_csv}"
    assert len(df.columns) >= 2, f"Too few columns: {sample_csv}"

    # Count numeric values per table (sample 100)
    numeric_counts = []
    for csv_path in csvs[:100]:
        try:
            df = pd.read_csv(csv_path)
            n_numeric = sum(
                pd.to_numeric(df[col], errors='coerce').notna().sum()
                for col in df.columns
            )
            numeric_counts.append(n_numeric)
        except Exception as e:
            print(f"  Warning: could not parse {csv_path}: {e}")

    avg_numeric = sum(numeric_counts) / max(len(numeric_counts), 1)
    print(f"ChartQA: {len(csvs)} tables, {len(pngs)} images, "
          f"avg {avg_numeric:.1f} numeric values per table")
    print(f"  CSV-PNG overlap: {len(overlap)}")

    return True


def verify_chartqa_qa_format():
    """Verify QA JSON format matches expected structure."""
    base = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

    for split_file in ["train_human.json", "train_augmented.json"]:
        path = os.path.join(base, f"data/chartqa/train/{split_file}")
        if not os.path.exists(path):
            print(f"  Warning: {split_file} not found at {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        assert isinstance(data, list), f"{split_file} is not a list"
        assert len(data) > 2000, f"{split_file} has only {len(data)} samples"

        sample = data[0]
        # Check for expected keys (may vary by version)
        has_question = "question" in sample or "query" in sample
        has_answer = "answer" in sample or "label" in sample
        has_image = "imgname" in sample or "image" in sample

        assert has_question, f"No question key in {split_file}: {list(sample.keys())}"
        assert has_answer, f"No answer key in {split_file}: {list(sample.keys())}"
        assert has_image, f"No image key in {split_file}: {list(sample.keys())}"

        print(f"{split_file}: {len(data)} QA pairs")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Sample: {json.dumps(sample, indent=2)[:300]}")

    # Also check test split
    for test_file in ["test_human.json", "test_augmented.json"]:
        path = os.path.join(base, f"data/chartqa/test/{test_file}")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"{test_file}: {len(data)} QA pairs")

    return True


if __name__ == "__main__":
    verify_chartqa()
    verify_chartqa_qa_format()
    print("\n✅ All data verifications passed")
