"""Build teacher-distillation source dataset.

Merges v9 SFT + v8 sft_30k + block_c_api_qa_v2 into a clean (image, question, gold)
corpus for teacher distillation. Applies source filter (reject low-signal rule
templates), resolves image paths to absolute, and deduplicates.

Usage:
    python scripts/build_distill_source.py \
        --v9 data/sft_v9.jsonl \
        --v8 data/sft_30k.jsonl \
        --api data/charts_v9/block_c_api_qa_v2.jsonl \
        --output data/distill_source.jsonl
"""
import argparse
import json
import os
from collections import Counter


# Keep only high-signal sources. Rule templates (block_a/b/c/d/f) produce
# short bimodal reasoning and are replaced by teacher output anyway — no need
# to waste teacher calls on them.
KEEP_SOURCES = {
    # sft_v9 sources
    "owid",
    "synthetic",
    "worldbank",
    "plotly_complex",
    "scientific_ext",
    # sft_30k (v8) sources
    "chartqa_train",
    "scientific",
    "kaggle_like",
    "additional",
}

# block_c_api_qa_v2 source tag
BLOCK_C_API_SOURCE = "scientific_ext"  # the file stores source="scientific_ext"


def resolve_image_path(path: str, project_root: str) -> str:
    """Return absolute image path if the file exists, else ''."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path if os.path.exists(path) else ""
    candidate = os.path.join(project_root, path)
    return candidate if os.path.exists(candidate) else ""


def dedup_key(sample: dict) -> str:
    """Dedup by (image_path, question[:80])."""
    img = sample.get("image_path", "")
    q = (sample.get("question") or "")[:80]
    return f"{img}|{q}"


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", default="data/sft_v9.jsonl")
    ap.add_argument("--v8", default="data/sft_30k.jsonl")
    ap.add_argument("--api", default="data/charts_v9/block_c_api_qa_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--project_root", default="/ex_disk2/mhpark/poc/chartvr")
    args = ap.parse_args()

    project_root = args.project_root

    seen = set()
    kept: list[dict] = []
    stats = Counter()

    def ingest(rows: list[dict], origin: str):
        for r in rows:
            stats[f"{origin}_total"] += 1
            src = r.get("source", "")
            if src not in KEEP_SOURCES:
                stats[f"{origin}_rejected_source"] += 1
                continue
            img = resolve_image_path(r.get("image_path", ""), project_root)
            if not img:
                stats[f"{origin}_missing_image"] += 1
                continue
            q = (r.get("question") or "").strip()
            ans = r.get("answer")
            if not q or ans is None or str(ans).strip() == "":
                stats[f"{origin}_missing_qa"] += 1
                continue

            # Preserve reasoning_steps / reasoning if present — they're used as
            # training assistant content when teacher-distilled assistant_text
            # is unavailable (train_sft.py legacy fallback path).
            reasoning = r.get("reasoning_steps", r.get("reasoning", ""))
            rec = {
                "question": q,
                "answer": ans,
                "answer_type": r.get("answer_type", ""),
                "reasoning_steps": reasoning,  # may be list[str] or str or ""
                "image_path": img,
                "csv_path": r.get("csv_path", ""),
                "source": src,
                "chart_slug": r.get("chart_slug", ""),
                "difficulty": r.get("difficulty", ""),
                "origin": origin,  # provenance for audit
            }
            key = dedup_key(rec)
            if key in seen:
                stats[f"{origin}_duplicate"] += 1
                continue
            seen.add(key)
            kept.append(rec)
            stats[f"{origin}_kept"] += 1
            stats[f"src_{src}"] += 1

    ingest(load_jsonl(args.v9), "v9")
    ingest(load_jsonl(args.v8), "v8")

    # block_c_api_qa_v2 already has source=scientific_ext but we include it
    # explicitly to guarantee inclusion (it may overlap v9 → dedup handles it).
    ingest(load_jsonl(args.api), "api")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print(f"\nWrote {len(kept)} samples → {args.output}")
    print("\nStats:")
    for k in sorted(stats.keys()):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
