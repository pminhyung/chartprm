#!/usr/bin/env python
"""Build block_c_eligible_v2 JSONs split by sub-prompt routing.

Merges scientific_ext (3500) + scientific v2 (3000) = 6500 unique charts,
then routes each chart to one of 4 sub-prompts based on chart_type structure.

Routing rules (matches dispatcher capabilities):
  - Entity-rich charts (Method/panel/variable names) → sci_ranking
  - Single continuous x-y charts → sci_numeric or sci_trend
  - Multi-metric comparison charts → sci_compare

Output:
  data/charts_v9/block_c_eligible_v2_ranking.json
  data/charts_v9/block_c_eligible_v2_numeric.json
  data/charts_v9/block_c_eligible_v2_trend.json
  data/charts_v9/block_c_eligible_v2_compare.json

Usage:
  python scripts/build_block_c_eligible_v2.py \
      --ext data/charts_v9/scientific_ext/manifest.jsonl \
      --v2 data/charts_v2/scientific/manifest.jsonl \
      --outdir data/charts_v9
"""
import argparse
import json
import os
import random
from collections import Counter


# chart_type → sub-prompt routing.
# Goal: match Qwen-chart ranking/numeric/trend/compare distribution and respect
# chart schema constraints (e.g., 2-col numeric charts can't produce ranking).
ROUTE = {
    # v9 scientific_ext (8 types, 3500 charts)
    "scatter_regression":  "numeric",   # continuous (x, y, Fitted_y) — value lookup, residual
    "subplot_panel":       "ranking",   # panel names are entities (4+ panels)
    "heatmap_corr":        "ranking",   # variable pair names as entities
    "errorbar_comparison": "ranking",   # Method names as entities
    "logscale_plot":       "trend",     # continuous y shape over x
    "dual_axis":           "compare",   # two metrics naturally invite comparison
    "confidence_band":     "ranking",   # Method names (when multi-method)
    "violin_comparison":   "ranking",   # Method names as entities

    # v2 scientific (4 types, 3000 charts, all 2-col numeric)
    "bar_scientific":      "numeric",
    "line":                "trend",
    "scatter":             "numeric",
    "scatter_line":        "trend",
}

# For 2-col v2 charts, we want some compare questions too (difference between x=a and x=b)
# Randomly re-route a fraction of numeric/trend v2 charts to compare.
V2_COMPARE_FRAC = 0.25


def normalize_chart(c):
    """Extract the minimal fields generate_qa.py needs."""
    return {
        "slug": c.get("slug", ""),
        "csv_path": c.get("csv_path", ""),
        "image_path": c.get("image_path", ""),
        "source": c.get("source", ""),
        "chart_type": c.get("chart_type", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Build block_c_eligible_v2 sub-prompt splits")
    parser.add_argument("--ext", default="data/charts_v9/scientific_ext/manifest.jsonl")
    parser.add_argument("--v2",  default="data/charts_v2/scientific/manifest.jsonl")
    parser.add_argument("--outdir", default="data/charts_v9")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load manifests
    charts = []
    for mpath in [args.ext, args.v2]:
        if not os.path.exists(mpath):
            print(f"  MISSING: {mpath}")
            continue
        with open(mpath) as f:
            for line in f:
                try:
                    charts.append(json.loads(line))
                except Exception:
                    pass
        print(f"  Loaded {mpath}: cumulative {len(charts)}")

    # Deduplicate by slug
    seen_slugs = set()
    dedup = []
    for c in charts:
        s = c.get("slug", "")
        if not s or s in seen_slugs:
            continue
        seen_slugs.add(s)
        dedup.append(c)
    print(f"  After dedup: {len(dedup)} unique charts")

    # Verify csv_path exists
    missing_csv = 0
    valid = []
    for c in dedup:
        if os.path.exists(c.get("csv_path", "")):
            valid.append(c)
        else:
            missing_csv += 1
    if missing_csv:
        print(f"  Dropped {missing_csv} charts with missing CSV")
    print(f"  Valid charts: {len(valid)}")

    # Route each chart to a sub-prompt
    splits = {"ranking": [], "numeric": [], "trend": [], "compare": []}
    unknown_types = Counter()

    for c in valid:
        ct = c.get("chart_type", "")
        route = ROUTE.get(ct)
        if route is None:
            unknown_types[ct] += 1
            continue

        # v2 re-routing: randomly promote some numeric/trend to compare for variety
        if ct in ("bar_scientific", "line", "scatter", "scatter_line"):
            if random.random() < V2_COMPARE_FRAC:
                route = "compare"

        splits[route].append(normalize_chart(c))

    if unknown_types:
        print(f"  WARNING: unknown chart types (skipped): {dict(unknown_types)}")

    # Shuffle within each split for balanced chart_type mixing
    for k in splits:
        random.shuffle(splits[k])

    # Write outputs
    total = 0
    for k, lst in splits.items():
        outpath = os.path.join(args.outdir, f"block_c_eligible_v2_{k}.json")
        with open(outpath, "w") as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
        print(f"  → {outpath}: {len(lst)} charts")
        total += len(lst)

    print(f"\n  Total across splits: {total}")

    # Print chart_type distribution per split
    print("\n  Chart type distribution per split:")
    for k, lst in splits.items():
        ct_counts = Counter(c["chart_type"] for c in lst)
        print(f"    {k}: {dict(ct_counts)}")


if __name__ == "__main__":
    main()
