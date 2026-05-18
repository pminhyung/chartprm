"""OC-VDM filter application + stratification (Gate 3).

Applies filter rules per sample, reports keep/drop distribution per combo
and per question type (where available).

Output: data/d2_hardbench/reports/oc_vdm_filter_stats.json
"""
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
REPORTS_DIR = BASE / "data/d2_hardbench/reports/image_dep_12combo"
OUT = BASE / "data/d2_hardbench/reports/oc_vdm_filter_stats.json"


def filter_sample(step_records):
    if not step_records:
        return "drop_no_steps"
    mc_w_vals = [(sr.get('mc_value') or 0) for sr in step_records]
    mc_wo_vals = [(sr.get('mc_without_value') or 0) for sr in step_records]
    mc_w_max = max(mc_w_vals)
    mc_wo_max = max(mc_wo_vals)
    if mc_w_max == 0 and mc_wo_max == 0:
        return "drop_hard_impossible"
    if mc_wo_max >= 0.5 and mc_w_max < mc_wo_max + 0.1:
        return "drop_leakage"
    if mc_w_max == 1 and all(v >= 0.8 for v in mc_w_vals):
        return "drop_trivial"
    return "keep"


def main():
    results = {}
    global_filter = Counter()
    for path in sorted(REPORTS_DIR.glob("*.jsonl")):
        combo = path.stem  # e.g., "qwen3vl_4b_chartmuseum"
        per_combo = Counter()
        per_type = defaultdict(Counter)
        n_samples = 0
        for line in open(path):
            try:
                s = json.loads(line)
            except Exception:
                continue
            if 'step_records' not in s:
                continue
            n_samples += 1
            outcome = filter_sample(s['step_records'])
            per_combo[outcome] += 1
            global_filter[outcome] += 1
            # Try to stratify by reasoning_type if present
            rt = s.get('reasoning_type') or 'unknown'
            per_type[rt][outcome] += 1

        results[combo] = {
            "n_samples": n_samples,
            "filter_counts": dict(per_combo),
            "kept_pct": round(100.0 * per_combo.get('keep', 0) / max(1, n_samples), 2),
            "drop_hard_pct": round(100.0 * per_combo.get('drop_hard_impossible', 0) / max(1, n_samples), 2),
            "drop_leakage_pct": round(100.0 * per_combo.get('drop_leakage', 0) / max(1, n_samples), 2),
            "drop_trivial_pct": round(100.0 * per_combo.get('drop_trivial', 0) / max(1, n_samples), 2),
            "per_reasoning_type": {k: dict(v) for k, v in per_type.items()},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_combo": results, "global": dict(global_filter)},
               open(OUT, "w"), indent=2, ensure_ascii=False)

    # Console
    print(f"\n{'Combo':45s} {'n':>5} {'kept%':>7} {'hard%':>7} {'leak%':>7} {'triv%':>7}")
    print("-" * 90)
    for combo, r in results.items():
        print(f"{combo:45s} {r['n_samples']:>5} {r['kept_pct']:>6.1f}% "
              f"{r['drop_hard_pct']:>6.1f}% {r['drop_leakage_pct']:>6.1f}% {r['drop_trivial_pct']:>6.1f}%")

    print(f"\n=== Global filter counts ===")
    for k, v in sorted(global_filter.items()):
        print(f"  {k}: {v}")

    n_total = sum(global_filter.values())
    if n_total > 0:
        kept_total = global_filter.get('keep', 0)
        print(f"\nGlobal kept: {kept_total}/{n_total} = {100.0*kept_total/n_total:.1f}%")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
