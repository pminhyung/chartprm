"""OC-VDM cross-bench analysis (Gate 1).

Reads data/d2_hardbench/reports/image_dep_12combo/<model>_<bench>.jsonl × 12 combo.
Computes H1 (vd distribution), H2 (Pattern A/B/C/D distribution) per combo and globally.

Output: data/d2_hardbench/reports/oc_vdm_cross_bench.json + console decision table.

Pattern classification per STEP (not per sample):
  Pattern A (hard-impossible):  mc_with=0 AND mc_without=0
  Pattern B (text leakage):     mc_without ≥ 0.5 AND vd < 0
  Pattern C (image-critical):   mc_with ≥ 0.5 AND vd > 0.3
  Pattern D (hard-perception):  0 < mc_with < 0.3 AND mc_without ≤ 0.1
  Pattern E (other):            else (informative but uncategorized)
"""
from __future__ import annotations
import json, os, statistics
from collections import Counter
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
REPORTS = BASE / "data/d2_hardbench/reports/image_dep_12combo"
OUT = BASE / "data/d2_hardbench/reports/oc_vdm_cross_bench.json"

MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]


def classify(mc_w, mc_wo):
    if mc_w is None or mc_wo is None:
        return "_invalid"
    vd = mc_w - mc_wo
    if mc_w == 0 and mc_wo == 0:
        return "A_hard_impossible"
    if mc_wo >= 0.5 and vd < 0:
        return "B_leakage"
    if mc_w >= 0.5 and vd > 0.3:
        return "C_image_critical"
    if 0 < mc_w < 0.3 and mc_wo <= 0.1:
        return "D_hard_perception"
    return "E_other"


def filter_sample(step_records):
    if not step_records:
        return "drop_no_steps"
    mc_w_max = max((sr.get('mc_value') or 0) for sr in step_records)
    mc_wo_max = max((sr.get('mc_without_value') or 0) for sr in step_records)
    if mc_w_max == 0 and mc_wo_max == 0:
        return "drop_hard_impossible"
    if mc_wo_max >= 0.5 and mc_w_max < mc_wo_max + 0.1:
        return "drop_leakage"
    if mc_w_max == 1 and all((sr.get('mc_value') or 0) >= 0.8 for sr in step_records):
        return "drop_trivial"
    return "keep"


def analyze_combo(path: Path):
    samples = []
    for line in open(path):
        try:
            samples.append(json.loads(line))
        except Exception:
            pass
    valid_samples = [s for s in samples if 'step_records' in s and s['step_records']]
    err_samples = [s for s in samples if 'error' in s]

    # Step-level metrics
    vds = []
    pattern_counts = Counter()
    mc_w_vals = []
    mc_wo_vals = []
    for s in valid_samples:
        for sr in s['step_records']:
            mc_w = sr.get('mc_value')
            mc_wo = sr.get('mc_without_value')
            vd = sr.get('visual_dep')
            if vd is not None:
                vds.append(vd)
            if mc_w is not None:
                mc_w_vals.append(mc_w)
            if mc_wo is not None:
                mc_wo_vals.append(mc_wo)
            pattern_counts[classify(mc_w, mc_wo)] += 1

    # Sample-level filter
    filter_counts = Counter()
    for s in valid_samples:
        filter_counts[filter_sample(s['step_records'])] += 1

    n_valid = len(valid_samples)
    n_steps = sum(len(s['step_records']) for s in valid_samples)
    vd_std = statistics.pstdev(vds) if len(vds) >= 2 else 0.0
    vd_informative_pct = 100.0 * sum(1 for v in vds if abs(v) > 0.3) / max(1, len(vds))
    vd_mean = statistics.mean(vds) if vds else 0.0

    return {
        "n_samples_attempted": len(samples),
        "n_samples_valid": n_valid,
        "n_samples_err": len(err_samples),
        "err_reasons": Counter(s.get('error','?') for s in err_samples),
        "n_steps_total": n_steps,
        "avg_steps_per_sample": n_steps / max(1, n_valid),
        "vd_mean": round(vd_mean, 4),
        "vd_std": round(vd_std, 4),
        "vd_informative_pct": round(vd_informative_pct, 2),
        "vd_negative_pct": round(100.0 * sum(1 for v in vds if v < -0.1) / max(1, len(vds)), 2),
        "vd_positive_pct": round(100.0 * sum(1 for v in vds if v > 0.1) / max(1, len(vds)), 2),
        "mc_w_mean": round(statistics.mean(mc_w_vals), 4) if mc_w_vals else None,
        "mc_wo_mean": round(statistics.mean(mc_wo_vals), 4) if mc_wo_vals else None,
        "pattern_counts": dict(pattern_counts),
        "pattern_pct": {k: round(100.0*v/max(1,sum(pattern_counts.values())), 2)
                         for k,v in pattern_counts.items()},
        "filter_counts": dict(filter_counts),
        "kept_pct": round(100.0 * filter_counts.get('keep',0) / max(1, n_valid), 2),
    }


def main():
    results = {}
    h1_pass = []
    h2_pass = []
    h4_pass = []
    for bench in BENCHES:
        for model in MODELS:
            combo = f"{model}__{bench}"
            path = REPORTS / f"{model}_{bench}.jsonl"
            if not path.exists():
                results[combo] = {"status": "missing", "path": str(path)}
                continue
            r = analyze_combo(path)
            results[combo] = r
            # Hypothesis checks
            if r["vd_std"] >= 0.15 and r["vd_informative_pct"] >= 20.0:
                h1_pass.append(combo)
            pat = r["pattern_pct"]
            if (pat.get("A_hard_impossible", 0) < 50.0
                and pat.get("C_image_critical", 0) >= 15.0):
                h2_pass.append(combo)
            if r["kept_pct"] >= 50.0:
                h4_pass.append(combo)

    summary = {
        "n_combo_measured": sum(1 for r in results.values() if r.get("n_samples_valid", 0) > 0),
        "h1_pass_combos": h1_pass,
        "h2_pass_combos": h2_pass,
        "h4_pass_combos": h4_pass,
        "h1_pass_count": len(h1_pass),
        "h2_pass_count": len(h2_pass),
        "h4_pass_count": len(h4_pass),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_combo": results, "summary": summary}, open(OUT, "w"),
               indent=2, ensure_ascii=False)

    # Console table
    print(f"\n{'Combo':45s} {'n_valid':>8} {'steps':>6} {'vd_std':>7} {'vd_info%':>9} "
          f"{'PatA%':>6} {'PatB%':>6} {'PatC%':>6} {'PatD%':>6} {'PatE%':>6} {'Kept%':>6}")
    print("-" * 130)
    for combo, r in results.items():
        if r.get("status") == "missing":
            print(f"{combo:45s} (missing: {r['path']})")
            continue
        if r["n_samples_valid"] == 0:
            print(f"{combo:45s} 0 valid samples ({r.get('err_reasons',{})})")
            continue
        pat = r["pattern_pct"]
        print(f"{combo:45s} {r['n_samples_valid']:>8} {r['n_steps_total']:>6} "
              f"{r['vd_std']:>7.3f} {r['vd_informative_pct']:>8.1f}% "
              f"{pat.get('A_hard_impossible',0):>5.1f}% "
              f"{pat.get('B_leakage',0):>5.1f}% "
              f"{pat.get('C_image_critical',0):>5.1f}% "
              f"{pat.get('D_hard_perception',0):>5.1f}% "
              f"{pat.get('E_other',0):>5.1f}% "
              f"{r['kept_pct']:>5.1f}%")

    print(f"\n=== GATE 1 SUMMARY ===")
    print(f"H1 (vd_std≥0.15 & vd_informative≥20%):  {summary['h1_pass_count']}/12 combos PASS")
    print(f"  PASS combos: {h1_pass}")
    print(f"H2 (PatA<50% & PatC≥15%):              {summary['h2_pass_count']}/12 combos PASS")
    print(f"  PASS combos: {h2_pass}")
    print(f"H4 (kept≥50%):                          {summary['h4_pass_count']}/12 combos PASS")
    print(f"  PASS combos: {h4_pass}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
