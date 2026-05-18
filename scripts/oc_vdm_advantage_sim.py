"""OC-VDM advantage modulation simulation (Gate 2).

Reads data/d2_hardbench/reports/image_dep_12combo/*.jsonl.
Classifies each step into Pattern A/B/C/D/E, computes simulated advantage modulation.
Verifies H3: Pattern C steps get ≥1.3× up-weight, Pattern B steps get ≤0.8× down-weight.

Output: data/d2_hardbench/reports/oc_vdm_advantage_sim.json + console table.
"""
from __future__ import annotations
import json, os, statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
REPORTS_DIR = BASE / "data/d2_hardbench/reports/image_dep_12combo"
OUT = BASE / "data/d2_hardbench/reports/oc_vdm_advantage_sim.json"

LAMBDA_VD = 1.0
GROUP_BASELINE = 0.5  # proxy: GRPO group-relative baseline assumed mean


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


def compute_advantage(mc_w, vd, lambda_vd=LAMBDA_VD, baseline=GROUP_BASELINE):
    base_adv = mc_w - baseline
    if base_adv > 0:
        modulation = max(0.2, 1.0 + lambda_vd * vd)
        return base_adv, modulation, base_adv * modulation
    return base_adv, 1.0, base_adv  # penalty unchanged


def main():
    by_pattern = defaultdict(list)
    for path in sorted(REPORTS_DIR.glob("*.jsonl")):
        for line in open(path):
            try:
                s = json.loads(line)
            except Exception:
                continue
            if 'step_records' not in s:
                continue
            for sr in s['step_records']:
                mc_w = sr.get('mc_value')
                mc_wo = sr.get('mc_without_value')
                if mc_w is None or mc_wo is None:
                    continue
                vd = sr.get('visual_dep') or (mc_w - mc_wo)
                p = classify(mc_w, mc_wo)
                base_adv, mod, final_adv = compute_advantage(mc_w, vd)
                by_pattern[p].append({
                    "id": s.get('id'), "combo": s.get('combo'), "k": sr.get('k'),
                    "mc_w": mc_w, "mc_wo": mc_wo, "vd": vd,
                    "base_adv": base_adv, "modulation": mod, "final_adv": final_adv,
                })

    # Aggregate per pattern
    summary = {}
    for pattern, steps in by_pattern.items():
        if not steps:
            continue
        mods = [s["modulation"] for s in steps]
        vds = [s["vd"] for s in steps]
        summary[pattern] = {
            "n": len(steps),
            "mean_modulation": round(statistics.mean(mods), 4),
            "median_modulation": round(statistics.median(mods), 4),
            "mean_vd": round(statistics.mean(vds), 4),
            "mean_final_adv": round(statistics.mean(s["final_adv"] for s in steps), 4),
        }

    # H3 PASS/FAIL
    h3_results = {}
    if "C_image_critical" in summary:
        h3_results["C_pass"] = summary["C_image_critical"]["mean_modulation"] >= 1.3
    if "B_leakage" in summary:
        h3_results["B_pass"] = summary["B_leakage"]["mean_modulation"] <= 0.8
    if "A_hard_impossible" in summary:
        h3_results["A_pass"] = summary["A_hard_impossible"]["mean_final_adv"] >= -0.6  # mostly neg
    if "D_hard_perception" in summary:
        h3_results["D_pass"] = 1.0 <= summary["D_hard_perception"]["mean_modulation"] <= 1.2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "summary": summary,
        "h3_results": h3_results,
        "lambda_vd": LAMBDA_VD,
        "group_baseline_proxy": GROUP_BASELINE,
    }, open(OUT, "w"), indent=2, ensure_ascii=False)

    # Print
    print(f"\n{'Pattern':25s} {'n':>6} {'mean_mod':>10} {'med_mod':>10} {'mean_vd':>10} {'mean_final_adv':>15}")
    print("-" * 80)
    for p in ["A_hard_impossible","B_leakage","C_image_critical","D_hard_perception","E_other"]:
        if p in summary:
            r = summary[p]
            print(f"{p:25s} {r['n']:>6} {r['mean_modulation']:>10.3f} {r['median_modulation']:>10.3f} "
                  f"{r['mean_vd']:>+10.3f} {r['mean_final_adv']:>+15.4f}")

    print(f"\n=== H3 CHECK (lambda_vd={LAMBDA_VD}) ===")
    for k, v in h3_results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # Per-pattern sample examples
    print(f"\n=== Pattern examples (5 steps each) ===")
    for p in ["A_hard_impossible","B_leakage","C_image_critical","D_hard_perception"]:
        if p in by_pattern and by_pattern[p]:
            print(f"\n[{p}] n={len(by_pattern[p])}")
            for ex in by_pattern[p][:5]:
                print(f"  {ex['combo']}/{ex['id']} k={ex['k']}: mc_w={ex['mc_w']:.2f} mc_wo={ex['mc_wo']:.2f} "
                      f"vd={ex['vd']:+.2f} base={ex['base_adv']:+.2f} mod={ex['modulation']:.2f}× final={ex['final_adv']:+.3f}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
