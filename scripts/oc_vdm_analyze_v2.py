"""OC-VDM analyzer v2 — applies M1 (terminal skip) + M3 (tightened PatB).

Reads same input as oc_vdm_analyze.py (image_dep_12combo/*.jsonl + optional no_cap).
Outputs side-by-side comparison: ORIGINAL vs v2 (M1+M3 only — no re-measurement).
"""
from __future__ import annotations
import json, os, statistics, argparse
from collections import Counter
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]


def classify_orig(mc_w, mc_wo):
    if mc_w is None or mc_wo is None: return None
    vd = mc_w - mc_wo
    if mc_w == 0 and mc_wo == 0: return "A"
    if mc_wo >= 0.5 and vd < 0: return "B"
    if mc_w >= 0.5 and vd > 0.3: return "C"
    if 0 < mc_w < 0.3 and mc_wo <= 0.1: return "D"
    return "E"


def classify_v2(mc_w, mc_wo):
    """M3: tightened Pattern B — true leakage only."""
    if mc_w is None or mc_wo is None: return None
    vd = mc_w - mc_wo
    if mc_w == 0 and mc_wo == 0: return "A"
    if mc_wo >= 0.7 and mc_w <= 0.3 and vd <= -0.4: return "B"
    if mc_w >= 0.5 and vd > 0.3: return "C"
    if 0 < mc_w < 0.3 and mc_wo <= 0.1: return "D"
    return "E"


def filter_sample(srs):
    mc_w_max = max((sr.get('mc_value') or 0) for sr in srs)
    mc_wo_max = max((sr.get('mc_without_value') or 0) for sr in srs)
    if mc_w_max == 0 and mc_wo_max == 0: return 'drop_hard'
    if mc_wo_max >= 0.5 and mc_w_max < mc_wo_max + 0.1: return 'drop_leakage'
    if mc_w_max == 1 and all((sr.get('mc_value') or 0) >= 0.8 for sr in srs): return 'drop_trivial'
    return 'keep'


def analyze(path, skip_terminal=False, use_v2_class=False):
    pat = Counter()
    vds = []; mc_w_vals = []; mc_wo_vals = []
    filter_counts = Counter()
    n_sample = 0; n_terminal_skipped = 0
    for line in open(path):
        try: s = json.loads(line)
        except: continue
        if 'step_records' not in s: continue
        n_sample += 1
        srs = s['step_records']
        nsr = len(srs)
        filter_counts[filter_sample(srs)] += 1
        for sr in srs:
            mc_w = sr.get('mc_value')
            mc_wo = sr.get('mc_without_value')
            if mc_w is None or mc_wo is None: continue
            # M1: skip terminal step (last index)
            if skip_terminal and sr['k'] == nsr - 1:
                n_terminal_skipped += 1
                continue
            classifier = classify_v2 if use_v2_class else classify_orig
            pat[classifier(mc_w, mc_wo)] += 1
            vds.append(mc_w - mc_wo)
            mc_w_vals.append(mc_w)
            mc_wo_vals.append(mc_wo)
    n_step = sum(pat.values())
    return {
        "n_sample": n_sample,
        "n_step": n_step,
        "n_terminal_skipped": n_terminal_skipped,
        "mc_w_mean": (sum(mc_w_vals) / max(1, len(mc_w_vals))) if mc_w_vals else 0,
        "vd_std": statistics.pstdev(vds) if len(vds) >= 2 else 0,
        "vd_info_pct": 100.0 * sum(1 for v in vds if abs(v) > 0.3) / max(1, len(vds)),
        "pat_pct": {k: 100.0 * pat[k] / max(1, n_step) for k in "ABCDE"},
        "informative_pct": 100.0 * (pat["B"] + pat["C"] + pat["D"]) / max(1, n_step),
        "kept_pct": 100.0 * filter_counts.get('keep', 0) / max(1, n_sample),
        "filter_counts": dict(filter_counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="image_dep_12combo",
                    help="image_dep_12combo or image_dep_12combo_no_cap")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src_dir = BASE / f"data/d2_hardbench/reports/{args.source}"
    out_path = BASE / f"data/d2_hardbench/reports/oc_vdm_analysis_v2_{args.source.split('_')[-1] if 'no_cap' in args.source else 'cap'}.json"

    rows = []
    print(f"\n{'Combo':45s} | {'OLD':>30s} | {'v2 (M1+M3)':>30s}")
    print(f"{'':45s} | {'info%':>6s} {'PatA%':>6s} {'PatB%':>6s} {'PatC%':>6s} | {'info%':>6s} {'PatA%':>6s} {'PatB%':>6s} {'PatC%':>6s} {'ΔPatC':>6s}")
    print("-" * 150)
    for bench in BENCHES:
        for model in MODELS:
            fn = f"{model}_{bench}.jsonl"
            p = src_dir / fn
            if not p.exists():
                continue
            r_old = analyze(p, skip_terminal=False, use_v2_class=False)
            r_new = analyze(p, skip_terminal=True, use_v2_class=True)
            delta_c = r_new["pat_pct"].get("C", 0) - r_old["pat_pct"].get("C", 0)
            print(f"{model+'__'+bench:45s} | {r_old['informative_pct']:5.1f}% {r_old['pat_pct'].get('A',0):5.1f}% {r_old['pat_pct'].get('B',0):5.1f}% {r_old['pat_pct'].get('C',0):5.1f}% | {r_new['informative_pct']:5.1f}% {r_new['pat_pct'].get('A',0):5.1f}% {r_new['pat_pct'].get('B',0):5.1f}% {r_new['pat_pct'].get('C',0):5.1f}% {delta_c:+5.1f}%")
            rows.append({"combo": f"{model}__{bench}", "old": r_old, "v2": r_new, "delta_PatC": delta_c})

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
    else:
        json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
