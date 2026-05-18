"""Recompute 4-cell + pAUC with REJUDGE outcomes (extended answer extraction)
for 2-bench × 4-baseline = 8 combos. Compares orig vs rejudge.

Outputs:
- data/d2_hardbench/reports/4cell_per_bench_rejudge.json
- console diff table
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
HB = BASE / "data/d2_hardbench"
MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]
DRIFT = 0.5


def load_outcomes_orig(p):
    return {json.loads(l)["id"]: json.loads(l).get("outcome") for l in open(p) if l.strip()}


def load_outcomes_rej(p, key="outcome_ext"):
    return {json.loads(l)["id"]: json.loads(l).get(key) for l in open(p) if l.strip()}


def cell_of(p_agg, o):
    if o is None or o == -1: return "unknown"
    low_drift = p_agg >= DRIFT
    if o == 1 and low_drift: return "grounded_correct"
    if o == 1: return "shortcut"
    if low_drift: return "careful_flawed"
    return "hallucinated"


def analyze(perc_p, outcomes):
    perc = [json.loads(l) for l in open(perc_p) if l.strip()]
    cells = []
    for r in perc:
        sid = r["id"]
        o = outcomes.get(sid)
        scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                  if s and s.get("mean_score") is not None]
        if not scores:
            cells.append((sid, None, o, "unknown")); continue
        p = float(np.mean(scores))
        c = cell_of(p, o)
        cells.append((sid, p, o, c))
    n = len(cells)
    br = {"n_samples": n}
    for c in ["grounded_correct","shortcut","careful_flawed","hallucinated","unknown"]:
        br[c] = sum(1 for x in cells if x[3]==c)/max(n,1)*100
    out_pct = sum(1 for x in cells if x[2]==1)/max(n,1)*100
    br["outcome_pct"] = out_pct
    pairs = [(x[1], x[2]) for x in cells if x[1] is not None and x[2] is not None]
    if len(pairs) >= 10:
        p = np.array([a for a,_ in pairs]); y = np.array([b for _,b in pairs])
        try: br["auc"] = float(roc_auc_score(y,p)) if len(set(y))>=2 else None
        except Exception: br["auc"] = None
        try: r,_ = pearsonr(p,y); br["r"] = float(r)
        except Exception: br["r"] = None
        br["meanP"] = float(p.mean())
    return br


def main():
    out = {}
    print(f"{'combo':<48}{'GC%':>7}{'CF%':>7}{'HL%':>7}{'UK%':>7}{'Out%':>7}{'pAUC':>7}{'r(p,o)':>9}")
    print("=" * 100)
    for bench in BENCHES:
        for model in MODELS:
            perc_p = HB / f"perception/{model}/{bench}.jsonl"
            orig_p = HB / f"outcome/outcome_{model}_{bench}.jsonl"
            rej_p = HB / f"outcome_rejudge/{model}_{bench}.jsonl"
            if not (perc_p.exists() and orig_p.exists() and rej_p.exists()): continue
            orig_o = load_outcomes_orig(orig_p)
            rej_o = load_outcomes_rej(rej_p, "outcome_ext")
            # Fill rej missing with orig (no change)
            for sid in orig_o:
                if sid not in rej_o or rej_o[sid] is None:
                    rej_o[sid] = orig_o[sid]
            orig_br = analyze(perc_p, orig_o)
            rej_br = analyze(perc_p, rej_o)
            out[f"{bench}__{model}"] = {"orig": orig_br, "rejudge": rej_br}
            for tag, br in [("ORIG ", orig_br), ("REJDG", rej_br)]:
                print(f"{tag}{bench:<22}{model:<22}{br['grounded_correct']:>6.1f}%{br['careful_flawed']:>6.1f}%"
                      f"{br['hallucinated']:>6.1f}%{br['unknown']:>6.1f}%{br['outcome_pct']:>6.1f}%"
                      f"{(br.get('auc') or 0):>7.3f}{(br.get('r') or 0):>+9.3f}")
    op = HB / "reports/4cell_per_bench_rejudge.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
