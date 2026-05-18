"""Hard-bench 4-cell + alignment per (baseline, bench).

Reads:
- data/d2_hardbench/perception/<model>/<bench>.jsonl  (perception scores)
- data/d2_hardbench/outcome/<model>_<bench>.jsonl     (per-sample outcome 0/1)

Outputs:
- data/d2_hardbench/reports/4cell_per_bench.json
- console table per bench: model × cell distribution
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
HB = BASE / "data/d2_hardbench"

MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]
DRIFT_THRESHOLD = 0.5


def load_outcome(p: Path) -> dict[str, int | None]:
    if not p.exists(): return {}
    out = {}
    for l in open(p):
        if not l.strip(): continue
        r = json.loads(l)
        out[r["id"]] = r.get("outcome")
    return out


def analyze(perception_p: Path, outcome_p: Path):
    if not perception_p.exists() or not outcome_p.exists():
        return None
    perc = [json.loads(l) for l in open(perception_p) if l.strip()]
    outcomes = load_outcome(outcome_p)
    cells = []
    valid = 0
    for r in perc:
        sid = r["id"]
        o = outcomes.get(sid)
        if o is None or o == -1:
            cells.append({"id": sid, "cell": "unknown", "p_agg": None, "o": None}); continue
        o = int(o)
        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                    if s is not None and s.get("mean_score") is not None]
        if not p_scores:
            cells.append({"id": sid, "cell": "unknown", "p_agg": None, "o": o}); continue
        p = float(np.mean(p_scores))
        low_drift = p >= DRIFT_THRESHOLD
        if o == 1 and low_drift: cell = "grounded_correct"
        elif o == 1: cell = "shortcut"
        elif low_drift: cell = "careful_flawed"
        else: cell = "hallucinated"
        cells.append({"id": sid, "cell": cell, "p_agg": p, "o": o})
        valid += 1
    n = len(cells)
    breakdown = {"n_samples": n, "n_valid": valid}
    for c in ["grounded_correct", "shortcut", "careful_flawed", "hallucinated", "unknown"]:
        breakdown[c] = sum(1 for x in cells if x["cell"] == c) / max(n, 1) * 100
    outcome_pct = sum(1 for x in cells if x["o"] == 1) / max(n, 1) * 100
    breakdown["outcome_pct"] = outcome_pct

    # Alignment metrics (perception vs outcome)
    pcells_valid = [c for c in cells if c["p_agg"] is not None and c["o"] is not None]
    if len(pcells_valid) >= 10:
        y = np.array([c["o"] for c in pcells_valid])
        p = np.array([c["p_agg"] for c in pcells_valid])
        try:
            auc = float(roc_auc_score(y, p)) if len(set(y)) >= 2 and len(set(p)) >= 2 else None
        except Exception:
            auc = None
        try:
            r_pear, _ = pearsonr(p, y)
        except Exception:
            r_pear = None
        breakdown["perception_auc_vs_outcome"] = auc
        breakdown["perception_r_vs_outcome"] = float(r_pear) if r_pear is not None else None
        breakdown["mean_perception"] = float(p.mean())
    return breakdown


def main():
    results = {}
    print(f"\n{'bench':<22}{'model':<22}{'N':>4}{'GC%':>7}{'SC%':>6}{'CF%':>7}{'HL%':>6}{'UK%':>6}{'Out%':>7}{'pAUC':>7}{'r(p,o)':>9}{'meanP':>8}")
    print("=" * 120)
    for bench in BENCHES:
        for model in MODELS:
            perc_p = HB / f"perception/{model}/{bench}.jsonl"
            outc_p = HB / f"outcome/outcome_{model}_{bench}.jsonl"
            b = analyze(perc_p, outc_p)
            if b is None:
                print(f"{bench:<22}{model:<22}  ERROR: missing files")
                continue
            results[f"{bench}__{model}"] = b
            auc = b.get("perception_auc_vs_outcome")
            r_po = b.get("perception_r_vs_outcome")
            mp = b.get("mean_perception")
            print(f"{bench:<22}{model:<22}{b['n_samples']:>4}{b['grounded_correct']:>6.1f}%{b['shortcut']:>5.1f}%"
                  f"{b['careful_flawed']:>6.1f}%{b['hallucinated']:>5.1f}%{b['unknown']:>5.1f}%{b['outcome_pct']:>6.1f}%"
                  f"{(auc if auc is not None else 0):>7.3f}{(r_po if r_po is not None else 0):>+9.3f}{(mp if mp is not None else 0):>8.3f}")
    out_path = HB / "reports/4cell_per_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
