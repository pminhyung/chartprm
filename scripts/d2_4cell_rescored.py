"""D2 4-cell + alignment recompute using PER-SOURCE rescored outcomes.

Reads:
- data/d2_pilot/rescore_<dataset>.jsonl  (id → outcome_new)
- data/d2_pilot/{perception_v2, baseline_*_perception}.jsonl  (perception)
- data/d2_pilot/mc_results.jsonl  (for main alignment only)

Outputs:
- data/d2_pilot/4cell_rescored.json + decision_report addendum
- data/d2_pilot/alignment_rescored.json (main only)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def load_outcomes(rescore_path: Path) -> dict[str, int | None]:
    if not rescore_path.exists():
        return {}
    out = {}
    for l in open(rescore_path):
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["id"]] = r.get("outcome_new")
    return out


def safe_auc(y, x):
    if len(set(y)) < 2 or len(set(x)) < 2:
        return None
    try:
        return float(roc_auc_score(y, x))
    except Exception:
        return None


def analyze(perception_file: Path, outcome_dict: dict[str, int | None],
            drift_threshold: float = 0.5):
    perc = [json.loads(l) for l in open(perception_file) if l.strip()]
    cells = []
    valid = 0
    excluded = 0
    by_source = {"chartqa_train": [], "reachqa_train": []}
    for r in perc:
        sid = r["id"]
        outcome = outcome_dict.get(sid)
        if outcome is None or outcome == -1:
            excluded += 1
            continue
        outcome = int(outcome)
        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                    if s is not None and s.get("mean_score") is not None]
        if not p_scores:
            cell = "unknown"
            p_agg = None
        else:
            p_agg = float(np.mean(p_scores))
            low_drift = p_agg >= drift_threshold
            if outcome == 1 and low_drift: cell = "grounded_correct"
            elif outcome == 1: cell = "shortcut"
            elif low_drift: cell = "careful_flawed"
            else: cell = "hallucinated"
        cells.append({"id": sid, "p_agg": p_agg, "outcome": outcome, "cell": cell})
        valid += 1
        source = "chartqa_train" if sid.startswith("chartqa_") else "reachqa_train"
        by_source[source].append({"id": sid, "p_agg": p_agg, "outcome": outcome, "cell": cell})
    n = valid
    breakdown = {"n_samples": n, "excluded_no_outcome": excluded}
    for c in ["grounded_correct", "shortcut", "careful_flawed", "hallucinated", "unknown"]:
        breakdown[c] = sum(1 for x in cells if x["cell"] == c) / max(n, 1) * 100
    breakdown["outcome_correct_rate"] = (
        sum(1 for x in cells if x["cell"] in ("grounded_correct", "shortcut")) / max(n, 1) * 100
    )
    breakdown["drift_rate_within_correct"] = (
        sum(1 for x in cells if x["cell"] == "shortcut") /
        max(sum(1 for x in cells if x["cell"] in ("grounded_correct", "shortcut")), 1) * 100
    )
    # Per source breakdown
    by_source_break = {}
    for src, slist in by_source.items():
        if not slist:
            continue
        nn = len(slist)
        by_source_break[src] = {
            "n": nn,
            "grounded_correct": sum(1 for x in slist if x["cell"] == "grounded_correct") / nn * 100,
            "shortcut":         sum(1 for x in slist if x["cell"] == "shortcut") / nn * 100,
            "careful_flawed":   sum(1 for x in slist if x["cell"] == "careful_flawed") / nn * 100,
            "hallucinated":     sum(1 for x in slist if x["cell"] == "hallucinated") / nn * 100,
            "unknown":          sum(1 for x in slist if x["cell"] == "unknown") / nn * 100,
            "outcome_rate":     sum(1 for x in slist if x["outcome"] == 1) / nn * 100,
        }
    return breakdown, by_source_break, cells


def detect_pattern(results):
    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) < 2:
        return "INSUFFICIENT_DATA"
    shortcut_rates = [v["shortcut"] for v in valid.values()]
    grounded_rates = [v["grounded_correct"] for v in valid.values()]
    universal_shortcut = all(s >= 15 for s in shortcut_rates)
    grounded_diff = max(grounded_rates) - min(grounded_rates)
    zs = valid.get("zeroshot_4b", {}); cg = valid.get("chartgemma_12b", {})
    if zs and cg and cg.get("grounded_correct", 0) - zs.get("grounded_correct", 0) >= 10:
        return "C  (ChartGemma grounded ≥ zero-shot +10pp)"
    if universal_shortcut and grounded_diff < 10:
        return "A  (drift universal → Phase 2 GO)"
    if universal_shortcut and grounded_diff >= 10:
        return "A-strong  (drift universal AND model-dependent → GO + sub-thesis)"
    if not universal_shortcut and grounded_diff < 10:
        return "B  (drift not universal → MC-only Math-Shepherd pivot)"
    return "AMBIGUOUS"


def main():
    # Per-dataset rescore inputs
    inputs = {
        "main": (BASE / "data/d2_pilot/perception_v2.jsonl",
                 BASE / "data/d2_pilot/rescore_main.jsonl"),
        "zeroshot_4b": (BASE / "data/d2_pilot/baseline_zeroshot_4b_perception.jsonl",
                        BASE / "data/d2_pilot/rescore_zeroshot_4b.jsonl"),
        "chart_r1_7b": (BASE / "data/d2_pilot/baseline_chart_r1_perception.jsonl",
                        BASE / "data/d2_pilot/rescore_chart_r1.jsonl"),
        "chartgemma_12b": (BASE / "data/d2_pilot/baseline_chartgemma_perception.jsonl",
                           BASE / "data/d2_pilot/rescore_chartgemma.jsonl"),
    }
    results = {}
    per_source_all = {}
    cells_all = {}
    for name, (perc_p, rescore_p) in inputs.items():
        outcome_dict = load_outcomes(rescore_p)
        if not outcome_dict:
            results[name] = {"error": f"no rescore file: {rescore_p.name}"}
            continue
        b, bs, cells = analyze(perc_p, outcome_dict)
        results[name] = b
        per_source_all[name] = bs
        cells_all[name] = cells

    # Print
    print(f"\n{'Model':<18}{'N':>5}{'Grounded+Correct':>20}{'Shortcut':>12}{'CarefulFlawed':>16}{'Hallucinated':>15}{'Unknown':>10}{'Outcome%':>12}")
    print("=" * 110)
    for name, b in results.items():
        if "error" in b:
            print(f"{name:<18}  ERROR: {b['error']}")
            continue
        print(f"{name:<18}{b['n_samples']:>5}{b['grounded_correct']:>19.1f}%{b['shortcut']:>11.1f}%"
              f"{b['careful_flawed']:>15.1f}%{b['hallucinated']:>14.1f}%{b['unknown']:>9.1f}%"
              f"{b['outcome_correct_rate']:>11.1f}%")

    pattern = detect_pattern(results)
    print(f"\n=== Rescored Pattern: {pattern} ===\n")

    # Per-source detail
    print("=== Per-source breakdown ===")
    for name, bs in per_source_all.items():
        if not bs:
            continue
        for src, b in bs.items():
            print(f"  {name:<18} {src:<18} n={b['n']:>3}  grounded+correct={b['grounded_correct']:.1f}%  shortcut={b['shortcut']:.1f}%  outcome={b['outcome_rate']:.1f}%")

    # Save
    out = {"baselines": results, "by_source": per_source_all, "pattern": pattern}
    out_path = BASE / "data/d2_pilot/4cell_rescored.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    # Main alignment recompute
    print("\n=== Main alignment v2 with rescored outcomes ===")
    outcome_main = load_outcomes(inputs["main"][1])
    perc_main = [json.loads(l) for l in open(inputs["main"][0]) if l.strip()]
    mc = {r["id"]: r for r in (json.loads(l) for l in open(BASE / "data/d2_pilot/mc_results.jsonl"))}
    y, p, m = [], [], []
    for r in perc_main:
        sid = r["id"]
        o = outcome_main.get(sid)
        if o is None or o == -1:
            continue
        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                    if s is not None and s.get("mean_score") is not None]
        mc_steps = mc.get(sid, {}).get("step_outcome_scores", [])
        mc_scores = [s["mean_score"] for s in mc_steps if s is not None]
        if not p_scores or not mc_scores:
            continue
        y.append(int(o)); p.append(np.mean(p_scores)); m.append(np.mean(mc_scores))
    y, p, m = np.array(y), np.array(p), np.array(m)
    n = len(y)
    if n > 0:
        p_auc = safe_auc(y, p); m_auc = safe_auc(y, m)
        pm_r = float(pearsonr(p, m)[0]) if n >= 3 else None
        # lift
        try:
            if len(set(y)) >= 2:
                mc_clf = LogisticRegression().fit(m.reshape(-1, 1), y)
                mc_only = float(roc_auc_score(y, mc_clf.predict_proba(m.reshape(-1, 1))[:, 1]))
                X = np.column_stack([m, p])
                both = float(roc_auc_score(y, LogisticRegression().fit(X, y).predict_proba(X)[:, 1]))
                lift = both - mc_only
            else:
                mc_only = both = lift = None
        except Exception:
            mc_only = both = lift = None
        print(f"  N={n}  outcome={y.mean()*100:.1f}%")
        print(f"  Perception AUC = {p_auc:.3f}")
        print(f"  MC         AUC = {m_auc:.3f}")
        print(f"  r(p,m)         = {pm_r:+.3f}")
        if lift is not None:
            print(f"  AUC lift       = {lift:+.3f}")
        rep = {"n": n, "outcome": float(y.mean()),
               "perception_auc": p_auc, "mc_auc": m_auc, "pm_r": pm_r,
               "mc_only_auc": mc_only, "mc_plus_p_auc": both, "lift": lift}
        align_path = BASE / "data/d2_pilot/alignment_rescored.json"
        align_path.write_text(json.dumps(rep, indent=2))
        print(f"  -> {align_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
