"""D2 MC validation — guide §3.4 five auto cross-checks. No human label.

Input:  data/d2_pilot/mc_results.jsonl
Output: data/d2_pilot/mc_validate.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN = BASE / "data/d2_pilot/mc_results.jsonl"
OUT = BASE / "data/d2_pilot/mc_validate.json"


def main():
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    print(f"[mc-validate] input={len(recs)}")

    # Trajectory correctness proxy: mean of step scores ≥ 0.5 → "correct trajectory"
    # (per guide §3.4: use last step proxy. but last-step's mean reflects the WHOLE chain.)
    correct_means = []
    wrong_means = []
    correct_slopes = []
    wrong_slopes = []

    for r in recs:
        scores = [s["mean_score"] for s in r["step_outcome_scores"]]
        if not scores:
            continue
        last_correct = scores[-1] >= 0.5
        avg = float(np.mean(scores))
        if last_correct:
            correct_means.append(avg)
        else:
            wrong_means.append(avg)
        if len(scores) >= 3:
            idx = list(range(len(scores)))
            slope = float(pearsonr(idx, scores)[0]) if np.std(scores) > 1e-9 else 0.0
            (correct_slopes if last_correct else wrong_slopes).append(slope)

    gap = (np.mean(correct_means) if correct_means else 0) - (np.mean(wrong_means) if wrong_means else 0)
    slope_c = float(np.mean(correct_slopes)) if correct_slopes else 0.0
    slope_w = float(np.mean(wrong_slopes)) if wrong_slopes else 0.0

    all_scores = [s["mean_score"] for r in recs for s in r["step_outcome_scores"]]
    n_zero = sum(1 for s in all_scores if s == 0.0)
    n_one = sum(1 for s in all_scores if s == 1.0)
    n_mid = len(all_scores) - n_zero - n_one
    total = max(len(all_scores), 1)
    p_bimodal = (n_zero + n_one) / total
    p_mid = n_mid / total

    print(f"[MC vs Outcome] correct_traj_avg={np.mean(correct_means):.3f} wrong_traj_avg={np.mean(wrong_means):.3f} gap={gap:.3f}")
    print(f"[Trend] correct_slope={slope_c:.3f}  wrong_slope={slope_w:.3f}")
    print(f"[Score dist] zero={n_zero/total*100:.1f}%  one={n_one/total*100:.1f}%  mid={p_mid*100:.1f}%")

    checks = {
        "mc_outcome_gap": bool(gap >= 0.20),
        "correct_traj_trend_positive": bool(slope_c > 0),
        "wrong_traj_trend_nonpositive": bool(slope_w <= 0.05),
        "bimodal_signal_present": bool(p_bimodal >= 0.30),
        "continuous_signal_present": bool(p_mid >= 0.15),
    }
    passed = sum(checks.values())
    print(f"\n[MC validation] {passed}/5 pass")
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")

    report = {
        "n_records": len(recs),
        "correct_traj_avg": float(np.mean(correct_means)) if correct_means else None,
        "wrong_traj_avg": float(np.mean(wrong_means)) if wrong_means else None,
        "gap": float(gap),
        "correct_slope": slope_c,
        "wrong_slope": slope_w,
        "bimodal_rate": float(p_bimodal),
        "mid_rate": float(p_mid),
        "checks": checks,
        "passed": passed,
        "verdict": "PASS" if passed >= 3 else "FAIL",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[mc-validate] -> {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
