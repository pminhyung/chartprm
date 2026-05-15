"""D2 verifier alignment with outcome — guide §3.7 4-metric framework.

Input:  data/d2_pilot/perception_results.jsonl
Output: data/d2_pilot/alignment.json
        docs/d2_report.md (decision summary)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartvr.extraction import extract_answer, relaxed_accuracy  # noqa: E402

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
PERCEPTION_JSONL = BASE / "data/d2_pilot/perception_results.jsonl"
MC_JSONL = BASE / "data/d2_pilot/mc_results.jsonl"
TRACES_SEG = BASE / "data/d1_pilot/segmented_v3.jsonl"  # original content/reasoning_content source
OUT_JSON = BASE / "data/d2_pilot/alignment.json"
OUT_REPORT = BASE / "docs/d2_report.md"


def extract_final_from_content(content: str, reasoning: str = "") -> str:
    if content and content.strip():
        return extract_answer(content) or content.strip()
    # Fallback: post-think portion of reasoning is what's between last \n\n and end
    return extract_answer(reasoning) or ""


def main():
    recs_p = [json.loads(l) for l in open(PERCEPTION_JSONL) if l.strip()]

    # mc_rollout/perception pipelines dropped the original trace content fields.
    # Recover from segmented_v3.jsonl (= the source of truth for the original trace).
    traces_by_id = {r["id"]: r for r in (json.loads(l) for l in open(TRACES_SEG))}

    per_sample = []
    for r in recs_p:
        # 1. Outcome (final answer of the original trace vs gold) — pull from segmented_v3
        src = traces_by_id.get(r["id"], {})
        final = extract_final_from_content(src.get("content", ""), src.get("reasoning_content", ""))
        outcome = int(relaxed_accuracy(final, r["gold_answer"]))

        # 2. Perception aggregate
        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", []) if s is not None]
        if not p_scores:
            continue
        perception_agg = float(np.mean(p_scores))

        # 3. MC aggregate
        mc_scores = [s["mean_score"] for s in r.get("step_outcome_scores", [])]
        if not mc_scores:
            continue
        mc_agg = float(np.mean(mc_scores))

        per_sample.append({
            "id": r["id"],
            "outcome": outcome,
            "perception_agg": perception_agg,
            "mc_agg": mc_agg,
        })

    if not per_sample:
        print("no samples with both perception and mc data")
        return 1

    y = np.array([s["outcome"] for s in per_sample])
    p = np.array([s["perception_agg"] for s in per_sample])
    m = np.array([s["mc_agg"] for s in per_sample])
    n = len(per_sample)

    correct_rate = float(y.mean())
    print(f"N samples: {n}, outcome correct rate: {correct_rate*100:.1f}%")

    def safe_auc(y, x):
        if len(set(y)) < 2 or len(set(x)) < 2:
            return None
        try:
            return float(roc_auc_score(y, x))
        except Exception:
            return None

    # Perception ↔ Outcome
    p_pearson = float(pearsonr(p, y)[0]) if n >= 3 else None
    p_spearman = float(spearmanr(p, y)[0]) if n >= 3 else None
    p_auc = safe_auc(y, p)

    # MC ↔ Outcome
    m_pearson = float(pearsonr(m, y)[0]) if n >= 3 else None
    m_auc = safe_auc(y, m)

    # Complementarity
    pm_r = float(pearsonr(p, m)[0]) if n >= 3 else None

    # AUC lift (mc only vs mc+perception)
    auc_lift = None
    try:
        if len(set(y)) >= 2:
            mc_clf = LogisticRegression().fit(m.reshape(-1, 1), y)
            mc_only_auc = float(roc_auc_score(y, mc_clf.predict_proba(m.reshape(-1, 1))[:, 1]))
            X = np.column_stack([m, p])
            both_clf = LogisticRegression().fit(X, y)
            both_auc = float(roc_auc_score(y, both_clf.predict_proba(X)[:, 1]))
            auc_lift = both_auc - mc_only_auc
        else:
            mc_only_auc = both_auc = None
    except Exception:
        mc_only_auc = both_auc = None

    def fmt(x, spec="+.3f"):
        return "na" if x is None else format(x, spec)
    print(f"\n[Perception ↔ Outcome]  Pearson={fmt(p_pearson)}  ROC AUC={fmt(p_auc, '.3f')}   (target ≥0.65)")
    print(f"[MC ↔ Outcome]          Pearson={fmt(m_pearson)}  ROC AUC={fmt(m_auc, '.3f')}   (target ≥0.70)")
    print(f"[Complementarity]       r(p,m)={fmt(pm_r)}              (target 0.3-0.7)")
    if auc_lift is not None:
        print(f"[AUC lift]              mc_only={fmt(mc_only_auc, '.3f')}  mc+p={fmt(both_auc, '.3f')}  lift={fmt(auc_lift)}   (target ≥+0.02)")

    checks = {
        "perception_auc_vs_outcome": (p_auc is not None) and (p_auc >= 0.65),
        "mc_auc_vs_outcome": (m_auc is not None) and (m_auc >= 0.70),
        "complementarity_in_range": (pm_r is not None) and (0.3 <= pm_r <= 0.7),
        "auc_lift_positive": (auc_lift is not None) and (auc_lift >= 0.02),
    }
    passed = sum(checks.values())
    print(f"\n[Alignment] {passed}/4 pass")
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")

    if passed >= 3:
        decision = "WEEK2_GO"
    elif passed == 2:
        decision = "WEEK2_PARTIAL"
    else:
        decision = "VERIFIER_REVISE"

    report = {
        "n": n,
        "outcome_correct_rate": correct_rate,
        "perception_pearson_vs_outcome": p_pearson,
        "perception_spearman_vs_outcome": p_spearman,
        "perception_auc_vs_outcome": p_auc,
        "mc_pearson_vs_outcome": m_pearson,
        "mc_auc_vs_outcome": m_auc,
        "perception_mc_correlation": pm_r,
        "mc_only_auc": mc_only_auc,
        "mc_plus_perception_auc": both_auc,
        "auc_lift": auc_lift,
        "checks": checks,
        "passed": passed,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\n[alignment] -> {OUT_JSON}")

    # Write d2_report.md
    def chk(name): return '✓' if checks[name] else '✗'
    md = f"""# D2 Pilot Report

## Setup
- N samples: {n}
- Outcome correct rate: {correct_rate*100:.1f}%

## Verifier alignment with outcome (guide §3.7)

| Metric | Value | Target | Pass |
|---|---:|---|:---:|
| Perception ROC AUC vs outcome | {fmt(p_auc, '.3f')} | ≥0.65 | {chk('perception_auc_vs_outcome')} |
| MC ROC AUC vs outcome | {fmt(m_auc, '.3f')} | ≥0.70 | {chk('mc_auc_vs_outcome')} |
| r(perception, mc) | {fmt(pm_r)} | 0.3-0.7 | {chk('complementarity_in_range')} |
| AUC lift (mc+p vs mc only) | {fmt(auc_lift)} | ≥+0.02 | {chk('auc_lift_positive')} |

**Passed: {passed}/4**

## Decision: **{decision}**

- `WEEK2_GO` (≥3/4): proceed to full PRM data construction (5K samples) → train PRM → GRPO with PRM
- `WEEK2_PARTIAL` (2/4): drop the weak component, proceed
- `VERIFIER_REVISE` (<2/4): stronger verifier or different framing
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(md)
    print(f"[alignment] -> {OUT_REPORT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
