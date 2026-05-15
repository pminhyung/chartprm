"""D2 alignment v2 — same 4-metric framework, v2 inputs (NOT_FOUND aware).

Input:  data/d2_pilot/perception_v2.jsonl  (sample-level perception scores)
        data/d2_pilot/mc_results.jsonl     (existing MC scores)
        data/d1_pilot/segmented_v3.jsonl   (original content/reasoning source)
Output: data/d2_pilot/alignment_v2.json
        docs/d2_report_v2.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartvr.extraction import extract_answer, relaxed_accuracy  # noqa: E402

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
PERCEPTION_JSONL = BASE / "data/d2_pilot/perception_v2.jsonl"
MC_JSONL = BASE / "data/d2_pilot/mc_results.jsonl"
TRACES_SEG = BASE / "data/d1_pilot/segmented_v3.jsonl"
OUT_JSON = BASE / "data/d2_pilot/alignment_v2.json"
OUT_REPORT = BASE / "docs/d2_report_v2.md"


def extract_final_from_content(content: str, reasoning: str = "") -> str:
    if content and content.strip():
        return extract_answer(content) or content.strip()
    return extract_answer(reasoning) or ""


def main():
    recs_p = [json.loads(l) for l in open(PERCEPTION_JSONL) if l.strip()]
    mc_by_id = {r["id"]: r for r in (json.loads(l) for l in open(MC_JSONL) if l.strip())}
    traces_by_id = {r["id"]: r for r in (json.loads(l) for l in open(TRACES_SEG) if l.strip())}

    per_sample = []
    for r in recs_p:
        src = traces_by_id.get(r["id"], {})
        final = extract_final_from_content(src.get("content", ""), src.get("reasoning_content", ""))
        outcome = int(relaxed_accuracy(final, r["gold_answer"]))

        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                    if s is not None and s.get("mean_score") is not None]
        nf_rates = [s.get("not_found_rate", 0.0) for s in r.get("step_perception_scores", [])
                    if s is not None]
        if not p_scores:
            continue
        perception_agg = float(np.mean(p_scores))
        not_found_agg = float(np.mean(nf_rates)) if nf_rates else 0.0

        mc_rec = mc_by_id.get(r["id"], {})
        mc_steps = mc_rec.get("step_outcome_scores", [])
        mc_scores = [s["mean_score"] for s in mc_steps if s is not None]
        if not mc_scores:
            continue
        mc_agg = float(np.mean(mc_scores))

        per_sample.append({
            "id": r["id"], "outcome": outcome,
            "perception_agg": perception_agg, "mc_agg": mc_agg,
            "not_found_agg": not_found_agg,
        })

    if not per_sample:
        print("[alignment_v2] no samples with both perception and mc data")
        return 1

    y = np.array([s["outcome"] for s in per_sample])
    p = np.array([s["perception_agg"] for s in per_sample])
    m = np.array([s["mc_agg"] for s in per_sample])
    nfa = np.array([s["not_found_agg"] for s in per_sample])
    n = len(per_sample)
    correct_rate = float(y.mean())
    print(f"[alignment_v2] N={n}  outcome correct rate={correct_rate*100:.1f}%")
    print(f"  mean NOT_FOUND rate per sample: {nfa.mean()*100:.1f}%")

    def safe_auc(yy, xx):
        if len(set(yy)) < 2 or len(set(xx)) < 2:
            return None
        try:
            return float(roc_auc_score(yy, xx))
        except Exception:
            return None

    p_pearson = float(pearsonr(p, y)[0]) if n >= 3 else None
    p_spearman = float(spearmanr(p, y)[0]) if n >= 3 else None
    p_auc = safe_auc(y, p)
    m_pearson = float(pearsonr(m, y)[0]) if n >= 3 else None
    m_auc = safe_auc(y, m)
    pm_r = float(pearsonr(p, m)[0]) if n >= 3 else None

    auc_lift = mc_only_auc = both_auc = None
    try:
        if len(set(y)) >= 2:
            mc_clf = LogisticRegression().fit(m.reshape(-1, 1), y)
            mc_only_auc = float(roc_auc_score(y, mc_clf.predict_proba(m.reshape(-1, 1))[:, 1]))
            X = np.column_stack([m, p])
            both_clf = LogisticRegression().fit(X, y)
            both_auc = float(roc_auc_score(y, both_clf.predict_proba(X)[:, 1]))
            auc_lift = both_auc - mc_only_auc
    except Exception:
        pass

    def fmt(x, spec="+.3f"):
        return "na" if x is None else format(x, spec)
    print(f"\n[Perception ↔ Outcome] Pearson={fmt(p_pearson)}  AUC={fmt(p_auc, '.3f')}  (target ≥0.65)")
    print(f"[MC ↔ Outcome]         Pearson={fmt(m_pearson)}  AUC={fmt(m_auc, '.3f')}  (target ≥0.70)")
    print(f"[Complementarity]      r(p,m)={fmt(pm_r)}                  (target 0.3-0.7)")
    if auc_lift is not None:
        print(f"[AUC lift]             mc_only={fmt(mc_only_auc,'.3f')}  mc+p={fmt(both_auc,'.3f')}  lift={fmt(auc_lift)}  (target ≥+0.02)")

    checks = {
        "perception_auc_vs_outcome": (p_auc is not None) and (p_auc >= 0.65),
        "mc_auc_vs_outcome": (m_auc is not None) and (m_auc >= 0.70),
        "complementarity_in_range": (pm_r is not None) and (0.3 <= pm_r <= 0.7),
        "auc_lift_positive": (auc_lift is not None) and (auc_lift >= 0.02),
    }
    passed = sum(checks.values())
    print(f"\n[Alignment v2] {passed}/4 pass")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nNOTE: 통과 못 해도 Day 2 4-cell measurement 진행 — decoupling 자체가 thesis.")

    report = {
        "n": n, "outcome_correct_rate": correct_rate,
        "not_found_rate_mean": float(nfa.mean()),
        "perception_pearson_vs_outcome": p_pearson,
        "perception_spearman_vs_outcome": p_spearman,
        "perception_auc_vs_outcome": p_auc,
        "mc_pearson_vs_outcome": m_pearson,
        "mc_auc_vs_outcome": m_auc,
        "perception_mc_correlation": pm_r,
        "mc_only_auc": mc_only_auc,
        "mc_plus_perception_auc": both_auc,
        "auc_lift": auc_lift,
        "checks": checks, "passed": passed,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\n[alignment_v2] -> {OUT_JSON}")
    # Markdown report
    def chk(name): return 'PASS' if checks[name] else 'FAIL'
    md = f"""# D2 Pilot Report v2 (Spec A+B+C patched)

## Setup
- Verifier: InternVL2.5-26B (cross-family, NOT_FOUND aware)
- Extractor: Qwen3.6-27B LLM-primary (strict entity validation)
- Tolerance: 10% rel / 1.0 abs → full; ≤25% rel → partial
- N samples: {n}, outcome correct rate: {correct_rate*100:.1f}%
- Mean NOT_FOUND rate per sample: {nfa.mean()*100:.1f}%

## Alignment 4-metric

| Metric | Value | Target | Pass |
|---|---:|---|:---:|
| Perception ROC AUC vs outcome | {fmt(p_auc, '.3f')} | ≥0.65 | {chk('perception_auc_vs_outcome')} |
| MC ROC AUC vs outcome         | {fmt(m_auc, '.3f')} | ≥0.70 | {chk('mc_auc_vs_outcome')} |
| r(perception, mc)             | {fmt(pm_r)}         | 0.3-0.7 | {chk('complementarity_in_range')} |
| AUC lift (mc+p vs mc)         | {fmt(auc_lift)}     | ≥+0.02 | {chk('auc_lift_positive')} |

passed: {passed}/4

NOTE: 4-cell distribution이 본 실험의 main decision metric (alignment 통과 못 해도 진행).
"""
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(md)
    print(f"[alignment_v2] markdown -> {OUT_REPORT}")
    return 0 if passed >= 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
