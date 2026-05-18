"""Analyze preflight_grpo_signal.jsonl: GRPO learning signal diagnostic.

Per prompt (K rollouts):
  outcome ∈ {0, 1}, perception ∈ [0,1], combined = perception × outcome.

Metrics:
  - std_outcome, std_perception, std_combined across K rollouts
  - frac_zero_std: fraction of prompts where std == 0 (no learning signal)
  - mean reward per prompt (for sanity)
  - perception_outcome correlation within prompt
  - cross-prompt aggregate

Output: console table + JSON summary.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from collections import Counter
import statistics

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def std(xs):
    if len(xs) < 2: return 0.0
    return statistics.pstdev(xs)


def main():
    recs = [json.loads(l) for l in open(BASE/"data/d2_hardbench/reports/preflight_grpo_signal.jsonl")]
    print(f"loaded {len(recs)} prompts")

    rows = []
    bench_groups = {}
    for r in recs:
        bench = r["bench"]
        K = r.get("K", 4)
        rollouts = r.get("rollouts", [])
        if not rollouts: continue
        outcomes = [ro.get("outcome") for ro in rollouts]
        percs = [ro.get("perception") for ro in rollouts]
        # filter Nones for variance
        out_valid = [x for x in outcomes if x is not None]
        per_valid = [x for x in percs if x is not None]
        if not out_valid or not per_valid: continue
        out_std = std(out_valid)
        per_std = std(per_valid)
        # combined per rollout (only where both defined)
        combined = []
        for o, p in zip(outcomes, percs):
            if o is not None and p is not None:
                combined.append(p * o)
        comb_std = std(combined) if combined else 0.0
        # mean
        out_mean = sum(out_valid)/len(out_valid)
        per_mean = sum(per_valid)/len(per_valid)
        comb_mean = sum(combined)/max(len(combined),1) if combined else 0.0
        row = {
            "combo": r["combo"], "id": r["id"], "bench": bench, "K": len(rollouts),
            "out_std": out_std, "per_std": per_std, "comb_std": comb_std,
            "out_mean": out_mean, "per_mean": per_mean, "comb_mean": comb_mean,
            "outcomes": outcomes, "perceptions": [round(p,3) if p is not None else None for p in percs]
        }
        rows.append(row)
        bench_groups.setdefault(bench, []).append(row)

    # Aggregate
    print(f"\n=== Per-prompt reward variance analysis (K=4 rollouts each) ===")
    print(f"{'bench':<22}{'N':>4}{'fzs_out':>9}{'fzs_per':>9}{'fzs_comb':>10}{'<out>':>8}{'<per>':>8}{'<comb>':>8}")
    print("=" * 80)
    EPS = 1e-6
    for bench in ["chartqa_pro","charxiv_reasoning","chartmuseum"]:
        bs = bench_groups.get(bench, [])
        if not bs:
            print(f"{bench:<22}  no data")
            continue
        n = len(bs)
        fzs_out = sum(1 for r in bs if r["out_std"] < EPS) / n
        fzs_per = sum(1 for r in bs if r["per_std"] < EPS) / n
        fzs_comb = sum(1 for r in bs if r["comb_std"] < EPS) / n
        m_out = sum(r["out_mean"] for r in bs)/n
        m_per = sum(r["per_mean"] for r in bs)/n
        m_comb = sum(r["comb_mean"] for r in bs)/n
        print(f"{bench:<22}{n:>4}{fzs_out:>9.2f}{fzs_per:>9.2f}{fzs_comb:>10.2f}{m_out:>8.2f}{m_per:>8.2f}{m_comb:>8.2f}")

    # overall
    n = len(rows)
    fzs_out = sum(1 for r in rows if r["out_std"] < EPS) / n
    fzs_per = sum(1 for r in rows if r["per_std"] < EPS) / n
    fzs_comb = sum(1 for r in rows if r["comb_std"] < EPS) / n
    m_out = sum(r["out_mean"] for r in rows)/n
    m_per = sum(r["per_mean"] for r in rows)/n
    m_comb = sum(r["comb_mean"] for r in rows)/n
    print("-" * 80)
    print(f"{'OVERALL':<22}{n:>4}{fzs_out:>9.2f}{fzs_per:>9.2f}{fzs_comb:>10.2f}{m_out:>8.2f}{m_per:>8.2f}{m_comb:>8.2f}")

    print(f"\nfzs_X = fraction of prompts where std(X across K rollouts) = 0 (no learning signal under X-only reward)")
    print(f"<X> = mean of X over all K×N rollouts")
    print(f"Lower fzs = better GRPO learning signal density. Combined should beat outcome-only.")

    # Concrete examples
    print(f"\n=== Examples where outcome=zero-std but perception adds signal ===")
    cases = [r for r in rows if r["out_std"] < EPS and r["per_std"] >= 0.05][:8]
    for c in cases:
        print(f"  {c['combo']}/{c['id']}: outcomes={c['outcomes']} perc={c['perceptions']}")
    print(f"\n=== Examples where ALL three (out/per/comb) have zero std (no signal) ===")
    cases = [r for r in rows if r["out_std"] < EPS and r["per_std"] < EPS][:8]
    for c in cases:
        print(f"  {c['combo']}/{c['id']}: outcomes={c['outcomes']} perc={c['perceptions']}")

    # save summary
    summary = {
        "n_prompts": n,
        "per_bench": {bench: {
            "n": len(bs),
            "fzs_outcome": sum(1 for r in bs if r["out_std"] < EPS)/len(bs),
            "fzs_perception": sum(1 for r in bs if r["per_std"] < EPS)/len(bs),
            "fzs_combined": sum(1 for r in bs if r["comb_std"] < EPS)/len(bs),
            "mean_outcome": sum(r["out_mean"] for r in bs)/len(bs),
            "mean_perception": sum(r["per_mean"] for r in bs)/len(bs),
            "mean_combined": sum(r["comb_mean"] for r in bs)/len(bs),
        } for bench, bs in bench_groups.items()},
        "overall": {
            "fzs_outcome": fzs_out, "fzs_perception": fzs_per, "fzs_combined": fzs_comb,
            "mean_outcome": m_out, "mean_perception": m_per, "mean_combined": m_comb,
        }
    }
    op = BASE/"data/d2_hardbench/reports/preflight_grpo_signal_summary.json"
    op.write_text(json.dumps(summary, indent=2))
    print(f"\n-> {op}")


if __name__ == "__main__":
    main()
