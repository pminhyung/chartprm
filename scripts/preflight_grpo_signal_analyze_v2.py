"""GRPO signal analysis v2: additive combined + step-level variance.

Compare reward designs:
  R_out  = outcome (binary)
  R_per  = perception (mean over steps)
  R_mult = perception × outcome
  R_add@α = α·perception + (1-α)·outcome     for α ∈ {0.3, 0.5, 0.7}

Metrics per prompt (K rollouts):
  std(R), used as GRPO advantage normalization basis.
  fzs (frac with std=0) = fraction of prompts with no learning signal under that reward.

Bonus: step-level perception variance per rollout (alternative finer-grained signal).
"""
from __future__ import annotations
import json, os, statistics
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
EPS = 1e-6


def std(xs):
    if len(xs) < 2: return 0.0
    return statistics.pstdev(xs)


def main():
    recs = [json.loads(l) for l in open(BASE/"data/d2_hardbench/reports/preflight_grpo_signal.jsonl")]
    rows = []
    for r in recs:
        rollouts = r.get("rollouts", [])
        if not rollouts: continue
        outs = []
        pers = []
        step_var_per_rollout = []
        for ro in rollouts:
            o = ro.get("outcome")
            p = ro.get("perception")
            outs.append(o); pers.append(p)
            # step-level perception variance within this rollout
            aps = ro.get("axis_per_step") or []
            step_percs = []
            for axr in aps:
                if not isinstance(axr, dict): continue
                scored = [v for v in axr.values() if v is not None]
                if scored: step_percs.append(sum(scored)/len(scored))
            step_var_per_rollout.append(std(step_percs))
        # Filter valid
        out_v = [x for x in outs if x is not None]
        per_v = [x for x in pers if x is not None]
        if not out_v or not per_v: continue
        # Pair-valid combined
        mult = [p*o for o, p in zip(outs, pers) if o is not None and p is not None]
        add = lambda a: [a*p + (1-a)*o for o, p in zip(outs, pers) if o is not None and p is not None]
        add30 = add(0.3); add50 = add(0.5); add70 = add(0.7)
        rows.append({
            "combo": r["combo"], "id": r["id"], "bench": r["bench"],
            "outs": outs, "pers": pers,
            "std_out": std(out_v), "std_per": std(per_v), "std_mult": std(mult),
            "std_add30": std(add30), "std_add50": std(add50), "std_add70": std(add70),
            "step_var_per_rollout": step_var_per_rollout,
            "n_rollouts": len(rollouts),
            "out_mean": sum(out_v)/len(out_v),
        })

    # Aggregate
    n = len(rows)
    print(f"\n=== GRPO signal analysis (N={n} prompts, K rollouts each) ===\n")
    by_bench = {}
    for r in rows: by_bench.setdefault(r["bench"], []).append(r)
    by_bench["ALL"] = rows
    cols = ["std_out","std_per","std_mult","std_add30","std_add50","std_add70"]
    print(f"{'bench':<22}{'N':>4}", end="")
    for c in cols: print(f"{c.replace('std_','fzs_'):>10}", end="")
    print(f"{'<out>':>8}{'mean_step_var':>14}")
    print("=" * 110)
    for bench in ["chartqa_pro","charxiv_reasoning","chartmuseum","ALL"]:
        bs = by_bench.get(bench, [])
        if not bs:
            print(f"{bench:<22}  no data"); continue
        line = f"{bench:<22}{len(bs):>4}"
        for c in cols:
            fzs = sum(1 for r in bs if r[c] < EPS) / len(bs)
            line += f"{fzs:>10.2f}"
        mean_out = sum(r["out_mean"] for r in bs)/len(bs)
        # mean step variance (only rollouts with >1 valid step)
        flat_step_vars = []
        for r in bs:
            for v in r["step_var_per_rollout"]:
                if v > 0: flat_step_vars.append(v)
        msv = (sum(flat_step_vars)/len(flat_step_vars)) if flat_step_vars else 0.0
        line += f"{mean_out:>8.2f}{msv:>14.3f}"
        print(line)
    print(f"\nfzs_X = fraction of prompts where std(X across K rollouts) = 0 (no learning signal)")
    print(f"add{{α}} = α·perception + (1-α)·outcome")
    print(f"mean_step_var = mean std of step-level perception WITHIN a rollout (non-zero only)")

    # Per-prompt example: which reward design unlocks variance?
    print(f"\n=== Examples: outcome zero-std + add@0.5 unlocks variance ===")
    rescues = [r for r in rows if r["std_out"] < EPS and r["std_add50"] >= 0.05][:10]
    for r in rescues:
        print(f"  {r['combo']}/{r['id']}: outs={r['outs']} pers={[round(p,2) if p is not None else None for p in r['pers']]}  std_add50={r['std_add50']:.3f}")
    print(f"\n=== All-K identical rollouts (true zero-signal cases) ===")
    dead = [r for r in rows if r["std_out"] < EPS and r["std_per"] < EPS and r["std_add50"] < EPS][:5]
    for r in dead:
        print(f"  {r['combo']}/{r['id']}: outs={r['outs']} pers={[round(p,2) if p is not None else None for p in r['pers']]}")

    out_p = BASE/"data/d2_hardbench/reports/preflight_grpo_signal_v2.json"
    out_p.write_text(json.dumps({
        "n": n,
        "per_bench": {
            bench: {
                "n": len(bs),
                **{c.replace("std_","fzs_"): sum(1 for r in bs if r[c]<EPS)/len(bs) for c in cols}
            } for bench, bs in by_bench.items()
        }
    }, indent=2))
    print(f"\n-> {out_p}")


if __name__ == "__main__":
    main()
