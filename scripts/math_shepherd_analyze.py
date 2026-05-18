"""Analyze Math-Shepherd MC sub-rollout results: 4-cell distribution + Design A/B/C/D evaluation.

Inputs: data/d2_hardbench/reports/math_shepherd_dead.jsonl
Outputs:
  - console summary: 4-cell distribution, per-step stats, per-design reward
  - data/d2_hardbench/reports/math_shepherd_design_eval.json
"""
import json, statistics
from pathlib import Path
from collections import Counter

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
INP = BASE / "data/d2_hardbench/reports/math_shepherd_dead.jsonl"
OUT = BASE / "data/d2_hardbench/reports/math_shepherd_design_eval.json"

EPS = 1e-9


def classify_cell(mc_var, mc_mean, perc):
    """Classify each step into 4-cell zone (mentor framework)."""
    mc_informative = (mc_var is not None and mc_var > 0.1)
    perc_informative = (perc is not None and 0.05 < perc < 0.95)
    if mc_informative and perc_informative:
        return "both_informative"
    if (mc_var is None or mc_var < EPS) and perc_informative:
        if mc_mean == 0: return "mc_dead_perc_var"        # perception salvage zone
        if mc_mean == 1: return "mc_ceiling_perc_var"     # easy/already-correct
        return "mc_dead_perc_var"
    if mc_informative and not perc_informative:
        return "mc_var_perc_sat"
    return "both_silent"


def design_a_reward(mc_var, mc_mean, perc):
    """Complementary Salvage: MC primary, perception salvages when MC dead."""
    if mc_var is not None and mc_var > 0.1:
        return mc_mean
    if mc_mean == 0:
        return 0.3 * (perc if perc is not None else 0.0)
    if mc_mean == 1:
        return 1.0
    return 0.0


def design_b_reward(traj_outcome, perc, lam_step=0.5):
    """Hierarchical: trajectory outcome + step perception combined additively."""
    return (1 - lam_step) * (traj_outcome if traj_outcome is not None else 0) + lam_step * (perc if perc is not None else 0)


def design_c_reward(mc_var, mc_mean, perc, perc_thresh=0.7):
    """Perception-Conditioned MC: trust perception if high, else use MC."""
    if perc is not None and perc >= perc_thresh:
        return perc
    return mc_mean if mc_mean is not None else 0.0


def design_d_reward(mc_var, mc_mean, perc, outcome):
    """Variance-aware: MC primary; if MC dead, use outcome ⊕ perception."""
    if mc_var is not None and mc_var > 0:
        return mc_mean
    if outcome == 1: return 1.0
    return perc if perc is not None else 0.0


def additive_05(perc, outcome):
    return 0.5 * (perc if perc is not None else 0) + 0.5 * (outcome if outcome is not None else 0)


def outcome_only(outcome):
    return outcome if outcome is not None else 0


def std0(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def main():
    recs = [json.loads(l) for l in open(INP)]
    print(f"=== Loaded {len(recs)} dead samples from {INP.name} ===\n")

    # Per-step accumulator
    all_steps = []  # each: {sample_id, category, bench, step_k, mc_value, mc_var, step_perception, outcome, cell}

    print("=== Per-sample step records ===")
    for r in recs:
        if "error" in r:
            print(f"  SKIP {r['id']} ({r['combo']}): {r['error']}")
            continue
        sample_id = f"{r['id']}__{r['combo'].split('__')[-1]}"
        cat = r.get('category', '?')
        bench = r['bench']
        # We use the sample's own outcome variance (= 0 for dead) but use trajectory outcome from baseline
        # For analysis we need: does step-MC give us variance that outcome-only doesn't?
        outcome = 0 if cat == 'wrong_consistent' else 1
        print(f"\n{sample_id} cat={cat} bench={bench} n_steps={r['n_steps']}")
        for sr in r['step_records']:
            k = sr['k']
            mc_v = sr.get('mc_value')
            mc_var = sr.get('mc_var')
            perc = sr.get('step_perception')
            cell = classify_cell(mc_var, mc_v, perc)
            print(f"  step{k}: perc={perc} mc_value={mc_v} mc_var={mc_var:.3f} outcomes={sr['sub_outcomes']} cell={cell}")
            all_steps.append({
                "sample_id": sample_id, "category": cat, "bench": bench, "step_k": k,
                "mc_value": mc_v, "mc_var": mc_var, "step_perception": perc,
                "outcome": outcome, "cell": cell, "sub_outcomes": sr['sub_outcomes'],
                "n_valid_outcomes": sr['n_valid_outcomes'],
            })

    print(f"\n\n=== 4-CELL DISTRIBUTION (all {len(all_steps)} steps) ===")
    cell_count = Counter(s['cell'] for s in all_steps)
    for cell, c in cell_count.most_common():
        pct = c / len(all_steps) * 100
        print(f"  {cell:<24} {c:>4}  ({pct:5.1f}%)")

    print(f"\n=== 4-CELL × CATEGORY ===")
    cc = Counter((s['cell'], s['category']) for s in all_steps)
    for (cell, cat), c in sorted(cc.items()):
        print(f"  {cell:<24} {cat:<22} {c:>4}")

    # Key metric: among wrong_consistent samples, what fraction of steps have MC variance?
    wc_steps = [s for s in all_steps if s['category'] == 'wrong_consistent']
    cc_steps = [s for s in all_steps if s['category'] == 'correct_consistent']

    print(f"\n=== HEADLINE METRICS ===")
    if wc_steps:
        wc_var_steps = [s for s in wc_steps if s['mc_var'] and s['mc_var'] > EPS]
        print(f"wrong_consistent steps: {len(wc_steps)} total, {len(wc_var_steps)} with MC variance > 0 ({len(wc_var_steps)/len(wc_steps)*100:.1f}%)")
        if wc_var_steps:
            mvs = [s['mc_var'] for s in wc_var_steps]
            print(f"  mean MC var (when > 0): {sum(mvs)/len(mvs):.3f}, max: {max(mvs):.3f}")
        # MC dead steps (all sub_outcomes = 0)
        mc_dead = [s for s in wc_steps if (s['mc_value'] == 0 or s['mc_value'] is None)]
        print(f"wrong_consistent steps with MC dead (mc_value=0): {len(mc_dead)} / {len(wc_steps)} ({len(mc_dead)/len(wc_steps)*100:.1f}%)")
        # Of MC dead, how many have perception variance > 0 across steps within a sample (i.e., perception salvage potential)
        salvage_perc_distrib = [s['step_perception'] for s in mc_dead if s['step_perception'] is not None]
        if salvage_perc_distrib:
            print(f"  step_perception distribution in MC-dead wrong_consistent: min={min(salvage_perc_distrib):.2f} median={statistics.median(salvage_perc_distrib):.2f} max={max(salvage_perc_distrib):.2f}")
    if cc_steps:
        cc_var_steps = [s for s in cc_steps if s['mc_var'] and s['mc_var'] > EPS]
        print(f"correct_consistent steps: {len(cc_steps)} total, {len(cc_var_steps)} with MC variance > 0 ({len(cc_var_steps)/len(cc_steps)*100:.1f}%)")

    # Per-step reward under each design
    print(f"\n=== PER-STEP REWARD UNDER EACH DESIGN ===")
    print(f"{'sample/step':<40}{'cat':<20}{'cell':<22}{'R_out':>6}{'R_add':>7}{'R_A':>6}{'R_B':>6}{'R_C':>6}{'R_D':>6}")
    print("-" * 130)
    sample_design_rewards = {}  # (sample_id, design) -> list of per-step rewards
    for s in all_steps:
        r_out = outcome_only(s['outcome'])
        r_add = additive_05(s['step_perception'], s['outcome'])
        r_a = design_a_reward(s['mc_var'], s['mc_value'], s['step_perception'])
        r_b = design_b_reward(s['outcome'], s['step_perception'])
        r_c = design_c_reward(s['mc_var'], s['mc_value'], s['step_perception'])
        r_d = design_d_reward(s['mc_var'], s['mc_value'], s['step_perception'], s['outcome'])
        for name, val in [('R_out', r_out), ('R_add', r_add), ('R_A', r_a), ('R_B', r_b), ('R_C', r_c), ('R_D', r_d)]:
            sample_design_rewards.setdefault((s['sample_id'], name), []).append(val)
        print(f"{s['sample_id']+' step'+str(s['step_k']):<40}{s['category']:<20}{s['cell']:<22}{r_out:>6.2f}{r_add:>7.2f}{r_a:>6.2f}{r_b:>6.2f}{r_c:>6.2f}{r_d:>6.2f}")

    # Per-sample reward variance under each design (this is what GRPO needs for learning signal)
    # For each (sample, design), compute std across K=6 sub-rollout based pseudo-rollouts
    # Actually GRPO is over K policy rollouts on the same prompt, not over steps.
    # So we need to estimate: would per-step reward give different totals if our K=4 outer rollouts had different traces?
    # Simpler: aggregate sample reward (mean across steps) and compare designs by within-sample step variance.
    print(f"\n=== PER-SAMPLE STEP-REWARD VARIANCE (= within-sample step diversity, proxy for GRPO advantage richness) ===")
    print(f"{'sample_id':<40}{'cat':<20}{'std(R_out)':>11}{'std(R_add)':>11}{'std(R_A)':>10}{'std(R_B)':>10}{'std(R_C)':>10}{'std(R_D)':>10}")
    for sid in sorted({s['sample_id'] for s in all_steps}):
        cat = next(s['category'] for s in all_steps if s['sample_id'] == sid)
        line = f"{sid:<40}{cat:<20}"
        for name in ['R_out', 'R_add', 'R_A', 'R_B', 'R_C', 'R_D']:
            vs = sample_design_rewards[(sid, name)]
            line += f"{std0(vs):>11.3f}" if name in ['R_out', 'R_add'] else f"{std0(vs):>10.3f}"
        print(line)

    # Aggregate: fraction of samples where each design provides non-zero step variance
    print(f"\n=== AGGREGATE: FRAC SAMPLES WITH NON-ZERO STEP-REWARD VARIANCE (= GRPO has step-level signal) ===")
    for name in ['R_out', 'R_add', 'R_A', 'R_B', 'R_C', 'R_D']:
        samples_with_var = sum(1 for sid in {s['sample_id'] for s in all_steps}
                                if std0(sample_design_rewards[(sid, name)]) > EPS)
        total_samples = len({s['sample_id'] for s in all_steps})
        print(f"  {name}: {samples_with_var}/{total_samples} samples ({samples_with_var/total_samples*100:.1f}%)")
        # split by category
        wc_with_var = sum(1 for sid in {s['sample_id'] for s in all_steps if s['category']=='wrong_consistent'}
                          if std0(sample_design_rewards[(sid, name)]) > EPS)
        wc_total = len({s['sample_id'] for s in all_steps if s['category']=='wrong_consistent'})
        cc_with_var = sum(1 for sid in {s['sample_id'] for s in all_steps if s['category']=='correct_consistent'}
                          if std0(sample_design_rewards[(sid, name)]) > EPS)
        cc_total = len({s['sample_id'] for s in all_steps if s['category']=='correct_consistent'})
        print(f"     wrong_consistent: {wc_with_var}/{wc_total}, correct_consistent: {cc_with_var}/{cc_total}")

    # Save
    out_data = {
        "n_samples": len({s['sample_id'] for s in all_steps}),
        "n_steps_total": len(all_steps),
        "cell_distribution": dict(cell_count),
        "cell_by_category": {f"{cell}__{cat}": c for (cell,cat),c in cc.items()},
        "per_step": all_steps,
        "per_sample_step_reward_variance": {
            sid: {name: std0(sample_design_rewards[(sid, name)])
                  for name in ['R_out','R_add','R_A','R_B','R_C','R_D']}
            for sid in {s['sample_id'] for s in all_steps}
        },
    }
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
