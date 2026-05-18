"""Compare ORIGINAL vs STRICT verifier perception on 64 dead-sample steps.

Metrics:
  1. Mean perception (target 0.55-0.75)
  2. NO rate (target ≥10%)
  3. NA rate (info)
  4. Perception salvage zone fill: MC-dead steps with perception_variance > 0 within same sample
  5. Step-reward variance under 3 designs:
       Design A (Complementary Salvage): mc primary, 0.3·perception salvage when mc dead
       refined NO-gate: r = mc_value × {0.3 if any NO else 1.0}
       Design D: mc primary, outcome anchor when mc dead
"""
import json, statistics
from pathlib import Path
from collections import Counter

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
INP = BASE / "data/d2_hardbench/reports/math_shepherd_dead_sharpened.jsonl"
EPS = 1e-9


def design_a(mc_var, mc_mean, perc):
    if mc_var is not None and mc_var > 0.1:
        return mc_mean
    if mc_mean == 0:
        return 0.3 * (perc if perc is not None else 0.0)
    if mc_mean == 1:
        return 1.0
    return 0.0


def refined_no_gate(mc_var, mc_mean, axis_results):
    """r = mc_value × gate(axis_results); gate = 0.3 if any NO else 1.0."""
    has_no = any(v == 0 for v in axis_results.values())
    gate = 0.3 if has_no else 1.0
    return mc_mean * gate if mc_mean is not None else 0.0


def design_d(mc_var, mc_mean, perc, outcome):
    if mc_var is not None and mc_var > 0:
        return mc_mean
    if outcome == 1: return 1.0
    return perc if perc is not None else 0.0


def std0(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def perception_summary(records, key_axis, key_perc):
    """For original ('axis_results', 'step_perception') vs strict ('axis_results_strict', 'step_perception_strict')."""
    all_axis = []
    all_perc = []
    for r in records:
        if 'error' in r: continue
        for sr in r['step_records']:
            ax = sr.get(key_axis)
            if ax is not None:
                all_axis.append(ax)
            p = sr.get(key_perc)
            all_perc.append(p)  # may be None
    # axis-level counts
    yes = no = na = 0
    for ax in all_axis:
        for v in ax.values():
            if v == 1: yes += 1
            elif v == 0: no += 1
            else: na += 1  # None / NA
    total_ax = yes + no + na
    # step-level
    valid_perc = [p for p in all_perc if p is not None]
    mean_perc = sum(valid_perc) / len(valid_perc) if valid_perc else None
    return {
        "n_steps_total": len(all_perc),
        "n_steps_with_perc": len(valid_perc),
        "mean_perception": mean_perc,
        "yes_count": yes, "no_count": no, "na_count": na, "total_axis_verdicts": total_ax,
        "yes_rate": yes/total_ax if total_ax else 0,
        "no_rate": no/total_ax if total_ax else 0,
        "na_rate": na/total_ax if total_ax else 0,
    }


def salvage_zone_check(records, key_perc):
    """For each sample, check if any MC-dead steps have non-zero perception variance across sample steps."""
    n_samples_with_salvage = 0
    n_samples_total = 0
    n_mc_dead_steps_total = 0
    n_mc_dead_perc_varied = 0
    for r in records:
        if 'error' in r: continue
        n_samples_total += 1
        # collect perc of mc-dead steps
        mc_dead_percs = []
        for sr in r['step_records']:
            if sr.get('mc_value') == 0:
                p = sr.get(key_perc)
                mc_dead_percs.append(p)
                n_mc_dead_steps_total += 1
        valid = [p for p in mc_dead_percs if p is not None]
        if len(valid) >= 2 and std0(valid) > EPS:
            n_samples_with_salvage += 1
        # also count per-step variance: how many MC dead steps had a perc different from others
        if valid:
            for p in valid:
                if any(abs(p - q) > EPS for q in valid if q is not None):
                    n_mc_dead_perc_varied += 1
    return {
        "n_samples_total": n_samples_total,
        "n_samples_with_salvage_potential": n_samples_with_salvage,
        "n_mc_dead_steps": n_mc_dead_steps_total,
        "n_mc_dead_steps_with_perc_variance_within_sample": n_mc_dead_perc_varied,
    }


def design_eval(records, key_axis, key_perc):
    """Per-sample step-reward variance under each design."""
    n_samples_with_var = {"R_designA": 0, "R_NOgate": 0, "R_designD": 0}
    n_samples = 0
    rows = []
    for r in records:
        if 'error' in r: continue
        n_samples += 1
        cat = r.get('category', '?')
        outcome = 0 if cat == 'wrong_consistent' else 1
        rewards_a, rewards_g, rewards_d = [], [], []
        for sr in r['step_records']:
            mc_v = sr.get('mc_value')
            mc_var = sr.get('mc_var')
            perc = sr.get(key_perc)
            ax = sr.get(key_axis) or {}
            r_a = design_a(mc_var, mc_v, perc)
            r_g = refined_no_gate(mc_var, mc_v, ax)
            r_d = design_d(mc_var, mc_v, perc, outcome)
            rewards_a.append(r_a); rewards_g.append(r_g); rewards_d.append(r_d)
        rows.append({"sample_id": f"{r['id']}__{r['combo'].split('__')[-1]}", "cat": cat,
                     "std_A": std0(rewards_a), "std_G": std0(rewards_g), "std_D": std0(rewards_d)})
        if std0(rewards_a) > EPS: n_samples_with_var["R_designA"] += 1
        if std0(rewards_g) > EPS: n_samples_with_var["R_NOgate"] += 1
        if std0(rewards_d) > EPS: n_samples_with_var["R_designD"] += 1
    return n_samples_with_var, n_samples, rows


def main():
    records = [json.loads(l) for l in open(INP)]
    print(f"=== Loaded {len(records)} samples ===\n")

    print("=" * 70)
    print("PERCEPTION SUMMARY: ORIGINAL vs STRICT")
    print("=" * 70)
    orig = perception_summary(records, 'axis_results', 'step_perception')
    strict = perception_summary(records, 'axis_results_strict', 'step_perception_strict')
    print(f"{'metric':<28}{'ORIGINAL':>15}{'STRICT':>15}")
    for k in ['n_steps_with_perc', 'mean_perception', 'yes_count', 'no_count', 'na_count',
              'yes_rate', 'no_rate', 'na_rate']:
        ov = orig[k]; sv = strict[k]
        if isinstance(ov, float):
            print(f"{k:<28}{ov:>15.3f}{sv:>15.3f}")
        else:
            print(f"{k:<28}{ov:>15}{sv:>15}")

    print()
    print("=" * 70)
    print("SALVAGE ZONE FILL (MC dead steps × perception variance within sample)")
    print("=" * 70)
    orig_sv = salvage_zone_check(records, 'step_perception')
    strict_sv = salvage_zone_check(records, 'step_perception_strict')
    print(f"{'metric':<55}{'ORIGINAL':>10}{'STRICT':>10}")
    for k in ['n_samples_with_salvage_potential', 'n_mc_dead_steps',
              'n_mc_dead_steps_with_perc_variance_within_sample']:
        print(f"{k:<55}{orig_sv[k]:>10}{strict_sv[k]:>10}")
    print(f"{'n_samples_total':<55}{orig_sv['n_samples_total']:>10}{strict_sv['n_samples_total']:>10}")

    print()
    print("=" * 70)
    print("DESIGN EVAL: fraction samples with non-zero step-reward variance")
    print("=" * 70)
    orig_de, n, _ = design_eval(records, 'axis_results', 'step_perception')
    strict_de, _, strict_rows = design_eval(records, 'axis_results_strict', 'step_perception_strict')
    print(f"{'design':<22}{'ORIGINAL':>15}{'STRICT':>15}")
    for k in ['R_designA', 'R_NOgate', 'R_designD']:
        print(f"{k:<22}{orig_de[k]}/{n}{'':>11}{strict_de[k]}/{n}")

    print()
    print("=" * 70)
    print("PER-SAMPLE STEP-REWARD STD UNDER STRICT (sorted by R_designA)")
    print("=" * 70)
    strict_rows.sort(key=lambda r: -r['std_A'])
    print(f"{'sample_id':<45}{'cat':<22}{'std(A)':>10}{'std(NO-gate)':>14}{'std(D)':>10}")
    for r in strict_rows:
        print(f"{r['sample_id']:<45}{r['cat']:<22}{r['std_A']:>10.3f}{r['std_G']:>14.3f}{r['std_D']:>10.3f}")

    # Per-step strict perception examples (first 30 steps with NO or non-1 perception)
    print()
    print("=" * 70)
    print("STRICT VERIFIER: per-step verdicts where ≥1 axis = NO (= candidate salvage signal)")
    print("=" * 70)
    n_shown = 0
    for r in records:
        if 'error' in r: continue
        sid = f"{r['id']}__{r['combo'].split('__')[-1]}"
        for sr in r['step_records']:
            ax = sr.get('axis_results_strict', {})
            if any(v == 0 for v in ax.values()):
                p = sr.get('step_perception_strict')
                p_orig = sr.get('step_perception')
                mc_v = sr.get('mc_value')
                print(f"  {sid:<40} step{sr['k']} mc={mc_v} perc(orig)={p_orig} perc(strict)={p} axis={ax}")
                n_shown += 1
                if n_shown >= 40: break
        if n_shown >= 40: break


if __name__ == "__main__":
    main()
