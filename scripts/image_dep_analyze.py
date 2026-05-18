"""Analyze image-dependency MC data: 3 pass criteria + reward design comparison.

PASS CRITERIA (from user spec §4):
  1. visual_dep std ≥ 0.15  (meaningful spread)
  2. 4-cell distribution (mc_with × visual_dep): each cell ≥ 10%
  3. previous perception-failed MC-dead steps with visual_dep > 0.2 ≥ 10

Reward designs compared:
  R_out:    outcome_only
  R_MC:     mc_with_image (standard Math-Shepherd)
  R_VPPO:   user's spec — case-branched on mc_with + visual_dep
  R_VPPO+:  refined to handle negative visual_dep explicitly (image misleading)
"""
import json, statistics
from pathlib import Path
from collections import Counter

BASE = Path("/ex_disk2/mhpark/poc/chartvr")
INP = BASE / "data/d2_hardbench/reports/image_dep_dead.jsonl"
OUT = BASE / "data/d2_hardbench/reports/image_dep_design_eval.json"
EPS = 1e-9


def vppo_reward(mc_with, mc_without):
    """User's spec — Section 1."""
    if mc_with is None or mc_without is None:
        return 0.0
    vd = mc_with - mc_without
    if mc_with >= 0.7:
        return mc_with * (1 + 0.3 * (1 - vd))
    if mc_with == 0:
        return 0.0 if vd < 0.2 else 0.1
    return mc_with  # standard MC


def vppo_plus_reward(mc_with, mc_without):
    """Refined: penalty for negative visual_dep (image misleading)."""
    if mc_with is None or mc_without is None:
        return 0.0
    vd = mc_with - mc_without
    # New explicit branch: large negative visual_dep = perception actively misleading
    if vd <= -0.3:
        return -0.2  # explicit penalty; flags perception-induced wrong path
    if mc_with >= 0.7:
        return mc_with * (1 + 0.3 * (1 - vd))
    if mc_with == 0:
        return 0.0 if vd < 0.2 else 0.1
    return mc_with


def classify_4cell(mc_with, visual_dep):
    """4-cell per user spec §1 Case 1/2/3/4 — using cutoff 0.7 mc_with and 0.2 visual_dep."""
    if mc_with is None or visual_dep is None:
        return "missing"
    mc_high = mc_with >= 0.7
    vd_high = visual_dep >= 0.2
    if mc_high and not vd_high: return "case1_mcHigh_vdLow_self_sufficient"
    if mc_high and vd_high:     return "case2_mcHigh_vdHigh_image_critical"
    if not mc_high and not vd_high: return "case3_mcLow_vdLow_bad_prefix"
    if not mc_high and vd_high:     return "case4_mcLow_vdHigh_perception_fail"
    return "?"


def std0(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) >= 2 else 0.0


def main():
    records = [json.loads(l) for l in open(INP)]
    print(f"=== Loaded {len(records)} samples ===\n")

    # Per-step accumulator
    all_steps = []
    for r in records:
        if 'error' in r or 'error_image_dep' in r: continue
        sid = f"{r['id']}__{r['combo'].split('__')[-1]}"
        for sr in r['step_records']:
            vd = sr.get('visual_dep')
            mc_w = sr.get('mc_value')
            mc_wo = sr.get('mc_without_value')
            cell = classify_4cell(mc_w, vd)
            all_steps.append({
                'sample_id': sid, 'cat': r.get('category'), 'bench': r['bench'],
                'step_k': sr['k'],
                'mc_with': mc_w, 'mc_without': mc_wo, 'visual_dep': vd,
                'mc_var': sr.get('mc_var'), 'mc_without_var': sr.get('mc_without_var'),
                'cell_4': cell,
                'step_perception': sr.get('step_perception'),
            })

    print("=== PER-STEP RECORD (sample / step / mc_with / mc_without / vd / cell) ===")
    for s in all_steps:
        print(f"  {s['sample_id']:<40} step{s['step_k']} {s['cat']:<22} "
              f"mc_w={s['mc_with']:.2f} mc_wo={s['mc_without']:.2f} "
              f"vd={s['visual_dep']:>+.2f} {s['cell_4']}")

    print(f"\n=== PASS CRITERION 1: visual_dep spread ===")
    vds = [s['visual_dep'] for s in all_steps if s['visual_dep'] is not None]
    sd = std0(vds)
    mn, mx = min(vds), max(vds)
    print(f"  N={len(vds)} mean={sum(vds)/len(vds):.3f} std={sd:.3f} min={mn:+.3f} max={mx:+.3f}")
    print(f"  PASS (std ≥ 0.15)? {'YES' if sd >= 0.15 else 'NO'}")
    # Distribution
    bins = [(-1.0,-0.3,"strongly_neg"),(-0.3,-0.1,"mildly_neg"),(-0.1,0.1,"zero"),(0.1,0.3,"mildly_pos"),(0.3,0.6,"strongly_pos"),(0.6,1.01,"very_strong_pos")]
    print(f"  Histogram:")
    for lo, hi, name in bins:
        c = sum(1 for v in vds if lo <= v < hi)
        print(f"    {name:<20} [{lo:+.2f},{hi:+.2f}): {c} ({c/len(vds)*100:5.1f}%)")

    print(f"\n=== PASS CRITERION 2: 4-cell distribution ===")
    cell_count = Counter(s['cell_4'] for s in all_steps)
    n = len(all_steps)
    cells_above_10 = 0
    for cell in ["case1_mcHigh_vdLow_self_sufficient", "case2_mcHigh_vdHigh_image_critical",
                 "case3_mcLow_vdLow_bad_prefix", "case4_mcLow_vdHigh_perception_fail"]:
        c = cell_count.get(cell, 0)
        pct = c / n * 100
        is_ok = pct >= 10
        cells_above_10 += int(is_ok)
        print(f"  {cell:<48} {c:>3} ({pct:5.1f}%) {'PASS' if is_ok else 'fail'}")
    print(f"  PASS (4/4 cells ≥10%)? {'YES' if cells_above_10 == 4 else f'PARTIAL ({cells_above_10}/4)'}")

    print(f"\n=== PASS CRITERION 3: MC-dead step recovery via visual_dep ===")
    mc_dead_steps = [s for s in all_steps if s['mc_with'] == 0]
    mc_dead_vd_high = [s for s in mc_dead_steps if s['visual_dep'] is not None and s['visual_dep'] > 0.2]
    print(f"  MC-dead steps: {len(mc_dead_steps)}")
    print(f"  among them, visual_dep > 0.2: {len(mc_dead_vd_high)}")
    print(f"  PASS (≥10)? {'YES' if len(mc_dead_vd_high) >= 10 else 'NO'}")
    # also count negative visual_dep among MC-dead (= image misleading)
    mc_dead_vd_neg = [s for s in mc_dead_steps if s['visual_dep'] is not None and s['visual_dep'] < -0.1]
    print(f"  MC-dead with visual_dep < -0.1 (image misleading): {len(mc_dead_vd_neg)}")

    print(f"\n=== REWARD DESIGN COMPARISON (per-sample step-reward variance) ===")
    sample_rewards = {}  # (sid, design) -> list[float]
    for s in all_steps:
        for name, fn in [('R_out', lambda: 0 if s['cat']=='wrong_consistent' else 1),
                          ('R_MC', lambda: s['mc_with']),
                          ('R_VPPO', lambda: vppo_reward(s['mc_with'], s['mc_without'])),
                          ('R_VPPO+', lambda: vppo_plus_reward(s['mc_with'], s['mc_without']))]:
            sample_rewards.setdefault((s['sample_id'], name), []).append(fn())

    sids = sorted({s['sample_id'] for s in all_steps})
    print(f"{'sample_id':<45}{'cat':<22}{'std(out)':>10}{'std(MC)':>10}{'std(VPPO)':>11}{'std(VPPO+)':>12}")
    for sid in sids:
        cat = next(s['cat'] for s in all_steps if s['sample_id'] == sid)
        line = f"{sid:<45}{cat:<22}"
        for name in ['R_out', 'R_MC', 'R_VPPO', 'R_VPPO+']:
            v = std0(sample_rewards[(sid, name)])
            line += f"{v:>10.3f}" if name in ['R_out','R_MC'] else f"{v:>11.3f}"
        print(line)

    print(f"\n=== AGGREGATE: fraction samples with non-zero step-reward variance ===")
    for name in ['R_out', 'R_MC', 'R_VPPO', 'R_VPPO+']:
        n_var = sum(1 for sid in sids if std0(sample_rewards[(sid, name)]) > EPS)
        print(f"  {name}: {n_var}/{len(sids)} ({n_var/len(sids)*100:.1f}%)")
        wc_n = sum(1 for sid in sids if next(s['cat'] for s in all_steps if s['sample_id']==sid)=='wrong_consistent')
        wc_var = sum(1 for sid in sids if next(s['cat'] for s in all_steps if s['sample_id']==sid)=='wrong_consistent' and std0(sample_rewards[(sid, name)]) > EPS)
        print(f"     wrong_consistent: {wc_var}/{wc_n}")

    # Save
    out_data = {
        "n_samples": len(sids), "n_steps": len(all_steps),
        "criterion_1_visual_dep_std": sd,
        "criterion_2_4cell": dict(cell_count),
        "criterion_3_mc_dead_recovery": len(mc_dead_vd_high),
        "mc_dead_image_misleading_count": len(mc_dead_vd_neg),
        "per_step": all_steps,
    }
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
