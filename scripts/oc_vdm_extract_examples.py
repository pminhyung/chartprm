"""Extract verbatim per-pattern step examples for the validation report.

For each pattern (A/B/C/D), print:
  - combo, sample id, step k
  - mc_with, mc_without, vd
  - step_text (first 200c)
  - one with-image continuation pred, one without-image continuation pred
"""
from __future__ import annotations
import json, os, random
from collections import defaultdict
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
REPORTS_DIR = BASE / "data/d2_hardbench/reports/image_dep_12combo"
OUT = BASE / "data/d2_hardbench/reports/oc_vdm_pattern_examples.json"

random.seed(42)


def classify(mc_w, mc_wo):
    if mc_w is None or mc_wo is None:
        return "_invalid"
    vd = mc_w - mc_wo
    if mc_w == 0 and mc_wo == 0:
        return "A_hard_impossible"
    if mc_wo >= 0.5 and vd < 0:
        return "B_leakage"
    if mc_w >= 0.5 and vd > 0.3:
        return "C_image_critical"
    if 0 < mc_w < 0.3 and mc_wo <= 0.1:
        return "D_hard_perception"
    return "E_other"


def main():
    by_pattern = defaultdict(list)
    for path in sorted(REPORTS_DIR.glob("*.jsonl")):
        for line in open(path):
            try:
                s = json.loads(line)
            except Exception:
                continue
            if 'step_records' not in s:
                continue
            for sr in s['step_records']:
                mc_w = sr.get('mc_value')
                mc_wo = sr.get('mc_without_value')
                if mc_w is None or mc_wo is None:
                    continue
                p = classify(mc_w, mc_wo)
                ex = {
                    "pattern": p,
                    "combo": s.get('combo'),
                    "id": s.get('id'),
                    "question": (s.get('question') or '')[:200],
                    "gold": s.get('gold'),
                    "k": sr.get('k'),
                    "n_steps": s.get('n_steps'),
                    "mc_with": mc_w,
                    "mc_without": mc_wo,
                    "visual_dep": mc_w - mc_wo,
                    "step_text": (sr.get('step_text') or '')[:250],
                    "with_preds": sr.get('sub_preds_with', [])[:3],
                    "without_preds": sr.get('sub_preds_without', [])[:3],
                }
                by_pattern[p].append(ex)

    # Sample up to 5 per pattern
    examples = {}
    for p, items in by_pattern.items():
        if not items:
            continue
        sample = random.sample(items, min(5, len(items)))
        examples[p] = sample

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "counts": {p: len(items) for p, items in by_pattern.items()},
        "examples": examples,
    }, open(OUT, "w"), indent=2, ensure_ascii=False)

    print(f"\nPattern counts:")
    for p, items in sorted(by_pattern.items(), key=lambda kv: -len(kv[1])):
        print(f"  {p:25s}: {len(items)}")

    print(f"\n=== Per-pattern examples (5 each, random seed=42) ===")
    for p in ["C_image_critical", "B_leakage", "D_hard_perception", "A_hard_impossible", "E_other"]:
        if p not in examples:
            continue
        print(f"\n--- Pattern {p} (n={len(by_pattern[p])}) ---")
        for ex in examples[p][:5]:
            print(f"\n{ex['combo']}/{ex['id']} step k={ex['k']}/{ex['n_steps']}")
            print(f"  Q: {ex['question'][:120]}")
            print(f"  gold: {(ex['gold'] or '')[:80]}")
            print(f"  mc_w={ex['mc_with']:.2f} mc_wo={ex['mc_without']:.2f} vd={ex['visual_dep']:+.2f}")
            print(f"  step: {ex['step_text'][:150]}")
            print(f"  with preds: {[p[:40] for p in ex['with_preds']]}")
            print(f"  without preds: {[p[:40] for p in ex['without_preds']]}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
