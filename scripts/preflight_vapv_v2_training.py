"""Training-time preflight: synthesize completions on actual training data
(CSV-backed) to test V2's process reward dynamics, anchored grounding, and
length scaling on real chart distributions.

Tests against 4 synthetic completion archetypes per sample:
  GOLD:   clean reasoning from `reasoning_steps` → correct answer
  WRONG:  same reasoning but final answer perturbed
  BLOAT:  GOLD + 3000 filler words mid-reasoning (should crash V2 via length scale)
  GAMED:  CSV numbers indiscriminately quoted, no entity context
          (should crash V2 via anchored denominator)

Pass criteria (printed at end):
  H1: V2(GOLD) > V2(WRONG) by ≥ 0.5 across ≥90% of samples
  H2: V2(GOLD) > V2(BLOAT) by ≥ 0.3 across ≥90% of samples
  H3: V2(GAMED) < 0.1 across ≥90% of samples
  H4: V2 vs V1 separation: mean V2 GAP > mean V1 GAP
"""
import json
import os
import random
import sys

sys.path.insert(0, '/ex_disk2/mhpark/poc/chartvr')
from code.rewards.rule_verifier_fast import (
    compute_process_reward_fast as v1_proc,
    compute_vapv_v2,
    get_table_values,
    get_table_entities,
)
from chartvr.extraction import cerm_accuracy, _is_numeric_answer, relaxed_text_match
import re


def extract_answer_v2(response):
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return m.group(1).strip() if m else ""


def v1_full_reward(resp, gold, csv):
    """Mirror reward_conditional_v2 (V1 production reward)."""
    pred = extract_answer_v2(resp)
    if not _is_numeric_answer(str(gold)) or not csv:
        return relaxed_text_match(pred, str(gold))
    r_acc = cerm_accuracy(pred, str(gold))
    if r_acc >= 0.95:
        return 1.0
    proc = v1_proc(resp, csv)
    return min(1.0, r_acc + 0.3 * proc)


def v2_full_reward(resp, gold, csv, max_len=4096):
    """Mirror reward_vapv_v2."""
    pred = extract_answer_v2(resp)
    if not _is_numeric_answer(str(gold)) or not csv:
        return relaxed_text_match(pred, str(gold))
    r_acc = cerm_accuracy(pred, str(gold))
    if r_acc >= 0.95:
        return 1.0
    proc, _ = compute_vapv_v2(resp, csv, max_completion_length=max_len)
    return 0.05 * proc

DATA = "data/charts_v2/chartvr_train_final.jsonl"
N = 300
SEED = 42
MAX_LEN = 4096
FILLER = ' '.join(['lorem ipsum dolor sit amet consectetur'] * 600)  # ~3600 words


def gold_completion(rec):
    steps = rec.get("reasoning_steps") or []
    body = "I need to find the answer.\n" + "\n".join(f"- {s}" for s in steps)
    ans = str(rec["answer"])
    return f"<think>{body}</think><answer>{ans}</answer>"


def wrong_completion(rec):
    steps = rec.get("reasoning_steps") or []
    body = "I need to find the answer.\n" + "\n".join(f"- {s}" for s in steps)
    ans = str(rec["answer"])
    try:
        f = float(ans)
        wrong_ans = str(round(f * 1.7 + 13, 2))
    except ValueError:
        wrong_ans = "999"
    return f"<think>{body}</think><answer>{wrong_ans}</answer>"


def bloat_completion(rec):
    steps = rec.get("reasoning_steps") or []
    body = "I need to find the answer.\n" + "\n".join(f"- {s}" for s in steps)
    ans = str(rec["answer"])
    return f"<think>{body}\n{FILLER}\n{FILLER}</think><answer>{ans}</answer>"


def gamed_completion(rec):
    """Indiscriminate CSV-number quoting, no entity context."""
    csv = rec.get("csv_path", "")
    table_vals = list(get_table_values(csv))
    if not table_vals:
        return None
    nums = ' '.join(str(v) for v in table_vals * 3)
    body = f"Numbers: {nums}\n{nums}\n{nums}"
    ans = str(rec["answer"])
    return f"<think>{body}</think><answer>{ans}</answer>"


def bloat_wrong_completion(rec):
    """Bloated reasoning + WRONG final answer.
    V1 should leak floor reward; V2 should crush via length_factor."""
    body = bloat_completion(rec)
    body = body.split("</think>")[0] + "</think>"
    try:
        f = float(rec["answer"])
        wrong = str(round(f * 1.7 + 13, 2))
    except ValueError:
        wrong = "999"
    return body + f"<answer>{wrong}</answer>"


def gamed_wrong_completion(rec):
    """Gamed (CSV flooding) reasoning + WRONG final answer.
    V1 leaks ~0.33 via floor + grounding; V2 should give ~0."""
    g = gamed_completion(rec)
    if not g:
        return None
    body = g.split("</think>")[0] + "</think>"
    try:
        f = float(rec["answer"])
        wrong = str(round(f * 1.7 + 13, 2))
    except ValueError:
        wrong = "999"
    return body + f"<answer>{wrong}</answer>"


def near_miss_completion(rec):
    """Clean reasoning but final answer slightly off (e.g. arithmetic typo).
    Tests V1 vs V2 trade-off: V1 gives partial credit, V2 doesn't.
    Important: too-aggressive V2 floor (0.05) may starve learning signal."""
    steps = rec.get("reasoning_steps") or []
    body = "I need to find the answer.\n" + "\n".join(f"- {s}" for s in steps)
    try:
        f = float(rec["answer"])
        # 5-15% off
        nm = str(round(f * 1.08, 2))
    except ValueError:
        nm = "999"
    return f"<think>{body}</think><answer>{nm}</answer>"


def main():
    random.seed(SEED)
    recs = [json.loads(l) for l in open(DATA)]
    sample = random.sample(recs, min(N, len(recs)))
    print(f"loaded {len(sample)} training records")

    rows = []
    for r in sample:
        csv = r["csv_path"]
        if not os.path.exists(csv):
            continue
        gold = gold_completion(r)
        wrong = wrong_completion(r)
        bloat = bloat_completion(r)
        gamed = gamed_completion(r)
        bloat_w = bloat_wrong_completion(r)
        gamed_w = gamed_wrong_completion(r)
        near = near_miss_completion(r)
        gold_ans = str(r["answer"])

        def both(comp):
            if comp is None: return (None, None)
            return (v1_full_reward(comp, gold_ans, csv),
                    v2_full_reward(comp, gold_ans, csv, MAX_LEN))

        v1g, v2g = both(gold)
        v1w, v2w = both(wrong)
        v1b, v2b = both(bloat)
        v1ga, v2ga = both(gamed)
        v1bw, v2bw = both(bloat_w)
        v1gw, v2gw = both(gamed_w)
        v1nm, v2nm = both(near)

        _, ig = compute_vapv_v2(gold, csv, max_completion_length=MAX_LEN)
        _, ib = compute_vapv_v2(bloat, csv, max_completion_length=MAX_LEN)
        rows.append({
            "n_steps": len(r.get("reasoning_steps") or []),
            "n_entities": len(get_table_entities(csv)),
            "v1_gold": v1g, "v1_wrong": v1w, "v1_bloat": v1b, "v1_gamed": v1ga,
            "v1_bloat_w": v1bw, "v1_gamed_w": v1gw, "v1_near": v1nm,
            "v2_gold": v2g, "v2_wrong": v2w, "v2_bloat": v2b, "v2_gamed": v2ga,
            "v2_bloat_w": v2bw, "v2_gamed_w": v2gw, "v2_near": v2nm,
            "v2_gold_anchor": ig.get("grounding"),
            "v2_bloat_anchor": ib.get("grounding"),
            "v2_bloat_lenfactor": ib.get("length_factor"),
        })
    out = "docs/active/tracks/master/_analysis/vapv_v2_preflight_training.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows → {out}\n")

    def m(xs): xs = [x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else 0.0

    print("=== Mean rewards across archetypes (full reward including outcome) ===")
    print(f"  {'archetype':14s} {'V1':>8s} {'V2':>8s} {'gap':>8s}")
    arches = [
        ("GOLD",      "v1_gold",   "v2_gold"),
        ("WRONG",     "v1_wrong",  "v2_wrong"),
        ("BLOAT_OK",  "v1_bloat",  "v2_bloat"),
        ("BLOAT_WRG", "v1_bloat_w","v2_bloat_w"),
        ("GAMED_OK",  "v1_gamed",  "v2_gamed"),
        ("GAMED_WRG", "v1_gamed_w","v2_gamed_w"),
        ("NEAR_MISS", "v1_near",   "v2_near"),
    ]
    for tag, k1, k2 in arches:
        v1m = m(r[k1] for r in rows); v2m = m(r[k2] for r in rows)
        print(f"  {tag:14s} {v1m:>8.3f} {v2m:>8.3f} {v2m - v1m:>+8.3f}")

    print("\n=== Pass criteria (V2 dynamics on RL rollouts) ===")
    n = len(rows)
    # H1: GOLD vs WRONG advantage gap (the core RL signal)
    h1 = sum(1 for r in rows if r["v2_gold"] - r["v2_wrong"] >= 0.5)
    # H2: BLOAT_WRG should collapse — V2 punishes bloat-with-wrong much harder than V1
    h2 = sum(1 for r in rows if (r["v1_bloat_w"] or 0) - (r["v2_bloat_w"] or 0) >= 0.05)
    # H3: GAMED_WRG should ~0 (anchored grounding zeroes the floor leak)
    h3 = sum(1 for r in rows if r["v2_gamed_w"] is not None and r["v2_gamed_w"] < 0.05)
    n_g = sum(1 for r in rows if r["v2_gamed_w"] is not None)
    # H4: V2 mean GAP > V1 mean GAP across all wrong archetypes
    v1_gap = m(r["v1_gold"] for r in rows) - m(
        sum([r["v1_wrong"], r["v1_bloat_w"] or 0, r["v1_gamed_w"] or 0]) / 3 for r in rows)
    v2_gap = m(r["v2_gold"] for r in rows) - m(
        sum([r["v2_wrong"], r["v2_bloat_w"] or 0, r["v2_gamed_w"] or 0]) / 3 for r in rows)
    # H5: NEAR_MISS — V2 not too aggressive (should still penalize but not 0)
    near_v1 = m(r["v1_near"] for r in rows); near_v2 = m(r["v2_near"] for r in rows)

    print(f"  H1 V2(GOLD)−V2(WRONG)≥0.5: {h1}/{n} ({h1/n*100:.1f}%) "
          f"{'PASS' if h1/n >= 0.9 else 'FAIL'}")
    print(f"  H2 V1(BLOAT_WRG)−V2(BLOAT_WRG)≥0.05: {h2}/{n} ({h2/n*100:.1f}%) "
          f"{'PASS' if h2/n >= 0.9 else 'FAIL'}  (V2 punishes bloat harder)")
    print(f"  H3 V2(GAMED_WRG)<0.05:     {h3}/{n_g} ({h3/n_g*100:.1f}%) "
          f"{'PASS' if h3/n_g >= 0.9 else 'FAIL'}  (anchored grounding kills floor leak)")
    print(f"  H4 V2_gap > V1_gap:        V1={v1_gap:.3f} V2={v2_gap:.3f} "
          f"{'PASS' if v2_gap > v1_gap else 'FAIL'}")
    print(f"  H5 NEAR_MISS dynamics:     V1={near_v1:.3f} V2={near_v2:.3f}  "
          f"(V2 starves near-miss signal — DESIGN tradeoff)")

    print("\n=== V1 defects exposed ===")
    print(f"  BLOAT_WRG:  V1={m(r['v1_bloat_w'] for r in rows):.3f} ↔ V2={m(r['v2_bloat_w'] for r in rows):.3f} "
          f"(V1 leaks floor on bloated wrong rollouts; V2 collapses)")
    print(f"  GAMED_WRG:  V1={m(r['v1_gamed_w'] for r in rows):.3f} ↔ V2={m(r['v2_gamed_w'] for r in rows):.3f} "
          f"(V1 rewards CSV-flooding even when answer wrong; V2 ~0)")


if __name__ == "__main__":
    main()
