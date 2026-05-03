"""Preflight VAPV-V2 reward against V1 on real R3/R4 inverse-flip completions.

Validates 4 hypotheses BEFORE training Row 4-V2:
  H1 (advantage-gap restored): correct − wrong reward gap > V1's gap
  H2 (length penalty bites): truncated/bloated completions get << reward
       than V1 awarded them
  H3 (anchored grounding gameability blocked): substring-match completions
       lose credit vs anchored matches
  H4 (multi-step UNIQUE wins survive): on the 77 R4-only UNIQUE samples,
       R4 reward > R3 reward by larger margin than V1
"""
import json
import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, '/ex_disk2/mhpark/poc/chartvr')

from code.rewards.rule_verifier_fast import (
    compute_process_reward_fast as v1_proc,
    compute_vapv_v2,
)
from chartvr.extraction import cerm_accuracy, _is_numeric_answer, relaxed_text_match

R3 = "results/row3_dapo_outcome_v2_nostop"
R4 = "results/row4_dapo_vapv_nostop"
BENCHES = ["chartqa_human", "chartqa_augmented", "chartqa_pro",
           "chartmuseum", "charxiv_reasoning"]
MAX_LEN = 4096


def load_bench(bench, dir_):
    out = {}
    with open(f"{dir_}/{bench}.jsonl") as fp:
        for ln in fp:
            d = json.loads(ln)
            out[d["sample_id"]] = d
    return out


def synth_completion(d):
    """Build a fake `completion` from eval JSONL (reasoning + answer tags)."""
    rc = d.get("reasoning_content") or ""
    pa = d.get("predicted_answer") or ""
    return f"<think>{rc}</think><answer>{pa}</answer>"


def find_csv(d, bench):
    """Best-effort CSV resolution. Eval JSONL has image_path; CSV is sibling.
    For benches without CSV (chartmuseum/charxiv), returns "" → V2 routes
    to outcome-only (the fix for the no-CSV silent-fallback bug)."""
    img = d.get("image_path", "")
    # synthetic charts have csv next to image
    if "/synthetic/png/" in img:
        return img.replace("/png/", "/tables/").rsplit(".", 1)[0] + ".csv"
    if "/owid/charts/" in img:
        return img.replace("/charts/", "/tables/").rsplit(".", 1)[0] + ".csv"
    return ""  # no CSV available


def v1_reward(d, csv):
    """Mirror reward_conditional_v2 from train_grpo_dapo.py."""
    resp = synth_completion(d)
    pred = d.get("predicted_answer", "")
    gold = str(d.get("gold_answer", ""))
    if not _is_numeric_answer(gold) or not csv or not os.path.exists(csv):
        return relaxed_text_match(pred, gold), {"branch": "text_or_nocsv"}
    r_acc = cerm_accuracy(pred, gold)
    if r_acc >= 0.95:
        return 1.0, {"branch": "correct", "r_acc": r_acc}
    proc = v1_proc(resp, csv)
    return min(1.0, r_acc + 0.3 * proc), {"branch": "wrong_proc",
                                          "r_acc": r_acc, "proc": proc}


def v2_reward(d, csv, is_multi_step=True):
    resp = synth_completion(d)
    pred = d.get("predicted_answer", "")
    gold = str(d.get("gold_answer", ""))
    if not _is_numeric_answer(gold):
        return relaxed_text_match(pred, gold), {"branch": "text"}
    if not csv or not os.path.exists(csv) or not is_multi_step:
        r_acc = cerm_accuracy(pred, gold)
        return (1.0 if r_acc >= 0.95 else 0.0), {"branch": "outcome_only",
                                                 "r_acc": r_acc}
    r_acc = cerm_accuracy(pred, gold)
    if r_acc >= 0.95:
        return 1.0, {"branch": "correct", "r_acc": r_acc}
    proc, info = compute_vapv_v2(resp, csv, max_completion_length=MAX_LEN)
    return 0.05 * proc, {"branch": "wrong_proc", "r_acc": r_acc,
                         "proc": proc, **info}


def main():
    # Load R3, R4
    r3, r4 = {}, {}
    for b in BENCHES:
        r3[b] = load_bench(b, R3)
        r4[b] = load_bench(b, R4)

    # Use union of all common samples (preflight is over real completions)
    rows = []
    for b in BENCHES:
        common = sorted(set(r3[b]) & set(r4[b]))
        for sid in common:
            for who, src in [("r3", r3[b][sid]), ("r4", r4[b][sid])]:
                csv = find_csv(src, b)
                v1, v1i = v1_reward(src, csv)
                v2, v2i = v2_reward(src, csv, is_multi_step=True)
                rows.append({
                    "bench": b, "sample_id": sid, "who": who,
                    "csv_present": bool(csv and os.path.exists(csv)),
                    "accuracy": float(src.get("accuracy") or 0),
                    "v1_reward": v1, "v2_reward": v2,
                    "think_words": v2i.get("think_len_words", 0),
                    "length_factor": v2i.get("length_factor", 1.0),
                    "v2_branch": v2i.get("branch"),
                    "v2_proc": v2i.get("proc"),
                    "v1_branch": v1i.get("branch"),
                    "v1_proc": v1i.get("proc"),
                })
    out_path = "docs/active/tracks/master/_analysis/vapv_v2_preflight.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows → {out_path}")

    # ── Aggregations ──
    import statistics as st

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    print("\n=== Overall (R3 + R4 combined) ===")
    correct = [r for r in rows if r["accuracy"] >= 1.0]
    wrong = [r for r in rows if r["accuracy"] < 1.0]
    print(f"  V1: correct {mean(r['v1_reward'] for r in correct):.3f}  "
          f"wrong {mean(r['v1_reward'] for r in wrong):.3f}  "
          f"GAP {mean(r['v1_reward'] for r in correct) - mean(r['v1_reward'] for r in wrong):.3f}")
    print(f"  V2: correct {mean(r['v2_reward'] for r in correct):.3f}  "
          f"wrong {mean(r['v2_reward'] for r in wrong):.3f}  "
          f"GAP {mean(r['v2_reward'] for r in correct) - mean(r['v2_reward'] for r in wrong):.3f}")

    print("\n=== By bench (advantage gap) ===")
    print(f"  {'bench':25s} {'V1_gap':>8s} {'V2_gap':>8s} {'V2-V1':>8s}")
    for b in BENCHES:
        bc = [r for r in rows if r["bench"] == b and r["accuracy"] >= 1.0]
        bw = [r for r in rows if r["bench"] == b and r["accuracy"] < 1.0]
        v1g = mean(r['v1_reward'] for r in bc) - mean(r['v1_reward'] for r in bw)
        v2g = mean(r['v2_reward'] for r in bc) - mean(r['v2_reward'] for r in bw)
        print(f"  {b:25s} {v1g:>8.3f} {v2g:>8.3f} {v2g - v1g:>+8.3f}")

    print("\n=== Length scaling effect (V2 only, wrong samples) ===")
    bw = [r for r in rows if r["accuracy"] < 1.0 and r["v2_branch"] == "wrong_proc"]
    if bw:
        # Quartile by think_words
        bw_sorted = sorted(bw, key=lambda r: r["think_words"])
        q = [bw_sorted[i*len(bw_sorted)//4:(i+1)*len(bw_sorted)//4] for i in range(4)]
        print(f"  {'quartile':10s} {'words_avg':>10s} {'len_factor':>11s} {'V1_proc':>9s} {'V2_proc':>9s} {'V1_rew':>8s} {'V2_rew':>8s}")
        for i, qi in enumerate(q):
            if not qi: continue
            print(f"  Q{i+1:<9d} {mean(r['think_words'] for r in qi):>10.0f} "
                  f"{mean(r['length_factor'] for r in qi):>11.3f} "
                  f"{mean((r.get('v1_proc') or 0) for r in qi):>9.3f} "
                  f"{mean((r.get('v2_proc') or 0) for r in qi):>9.3f} "
                  f"{mean(r['v1_reward'] for r in qi):>8.3f} "
                  f"{mean(r['v2_reward'] for r in qi):>8.3f}")

    print("\n=== UNIQUE-flip subset (n=77 R4-only UNIQUE) ===")
    classified = {}
    cls_path = ("docs/active/tracks/master/_analysis/"
                "r4only_sample196_classified.jsonl")
    if os.path.exists(cls_path):
        with open(cls_path) as fp:
            for ln in fp:
                c = json.loads(ln)
                classified[(c["bench"], c["sample_id"])] = c["_vapv_credit"]
    unique_pairs = [k for k, v in classified.items() if v == "UNIQUE"]
    # Compute reward delta R4-R3 per UNIQUE sample under V1 vs V2
    by_sid = defaultdict(dict)
    for r in rows:
        by_sid[(r["bench"], r["sample_id"])][r["who"]] = r
    deltas_v1, deltas_v2 = [], []
    for k in unique_pairs:
        if "r3" in by_sid[k] and "r4" in by_sid[k]:
            deltas_v1.append(by_sid[k]["r4"]["v1_reward"] - by_sid[k]["r3"]["v1_reward"])
            deltas_v2.append(by_sid[k]["r4"]["v2_reward"] - by_sid[k]["r3"]["v2_reward"])
    if deltas_v1:
        print(f"  UNIQUE samples covered: {len(deltas_v1)}/{len(unique_pairs)}")
        print(f"  V1 mean (R4_rew - R3_rew): {mean(deltas_v1):+.3f}")
        print(f"  V2 mean (R4_rew - R3_rew): {mean(deltas_v2):+.3f}")
        print(f"  V2 strengthens UNIQUE-win signal by {mean(deltas_v2) - mean(deltas_v1):+.3f}")

    print("\n=== INCIDENTAL-flip subset (n=91 INCIDENTAL — V2 should NOT credit these as much) ===")
    incidental_pairs = [k for k, v in classified.items() if v == "INCIDENTAL"]
    deltas_v1, deltas_v2 = [], []
    for k in incidental_pairs:
        if "r3" in by_sid[k] and "r4" in by_sid[k]:
            deltas_v1.append(by_sid[k]["r4"]["v1_reward"] - by_sid[k]["r3"]["v1_reward"])
            deltas_v2.append(by_sid[k]["r4"]["v2_reward"] - by_sid[k]["r3"]["v2_reward"])
    if deltas_v1:
        print(f"  INCIDENTAL samples covered: {len(deltas_v1)}")
        print(f"  V1 mean (R4_rew - R3_rew): {mean(deltas_v1):+.3f}")
        print(f"  V2 mean (R4_rew - R3_rew): {mean(deltas_v2):+.3f}")

    print("\n=== R3-only flips subset (n=200 — VAPV broke these) ===")
    r3_classified = {}
    r3_path = "docs/active/tracks/master/_analysis/_sample200_classified.jsonl"
    if os.path.exists(r3_path):
        with open(r3_path) as fp:
            for ln in fp:
                c = json.loads(ln)
                r3_classified[(c["bench"], c["sample_id"])] = c["_mode"]
    deltas_v1, deltas_v2 = [], []
    bloat_pairs = []
    for k, mode in r3_classified.items():
        if "r3" in by_sid[k] and "r4" in by_sid[k]:
            d_v1 = by_sid[k]["r4"]["v1_reward"] - by_sid[k]["r3"]["v1_reward"]
            d_v2 = by_sid[k]["r4"]["v2_reward"] - by_sid[k]["r3"]["v2_reward"]
            deltas_v1.append(d_v1)
            deltas_v2.append(d_v2)
            if mode == "BLOAT_TRUNC":
                bloat_pairs.append((d_v1, d_v2))
    if deltas_v1:
        print(f"  R3-only covered: {len(deltas_v1)}/{len(r3_classified)}")
        print(f"  V1 mean (R4_rew - R3_rew): {mean(deltas_v1):+.3f}  (negative = VAPV correctly penalizes R4)")
        print(f"  V2 mean (R4_rew - R3_rew): {mean(deltas_v2):+.3f}")
    if bloat_pairs:
        v1b = mean(d[0] for d in bloat_pairs); v2b = mean(d[1] for d in bloat_pairs)
        print(f"  BLOAT_TRUNC subset (n={len(bloat_pairs)}): V1={v1b:+.3f}  V2={v2b:+.3f} "
              f"(more negative = V2 punishes R4 bloat better)")


if __name__ == "__main__":
    main()
