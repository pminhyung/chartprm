"""Audit sft_hq_lite.jsonl distribution (source, length, reasoning presence).

Separate from audit_distill.py (which expects teacher-distilled fields).
"""
import argparse
import json
import os
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", default="data/sft_hq_lite.jsonl", nargs="?")
    ap.add_argument("--manual", type=int, default=10)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = len(rows)
    print(f"Total: {n}")

    # Basic checks
    src = Counter()
    origin = Counter()
    ans_type = Counter()
    has_reason = Counter()
    missing_image = 0
    reason_lens = []
    q_lens = []

    for r in rows:
        src[r.get("source", "")] += 1
        origin[r.get("origin", "")] += 1
        at = r.get("answer_type", "")
        ans_type[at or "unspecified"] += 1
        q_lens.append(len(r.get("question", "")))

        if not os.path.exists(r.get("image_path", "")):
            missing_image += 1

        rs = r.get("reasoning_steps", "")
        if isinstance(rs, list):
            rs_str = "\n".join(rs)
        else:
            rs_str = str(rs or "")
        if rs_str:
            has_reason["with"] += 1
            reason_lens.append(len(rs_str))
        else:
            has_reason["without"] += 1

    print(f"\nImage existence: {n - missing_image}/{n} OK ({missing_image} missing)")
    print(f"\nSource:")
    for k, v in src.most_common():
        print(f"  {k:20s} {v:5d} ({v*100/n:.1f}%)")
    print(f"\nOrigin:")
    for k, v in origin.most_common():
        print(f"  {k:20s} {v:5d} ({v*100/n:.1f}%)")
    print(f"\nAnswer type:")
    for k, v in ans_type.most_common():
        print(f"  {k:20s} {v:5d} ({v*100/n:.1f}%)")

    print(f"\nReasoning presence:")
    print(f"  with_reasoning:    {has_reason['with']:5d} ({has_reason['with']*100/n:.1f}%)")
    print(f"  without_reasoning: {has_reason['without']:5d} ({has_reason['without']*100/n:.1f}%)")

    if reason_lens:
        reason_lens.sort()
        m = len(reason_lens)
        def p(a,q): return a[min(int(len(a)*q/100), len(a)-1)]
        print(f"\nReasoning char length (non-empty only, n={m}):")
        print(f"  min={reason_lens[0]}  p50={p(reason_lens,50)}  p90={p(reason_lens,90)}  p95={p(reason_lens,95)}  p99={p(reason_lens,99)}  max={reason_lens[-1]}  mean={sum(reason_lens)/m:.0f}")

    q_lens.sort()
    def p(a,q): return a[min(int(len(a)*q/100), len(a)-1)]
    print(f"\nQuestion char length (n={n}):")
    print(f"  min={q_lens[0]}  p50={p(q_lens,50)}  p95={p(q_lens,95)}  max={q_lens[-1]}  mean={sum(q_lens)/n:.0f}")

    # Manual sample dump
    print(f"\n{args.manual} random samples:")
    random.seed(0)
    for i in random.sample(range(n), min(args.manual, n)):
        r = rows[i]
        rs = r.get("reasoning_steps", "")
        if isinstance(rs, list):
            rs = "\n".join(rs)
        rs_str = str(rs or "").replace("\n", " / ")
        print(f"  [{i}] src={r.get('source',''):20s} ans_type={r.get('answer_type','')[:10]:10s}")
        print(f"       q={r.get('question','')[:110]}")
        print(f"       ans={r.get('answer','')}  reason={rs_str[:120]}")


if __name__ == "__main__":
    main()
