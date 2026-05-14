"""Audit distilled SFT data produced by teacher_distill.py.

Computes:
- Count, per-source distribution, per-origin distribution
- reasoning_content length stats (chars and approx tokens)
- content length stats
- assistant_text length stats (chars and approx tokens)
- finish_reason distribution
- gold_answer shape (text vs numeric)
- Runaway rate (finish_reason=length)
- Sample-level dump for manual inspection (top N by length, random N)

Usage:
    python scripts/audit_distill.py data/pilot_hq.jsonl
    python scripts/audit_distill.py data/sft_hq.jsonl --manual 30
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100)))
    return sorted_vals[k]


def approx_tokens(text):
    """Approximate token count. 1 token ~ 3.5 chars for English, lower for CJK.
    For audit purposes this is rough but consistent — actual tokenizer used at
    training time (Qwen) would give exact values, but distribution shape is
    preserved by char/4 heuristic."""
    return max(1, len(text) // 4)


def hist(vals, buckets):
    counter = Counter()
    for v in vals:
        for lo, hi in buckets:
            if lo <= v < hi:
                counter[(lo, hi)] += 1
                break
        else:
            counter[("inf", "inf")] += 1
    return counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to distilled jsonl (e.g. data/pilot_hq.jsonl)")
    ap.add_argument("--manual", type=int, default=20,
                    help="number of random samples to dump for manual inspection")
    ap.add_argument("--top", type=int, default=5,
                    help="number of longest samples to dump")
    ap.add_argument("--out", default="",
                    help="optional output path for the audit report (markdown)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print(f"ERROR: {args.input} is empty")
        sys.exit(1)

    n = len(rows)
    src_count = Counter(r.get("source", "") for r in rows)
    origin_count = Counter(r.get("origin", "") for r in rows)
    finish_count = Counter(r.get("teacher_finish_reason", "") for r in rows)

    reasoning_lens = [r.get("teacher_reasoning_len", 0) for r in rows]
    content_lens = [r.get("teacher_content_len", 0) for r in rows]
    assistant_text_lens = [len(r.get("assistant_text", "")) for r in rows]
    assistant_text_tok = [approx_tokens(r.get("assistant_text", "")) for r in rows]

    reasoning_lens_s = sorted(reasoning_lens)
    content_lens_s = sorted(content_lens)
    assistant_text_lens_s = sorted(assistant_text_lens)
    assistant_text_tok_s = sorted(assistant_text_tok)

    # Format diagnostics on assistant_text
    has_think_open = sum(1 for r in rows if "<think>" in r.get("assistant_text", ""))
    has_think_close = sum(1 for r in rows if "</think>" in r.get("assistant_text", ""))
    has_answer_open = sum(1 for r in rows if "<answer>" in r.get("assistant_text", ""))
    has_answer_close = sum(1 for r in rows if "</answer>" in r.get("assistant_text", ""))
    double_think = sum(1 for r in rows if r.get("assistant_text", "").count("<think>") > 1)
    double_answer = sum(1 for r in rows if r.get("assistant_text", "").count("<answer>") > 1)

    # Answer type
    def _is_num(a):
        try:
            float(str(a).replace(",", "").replace("%", ""))
            return True
        except Exception:
            return False
    num_count = sum(1 for r in rows if _is_num(r.get("answer", "")))

    # Runaway: finish_reason=length means teacher hit the ceiling = runaway
    runaway = finish_count.get("length", 0)

    report = []
    def p(s=""):
        report.append(s)
        print(s)

    p(f"# Distill Audit — {args.input}")
    p()
    p(f"**Total samples**: {n}")
    p(f"**Runaway (finish_reason=length)**: {runaway} ({runaway*100/n:.1f}%)")
    p(f"**Numeric answers**: {num_count} ({num_count*100/n:.1f}%)")
    p(f"**Text answers**: {n - num_count} ({(n-num_count)*100/n:.1f}%)")
    p()
    p("## Source distribution")
    for k, v in src_count.most_common():
        p(f"  {k}: {v} ({v*100/n:.1f}%)")
    p()
    p("## Origin distribution (v9 / v8 / api)")
    for k, v in origin_count.most_common():
        p(f"  {k}: {v} ({v*100/n:.1f}%)")
    p()
    p("## Finish reason")
    for k, v in finish_count.most_common():
        p(f"  {k}: {v} ({v*100/n:.1f}%)")
    p()
    p("## Length stats (chars)")
    p(f"{'field':<22s} {'min':>8s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s} {'mean':>8s}")
    def row(name, arr_sorted):
        mean = sum(arr_sorted) / max(len(arr_sorted), 1)
        p(f"{name:<22s} {arr_sorted[0]:>8d} {percentile(arr_sorted,50):>8d} {percentile(arr_sorted,90):>8d} {percentile(arr_sorted,95):>8d} {percentile(arr_sorted,99):>8d} {arr_sorted[-1]:>8d} {mean:>8.0f}")
    row("reasoning_content", reasoning_lens_s)
    row("content", content_lens_s)
    row("assistant_text", assistant_text_lens_s)
    p()
    p("## Assistant text (approx tokens = chars/4)")
    p(f"{'field':<22s} {'min':>8s} {'p50':>8s} {'p90':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s} {'mean':>8s}")
    row("assistant_text_tok", assistant_text_tok_s)
    p()
    p("## Format integrity")
    p(f"  has <think>     : {has_think_open}/{n} ({has_think_open*100/n:.1f}%)")
    p(f"  has </think>    : {has_think_close}/{n} ({has_think_close*100/n:.1f}%)")
    p(f"  has <answer>    : {has_answer_open}/{n} ({has_answer_open*100/n:.1f}%)")
    p(f"  has </answer>   : {has_answer_close}/{n} ({has_answer_close*100/n:.1f}%)")
    p(f"  duplicate <think> : {double_think}")
    p(f"  duplicate <answer>: {double_answer}")
    p()
    p("## Training-length gate (char → approx tokens)")
    thresholds_tok = [2048, 3072, 4096, 6144, 8192, 12288, 16384]
    for t in thresholds_tok:
        keep = sum(1 for x in assistant_text_tok if x <= t)
        p(f"  fit ≤ {t:>5d} tokens: {keep}/{n} ({keep*100/n:.1f}%)")
    p()
    p(f"## Top {args.top} longest (assistant_text)")
    top_idx = sorted(range(n), key=lambda i: -assistant_text_lens[i])[: args.top]
    for rank, i in enumerate(top_idx, 1):
        r = rows[i]
        p(f"  #{rank} src={r.get('source',''):20s} src_chars={assistant_text_lens[i]} tok≈{assistant_text_tok[i]} finish={r.get('teacher_finish_reason','')}")
        p(f"       q={(r.get('question','')[:80])}")
        p(f"       gold={r.get('answer','')}  pred={r.get('teacher_pred','')}")
    p()
    p(f"## Random {args.manual} samples (for manual inspection)")
    random.seed(0)
    sample_idx = random.sample(range(n), min(args.manual, n))
    for rank, i in enumerate(sample_idx, 1):
        r = rows[i]
        at = r.get("assistant_text", "")
        head = at[:200].replace("\n", " / ")
        tail = at[-200:].replace("\n", " / ") if len(at) > 200 else ""
        p(f"  #{rank} src={r.get('source',''):20s} tok≈{assistant_text_tok[i]:5d} finish={r.get('teacher_finish_reason',''):8s} gold={r.get('answer','')}")
        p(f"       q={(r.get('question','')[:100])}")
        p(f"       at_head={head}")
        if tail:
            p(f"       at_tail=...{tail}")
    p()

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
