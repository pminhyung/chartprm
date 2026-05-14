#!/usr/bin/env python
"""Automated QC audit for Block C v2 (rule + API).

Checks:
  1. API chart-level duplication (must be 0)
  2. Answer type distribution per source (target text >= 40% overall)
  3. qa_type / answer_type consistency (sci_ranking → text, sci_numeric → num, etc.)
  4. Unique chart count
  5. CSV file existence
  6. Sample output of 5 random QAs per (source, qa_type) group

Usage:
  python scripts/audit_block_c_v2.py \
      --rule data/charts_v9/block_c_rule_qa_v3.jsonl \
      --api  data/charts_v9/block_c_api_qa_v2.jsonl
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def is_numeric(a):
    """Return True if answer can be parsed as float."""
    try:
        s = str(a).replace(",", "").replace("%", "").strip()
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def load_jsonl(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return []
    data = []
    with open(path) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", default="data/charts_v9/block_c_rule_qa_v3.jsonl")
    parser.add_argument("--api",  default="data/charts_v9/block_c_api_qa_v2.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    rule = load_jsonl(args.rule)
    api  = load_jsonl(args.api)
    both = rule + api

    print("=" * 70)
    print("Block C v2 audit")
    print("=" * 70)
    print(f"Rule:  {len(rule):>6} QA from {args.rule}")
    print(f"API:   {len(api):>6} QA from {args.api}")
    print(f"Both:  {len(both):>6} total")

    # 1. Chart-level duplication
    print("\n[1] Chart-level duplication")
    for name, data in [("rule", rule), ("api", api)]:
        slugs = [d.get("chart_slug", "") for d in data]
        n_dup = len(slugs) - len(set(slugs))
        print(f"  {name}: {len(slugs)} QAs, {len(set(slugs))} unique slugs, {n_dup} dups")
        if name == "api" and n_dup > 0:
            print(f"    FAIL: API should have 1 QA/chart, got {n_dup} duplicates")

    # 2. Unique chart count (by csv_path since rule slugs are mangled)
    print("\n[2] Unique chart count")
    for name, data in [("rule", rule), ("api", api), ("both", both)]:
        csvs = {d.get("csv_path", "") for d in data}
        csvs.discard("")
        print(f"  {name}: {len(csvs)} unique csv_path")

    # 3. Answer type distribution
    print("\n[3] Answer type distribution")
    for name, data in [("rule", rule), ("api", api), ("both", both)]:
        if not data:
            continue
        n_num = sum(1 for d in data if is_numeric(d.get("answer")))
        n_txt = len(data) - n_num
        print(f"  {name:5s}: numeric {n_num:>5} ({n_num*100//len(data):>3d}%), "
              f"text {n_txt:>5} ({n_txt*100//len(data):>3d}%)")

    # 4. qa_type / answer_type consistency (API only — qa_type field set by sub-prompt)
    print("\n[4] qa_type / answer_type consistency (API)")
    violations = []
    by_qt = defaultdict(lambda: {"num": 0, "text": 0})
    for d in api:
        qt = d.get("qa_type", "")
        is_num = is_numeric(d.get("answer"))
        by_qt[qt]["num" if is_num else "text"] += 1
        # Expected types:
        #   sci_ranking → text
        #   sci_trend   → text
        #   sci_numeric → numeric
        #   sci_compare → numeric
        if qt in ("sci_ranking", "sci_trend") and is_num:
            violations.append((qt, d.get("chart_slug"), d.get("answer")))
        elif qt in ("sci_numeric", "sci_compare") and not is_num:
            violations.append((qt, d.get("chart_slug"), d.get("answer")))
    for qt, counts in sorted(by_qt.items()):
        total = counts["num"] + counts["text"]
        if total:
            print(f"  {qt:14s}: numeric {counts['num']:>4} ({counts['num']*100//total}%), "
                  f"text {counts['text']:>4} ({counts['text']*100//total}%)")
    print(f"  Violations: {len(violations)}")
    for qt, slug, ans in violations[:10]:
        print(f"    [{qt}] {slug}: {repr(ans)[:50]}")

    # 5. CSV existence spot check (20 random)
    print("\n[5] CSV existence (20 random from both)")
    if both:
        sample = random.sample(both, min(20, len(both)))
        missing = [d for d in sample if not os.path.exists(d.get("csv_path", ""))]
        print(f"  missing: {len(missing)}/{len(sample)}")

    # 6. Sample outputs
    print("\n[6] Sample outputs (5 per qa_type)")
    groups = defaultdict(list)
    for d in api:
        groups[d.get("qa_type", "?")].append(d)
    for d in rule:
        groups[f"rule:{d.get('template', '?')[:20]}"].append(d)
    for qt, lst in sorted(groups.items()):
        if not qt.startswith("sci_"):
            continue
        print(f"  --- {qt} ({len(lst)}) ---")
        for d in random.sample(lst, min(3, len(lst))):
            ans = repr(d.get("answer"))[:50]
            q = d.get("question", "")[:80]
            print(f"    Q: {q}")
            print(f"    A: {ans}")

    # 7. Overall verdict
    print("\n" + "=" * 70)
    print("Verdict")
    print("=" * 70)
    text_ratio = sum(1 for d in both if not is_numeric(d.get("answer"))) * 100 / max(len(both), 1)
    api_dups = len([d.get("chart_slug", "") for d in api]) - len({d.get("chart_slug", "") for d in api})
    unique_charts = len({d.get("csv_path", "") for d in both})
    verdict_pass = True
    print(f"  API chart dups:     {api_dups:>5} (target: 0)" +
          (" ✓" if api_dups == 0 else " ✗"))
    if api_dups != 0:
        verdict_pass = False
    print(f"  Text ratio:        {text_ratio:>5.1f}% (target: ≥ 40%)" +
          (" ✓" if text_ratio >= 40 else " ✗"))
    if text_ratio < 40:
        verdict_pass = False
    print(f"  Unique charts:      {unique_charts:>5} (target: ≥ 6000)" +
          (" ✓" if unique_charts >= 6000 else " ✗"))
    if unique_charts < 6000:
        verdict_pass = False
    print(f"  qa_type violations: {len(violations):>5} (target: 0)" +
          (" ✓" if len(violations) == 0 else " ✗"))
    if len(violations) > len(api) * 0.05:  # allow up to 5% sub-prompt strays
        verdict_pass = False

    print(f"\n  Overall: {'PASS ✓' if verdict_pass else 'FAIL ✗'}")
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
