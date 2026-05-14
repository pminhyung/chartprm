#!/usr/bin/env python
"""Filter out problematic Block C QA records.

Issues detected:
1. block_c_rule: 'x' used as label column (206 numeric entity names, 59 x-col)
2. block_c_api: trend/pattern questions with numeric answers (612)
3. block_d_rule: same-entity diff questions (38)

Usage:
  python scripts/filter_block_c_issues.py
"""
import json
import re
import os


def is_block_c_rule_bad(r):
    q = r.get('question', '').lower()
    # "between 15.122 and 4.018" → numeric entity
    m = re.search(r'between ([-\d.]+) and ([-\d.]+)', q)
    if m:
        try:
            float(m.group(1))
            float(m.group(2))
            return True, 'numeric_entity'
        except Exception:
            pass
    # "maximum x" "total x" etc — x as column
    if re.search(r'\bof x\b|\btotal x\b|\bmaximum x\b|\bminimum x\b|\baverage x\b|\bsum x\b|\bx column\b', q):
        return True, 'x_as_column'
    # "Is 8.67's Activity greater than 19.642's?" — numeric as entity in yes/no
    m = re.search(r"is ([-\d.]+)'s .+? (greater|less) than ([-\d.]+)", q)
    if m:
        try:
            float(m.group(1))
            float(m.group(3))
            return True, 'numeric_entity_yesno'
        except Exception:
            pass
    return False, None


def is_block_c_api_bad(r):
    q = r.get('question', '').lower()
    a = str(r.get('answer', ''))
    # Trend/pattern questions should not have pure numeric answer
    if 'trend' in q or 'pattern observed' in q or 'what is observed' in q:
        # If answer is pure numeric, it's inconsistent
        try:
            float(a.replace(',', '').replace('%', ''))
            return True, 'trend_numeric_answer'
        except Exception:
            return False, None
    return False, None


def is_block_d_rule_bad(r):
    q = r.get('question', '').lower()
    # "difference between A and A" — same entity
    m = re.search(r'between (.+?) and (.+?)(?:\?| in )', q)
    if m:
        e1 = m.group(1).strip().strip("'\"")
        e2 = m.group(2).strip().strip("'\"")
        if e1 == e2 and len(e1) > 0:
            return True, 'same_entity'
    return False, None


def filter_file(input_path, output_path, checker):
    if not os.path.exists(input_path):
        print(f"  Skip (not found): {input_path}")
        return 0, 0

    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    total = len(records)
    clean = []
    issues = {}

    for r in records:
        bad, reason = checker(r)
        if bad:
            issues[reason] = issues.get(reason, 0) + 1
        else:
            clean.append(r)

    removed = total - len(clean)
    with open(output_path, 'w') as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')

    print(f"  {input_path}")
    print(f"    Total: {total}, Kept: {len(clean)}, Removed: {removed}")
    for reason, count in issues.items():
        print(f"      {reason}: {count}")
    return total, removed


def main():
    print("=== Filtering Block C/D problematic records ===\n")

    total_before = 0
    total_removed = 0

    # Block C rule
    t, r = filter_file(
        'data/charts_v9/block_c_rule_qa_v2.jsonl',
        'data/charts_v9/block_c_rule_qa_v2_filtered.jsonl',
        is_block_c_rule_bad,
    )
    total_before += t
    total_removed += r

    # Block C API
    t, r = filter_file(
        'data/charts_v9/block_c_api_qa.jsonl',
        'data/charts_v9/block_c_api_qa_filtered.jsonl',
        is_block_c_api_bad,
    )
    total_before += t
    total_removed += r

    # Block D rule
    t, r = filter_file(
        'data/charts_v9/block_d_rule_qa_v2.jsonl',
        'data/charts_v9/block_d_rule_qa_v2_filtered.jsonl',
        is_block_d_rule_bad,
    )
    total_before += t
    total_removed += r

    print(f"\n=== Summary ===")
    print(f"Total before: {total_before}")
    print(f"Total removed: {total_removed} ({total_removed/max(total_before,1)*100:.1f}%)")
    print(f"Total kept: {total_before - total_removed}")


if __name__ == '__main__':
    main()
