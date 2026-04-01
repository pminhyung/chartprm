"""Rescore existing eval JSONL files with improved answer extraction.

Extracts answers from reasoning_content when content is empty.
Marks samples with both empty as 'needs_rerun'.
"""
import json
import os
import re
import sys


def relaxed_accuracy(pred, gold):
    """ChartQA relaxed accuracy with unit stripping (v3)."""
    units_re = r'\s*(billion|million|thousand|trillion|percent|k|K|M|B|T|tons?|t|kg|lbs?|GiB|MB|GB)\b'
    p = re.sub(units_re, '', pred.strip(), flags=re.I).strip()
    g = re.sub(units_re, '', gold.strip(), flags=re.I).strip()
    p = re.sub(r'[,%$]', '', p).rstrip('.')
    g = re.sub(r'[,%$]', '', g).rstrip('.')
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower().strip() == g.lower().strip() else 0.0


def _normalize(answer):
    answer = answer.replace("**", "").strip()
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()
    if len(answer) < 80 and answer.endswith('.'):
        answer = answer[:-1].strip()
    prefixes = [
        "Final answer:", "The answer is", "Answer:", "Therefore,",
        "So the answer is", "Thus,", "Hence,", "In conclusion,",
        "There are", "There is", "The value is", "The difference is",
        "It is", "The total is", "The average is",
    ]
    for p in prefixes:
        if answer.lower().startswith(p.lower()):
            answer = answer[len(p):].strip()
            break
    # Handle "X to Y" ranges
    range_match = re.match(r'^(\d{4})\s+to\s+(\d{4})$', answer.strip())
    if range_match:
        return range_match.group(2)
    ans_lower = answer.lower()
    if 'no' in ans_lower.split()[:3] or ans_lower.startswith('no'):
        return 'No'
    if 'yes' in ans_lower.split()[:3] or ans_lower.startswith('yes'):
        return 'Yes'
    # Strip units
    answer = re.sub(r'\s*(billion|million|thousand|trillion|percent)\b', '', answer, flags=re.I).strip()
    answer = re.sub(r'\s*[kKMBT]$', '', answer).strip()
    if len(answer) > 15:
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', answer)
        if nums:
            return nums[-1].replace(',', '')
    return answer


def extract_answer(response):
    if not response:
        return ""
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return _normalize(m.group(1).strip())
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            m2 = re.search(r'<answer>(.*?)</answer>', after, re.DOTALL | re.IGNORECASE)
            if m2:
                return _normalize(m2.group(1).strip())
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                return _normalize(lines[-1])
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return _normalize(lines[-1]) if lines else ""


def rescore_file(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))

    original_correct = sum(1 for r in results if r.get('accuracy', 0) == 1.0)
    needs_rerun = 0
    rescored = 0
    new_correct = 0

    for r in results:
        content = (r.get('content', '') or '').strip()
        reasoning = (r.get('reasoning_content', '') or '').strip()

        # Try content first, then reasoning as fallback
        if content:
            pred = extract_answer(content)
        elif reasoning:
            pred = extract_answer(reasoning)
            rescored += 1
        else:
            pred = ""
            needs_rerun += 1

        gold = r['gold_answer']
        acc = relaxed_accuracy(pred, gold)
        r['predicted_answer_v2'] = pred
        r['accuracy_v2'] = acc
        new_correct += acc

    # Write rescored file
    out_path = path.replace('.jsonl', '_rescored.jsonl')
    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total = len(results)
    return {
        'total': total,
        'original_acc': original_correct / total if total else 0,
        'rescored_acc': new_correct / total if total else 0,
        'rescored_from_reasoning': rescored,
        'needs_rerun': needs_rerun,
    }


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'results/v7'

    print(f"{'Row':<12} {'Bench':<22} {'Original':>10} {'Rescored':>10} {'FromReas':>10} {'NeedRerun':>10}")
    print("-" * 76)

    for row in ['zeroshot', 'row_a', 'row_b']:
        row_dir = os.path.join(base, row)
        if not os.path.exists(row_dir):
            continue
        for bench_file in sorted(os.listdir(row_dir)):
            if not bench_file.endswith('.jsonl') or '_rescored' in bench_file:
                continue
            path = os.path.join(row_dir, bench_file)
            bench = bench_file.replace('.jsonl', '')
            stats = rescore_file(path)
            print(f"{row:<12} {bench:<22} {stats['original_acc']:>9.1%} {stats['rescored_acc']:>9.1%} "
                  f"{stats['rescored_from_reasoning']:>10d} {stats['needs_rerun']:>10d}")


if __name__ == '__main__':
    main()
