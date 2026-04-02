"""Rescore existing eval JSONL files with improved answer extraction.

Extracts answers from reasoning_content when content is empty.
Marks samples with both empty as 'needs_rerun'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.extraction import relaxed_accuracy, extract_answer


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
