"""Rescore eval JSONL with extraction v5 (in-think filter).

For each result file in <input_dir>/<row>/<bench>.jsonl:
- Reconstruct full response = <think>{reasoning_content}</think>{content}
- Run extract_answer (v5) — filters in-think example mentions
- Compute relaxed_accuracy
- Write to <input_dir>/<row>/<bench>_extv5.jsonl with fields:
    sample_id, question, gold_answer, image_path, content, reasoning_content,
    predicted_answer (v5), accuracy (v5), extraction_version, original_accuracy

Usage:
    python scripts/rescore_eval_v5.py <results_root_or_specific_dir> [<row1> <row2>...]

Examples:
    # Single dir (no row sub-structure)
    python scripts/rescore_eval_v5.py results/row4_dapo_vapv

    # Multi-row dir
    python scripts/rescore_eval_v5.py results/v7 row_a row_b zeroshot

If no rows specified, treats input_dir itself as the row directory.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.extraction import relaxed_accuracy, extract_answer


def reconstruct_response(content, reasoning):
    """Build the full assistant response with think tags so v5 extraction works."""
    content = (content or '').strip()
    reasoning = (reasoning or '').strip()
    if reasoning and content:
        return f"<think>\n{reasoning}\n</think>\n{content}"
    if reasoning and not content:
        # Reasoning truncated mid-thought (no </think> emitted, no answer)
        # v5 returns "" for this case (correct).
        return f"<think>\n{reasoning}"
    if content and not reasoning:
        return content
    return ""


def rescore_file(path, out_path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))

    n_total = len(results)
    n_orig_correct = 0
    n_v5_correct = 0
    n_orig_only = 0  # was correct, now wrong (false-positive removed)
    n_v5_only = 0    # was wrong, now correct (only possible if extract_answer recovered something)
    n_empty_pred_v5 = 0

    for r in results:
        orig_acc = float(r.get('accuracy', 0) or 0)
        gold = r['gold_answer']
        full = reconstruct_response(r.get('content'), r.get('reasoning_content'))
        pred = extract_answer(full)
        acc = relaxed_accuracy(pred, gold) if pred else 0.0
        if not pred:
            n_empty_pred_v5 += 1

        n_orig_correct += int(orig_acc == 1.0)
        n_v5_correct += int(acc == 1.0)
        if orig_acc == 1.0 and acc != 1.0:
            n_orig_only += 1
        if orig_acc != 1.0 and acc == 1.0:
            n_v5_only += 1

        # Write minimal but complete record
        out_record = {
            'sample_id': r.get('sample_id'),
            'question': r.get('question'),
            'gold_answer': gold,
            'image_path': r.get('image_path'),
            'content': r.get('content'),
            'reasoning_content': r.get('reasoning_content'),
            'predicted_answer': pred,
            'accuracy': acc,
            'extraction_version': 'v5',
            'original_predicted_answer': r.get('predicted_answer'),
            'original_accuracy': orig_acc,
        }
        results_out = out_record  # noqa
    # Stream-write
    with open(out_path, 'w') as f:
        for r in results:
            orig_acc = float(r.get('accuracy', 0) or 0)
            gold = r['gold_answer']
            full = reconstruct_response(r.get('content'), r.get('reasoning_content'))
            pred = extract_answer(full)
            acc = relaxed_accuracy(pred, gold) if pred else 0.0
            out_record = {
                'sample_id': r.get('sample_id'),
                'question': r.get('question'),
                'gold_answer': gold,
                'image_path': r.get('image_path'),
                'content': r.get('content'),
                'reasoning_content': r.get('reasoning_content'),
                'predicted_answer': pred,
                'accuracy': acc,
                'extraction_version': 'v5',
                'original_predicted_answer': r.get('predicted_answer'),
                'original_accuracy': orig_acc,
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + '\n')

    return {
        'total': n_total,
        'orig_acc': n_orig_correct / n_total if n_total else 0,
        'v5_acc': n_v5_correct / n_total if n_total else 0,
        'flips_lost': n_orig_only,
        'flips_gained': n_v5_only,
        'empty_pred_v5': n_empty_pred_v5,
    }


def process_dir(row_dir):
    print(f"\n=== {row_dir} ===")
    print(f"{'Bench':<22} {'N':>6} {'OrigAcc':>9} {'V5Acc':>9} {'Δ':>7} {'-Lost':>6} {'+Gain':>6} {'Empty':>6}")
    print("-" * 86)
    bench_results = {}
    for f in sorted(os.listdir(row_dir)):
        if not f.endswith('.jsonl'):
            continue
        if any(suffix in f for suffix in ('_rescored', '_extv', '_v2', '_v3', '_v4', '_v5')):
            continue
        path = os.path.join(row_dir, f)
        bench = f.replace('.jsonl', '')
        out_path = os.path.join(row_dir, f.replace('.jsonl', '_extv5.jsonl'))
        stats = rescore_file(path, out_path)
        delta = (stats['v5_acc'] - stats['orig_acc']) * 100
        bench_results[bench] = stats
        print(f"{bench:<22} {stats['total']:>6} {stats['orig_acc']:>8.2%} {stats['v5_acc']:>8.2%} "
              f"{delta:>+6.2f} {stats['flips_lost']:>6} {stats['flips_gained']:>6} {stats['empty_pred_v5']:>6}")
    if bench_results:
        avg_orig = sum(s['orig_acc'] for s in bench_results.values()) / len(bench_results) * 100
        avg_v5 = sum(s['v5_acc'] for s in bench_results.values()) / len(bench_results) * 100
        print(f"{'AVG':<22} {'':>6} {avg_orig:>8.2f} {avg_v5:>8.2f} {avg_v5 - avg_orig:>+6.2f}")
    return bench_results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    rows = sys.argv[2:]
    if not rows:
        # Treat root itself as the row directory
        process_dir(root)
    else:
        for row in rows:
            row_dir = os.path.join(root, row)
            if os.path.isdir(row_dir):
                process_dir(row_dir)
            else:
                print(f"(skip) {row_dir} not found")


if __name__ == '__main__':
    main()
