"""Assemble v8_improved SFT dataset: balanced 30K with 100% reasoning.

Strategy (based on sft_regression_diagnosis.md):
- chartqa_train: 10K selected (teacher reasoning required)
- v8 rest sources (owid, scientific, synthetic, kaggle_like, additional, worldbank): 12K with existing reasoning
- v9 blocks supplement: ~8K (scientific_ext, plotly_complex) for domain diversity
- Total: ~30K, 100% reasoning, balanced question/answer distribution

Selection criteria derived from actual data analysis:
- chartqa_train 18K: eval과 distribution 일치가 가장 높음 → 10K 선별
  - multi_step 우선 (eval human 40.8% vs train 24.6%)
  - which_entity 보강 (eval 10.7% vs train 5.0%)
  - text 답변 우선 (eval pro 60% text vs train 17% text)
  - comparison 보강 (eval charxiv 27.5%)
  - simple_value 축소 (eval 3% but train 2.2% — already low)
  - counting 축소 (eval 9.2% vs train 23.3% — overrepresented)

- v8 rest 12K: 100% reasoning, 이미 검증된 데이터 → 전량 유지

- v9 supplement: scientific_ext/plotly_complex for charxiv/pro coverage → ~8K
  - scientific_ext: charxiv 도메인 커버 (scatter, heatmap, scientific figure)
  - plotly_complex: pro/museum 도메인 커버 (infographic, interactive)
  - text 답변 비율 높은 것 우선

Usage:
    python scripts/assemble_v8_improved.py \
        --teacher data/sft_hq_teacher.jsonl \
        --v8_base data/sft_30k.jsonl \
        --v9_base data/sft_v9.jsonl \
        --output data/sft_v8_improved.jsonl \
        --stats
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

random.seed(42)


def classify_answer(ans: str) -> str:
    ans = str(ans).strip()
    if ans.lower() in ('yes', 'no'):
        return 'yesno'
    try:
        float(ans.replace(',', '').replace('%', ''))
        return 'numeric'
    except ValueError:
        return 'text'


def classify_question(q: str) -> str:
    q = q.lower().strip()
    multi_step_kw = ['difference', 'average', 'mean', 'ratio', 'sum of',
                     'product', 'median', 'percent', 'percentage of',
                     'total', 'overall', 'combined']
    comparison_kw = ['more than', 'less than', 'higher', 'lower', 'greater',
                     'compared', 'between', 'most', 'least', 'smallest', 'largest',
                     'maximum', 'minimum', 'longest', 'shortest']
    if any(w in q for w in multi_step_kw):
        return 'multi_step'
    elif any(w in q for w in comparison_kw):
        return 'comparison'
    elif q.startswith(('is ', 'are ', 'does ', 'do ', 'did ', 'was ', 'were ', 'has ')):
        return 'yesno'
    elif q.startswith(('which ', 'who ')):
        return 'which_entity'
    elif any(w in q for w in ['how many', 'how much']):
        return 'counting'
    elif any(w in q for w in ['what is the value', 'what was the value']):
        return 'simple_value'
    else:
        return 'other'


def has_reasoning(sample: dict) -> bool:
    """Check if sample has non-trivial reasoning."""
    at = sample.get('assistant_text', '')
    if at and len(at) > 50:
        return True
    r = sample.get('reasoning_steps', sample.get('reasoning', ''))
    if isinstance(r, list):
        r = '\n'.join(r)
    return bool(r and r.strip() and r.strip() not in ('N/A', 'None', ''))


def reasoning_length(sample: dict) -> int:
    """Get reasoning character length."""
    at = sample.get('assistant_text', '')
    if at:
        return len(at)
    r = sample.get('reasoning_steps', sample.get('reasoning', ''))
    if isinstance(r, list):
        r = '\n'.join(r)
    return len(r) if r else 0


def dedup_key(s: dict) -> str:
    """Normalize key: strip absolute prefix, use question[:80]."""
    img = s.get('image_path', '')
    # Normalize: strip /ex_disk2/mhpark/poc/chartvr/ prefix for cross-dataset matching
    img = img.replace('/ex_disk2/mhpark/poc/chartvr/', '')
    q = (s.get('question') or '')[:80]
    return f"{img}|{q}"


def select_chartqa_train(chartqa_samples: list, teacher_data: dict,
                         target: int = 10000) -> list:
    """Select 10K from 18K chartqa_train, prioritizing eval-aligned distribution.

    Target question type distribution (aligned with eval benchmarks):
      multi_step:    40%  (4000)  — eval human 40.8%, pro 45.7%
      comparison:    15%  (1500)  — eval charxiv 27.5%, pro 14.5%
      which_entity:  12%  (1200)  — eval human 10.7%, charxiv 8.1%
      counting:      10%  (1000)  — eval human 9.2%, charxiv 11.0%
      yesno:          5%  (500)   — eval human 2.4% but underrepresented in train
      other:         15%  (1500)  — diverse coverage
      simple_value:   3%  (300)   — minimal, eval only 3%

    Within each bucket, prioritize:
    1. Samples with teacher reasoning (highest quality)
    2. Text/yesno answers over numeric (address distribution gap)
    3. Longer questions (tend to be more complex)
    """
    # Build teacher lookup
    teacher_keys = set(teacher_data.keys())

    # Classify all samples
    buckets = {}
    for s in chartqa_samples:
        qt = classify_question(s['question'])
        if qt not in buckets:
            buckets[qt] = []
        buckets[qt].append(s)

    targets = {
        'multi_step': 4000,
        'comparison': 1500,
        'which_entity': 1200,
        'counting': 1000,
        'yesno': 500,
        'other': 1500,
        'simple_value': 300,
    }

    selected = []

    for qt, quota in targets.items():
        pool = buckets.get(qt, [])
        if not pool:
            continue

        # Score each sample: teacher reasoning > text answer > question length
        def score(s):
            key = dedup_key(s)
            has_teacher = 1 if key in teacher_keys else 0
            at = classify_answer(s['answer'])
            is_text = 1 if at == 'text' else (0.8 if at == 'yesno' else 0)
            q_len = min(len(s['question']) / 200, 1.0)  # normalize
            return (has_teacher * 10, is_text * 3, q_len)

        pool.sort(key=score, reverse=True)
        take = min(quota, len(pool))
        selected.extend(pool[:take])

    # If under target, fill from remaining unselected samples
    selected_keys = set(dedup_key(s) for s in selected)
    remaining = [s for s in chartqa_samples if dedup_key(s) not in selected_keys]

    # Prioritize teacher-distilled remaining samples
    remaining.sort(key=lambda s: (dedup_key(s) in teacher_keys, classify_answer(s['answer']) == 'text'), reverse=True)

    deficit = target - len(selected)
    if deficit > 0:
        selected.extend(remaining[:deficit])

    return selected[:target]


def select_v9_supplement(v9_samples: list, existing_keys: set,
                         target: int = 8000) -> list:
    """Select ~8K from v9 blocks for domain diversity.

    Prioritize:
    - scientific_ext: charxiv/pro domain coverage (scatter, heatmap, errorbar)
    - plotly_complex: diverse chart types for chartmuseum
    - Text answers to address the numeric bias
    - which_entity and comparison questions (underrepresented)
    """
    # Only non-duplicate, reasoning-bearing samples
    pool = [s for s in v9_samples
            if dedup_key(s) not in existing_keys and has_reasoning(s)]

    # Preferred sources (charxiv/pro/museum domain)
    preferred_sources = {'scientific_ext', 'plotly_complex', 'block_c', 'block_d'}

    # Score: preferred source > text answer > comparison/which > reasoning length
    def score(s):
        src_pref = 2 if s.get('source', '') in preferred_sources else 0
        at = classify_answer(s['answer'])
        text_bonus = 1.5 if at == 'text' else (1.2 if at == 'yesno' else 0)
        qt = classify_question(s['question'])
        qt_bonus = 1.0 if qt in ('comparison', 'which_entity', 'yesno') else 0
        r_len = min(reasoning_length(s) / 500, 1.0)
        return (src_pref, text_bonus, qt_bonus, r_len)

    pool.sort(key=score, reverse=True)
    return pool[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='data/sft_hq_teacher.jsonl',
                    help='Teacher distilled data (has assistant_text)')
    ap.add_argument('--v8_base', default='data/sft_30k.jsonl',
                    help='v8 SFT 30K base data')
    ap.add_argument('--v9_base', default='data/sft_v9.jsonl',
                    help='v9 SFT data (for supplement)')
    ap.add_argument('--output', default='data/sft_v8_improved.jsonl')
    ap.add_argument('--chartqa_target', type=int, default=10000,
                    help='How many chartqa_train samples to include')
    ap.add_argument('--v9_target', type=int, default=8000,
                    help='How many v9 supplement samples to include')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--dry_run', action='store_true',
                    help='Print stats without writing output')
    args = ap.parse_args()

    # === Load data ===
    print("Loading data...")
    with open(args.v8_base) as f:
        v8_all = [json.loads(l) for l in f]

    v8_chartqa = [s for s in v8_all if s.get('source') == 'chartqa_train']
    v8_rest = [s for s in v8_all if s.get('source') != 'chartqa_train']

    # Load teacher data (keyed by dedup_key for fast lookup)
    teacher_map = {}
    if os.path.exists(args.teacher):
        with open(args.teacher) as f:
            for line in f:
                s = json.loads(line)
                teacher_map[dedup_key(s)] = s
        print(f"  Teacher data: {len(teacher_map)} samples")
    else:
        print(f"  WARNING: Teacher data not found at {args.teacher}")

    with open(args.v9_base) as f:
        v9_all = [json.loads(l) for l in f]

    print(f"  v8 chartqa_train: {len(v8_chartqa)}")
    print(f"  v8 rest (with reasoning): {len(v8_rest)}")
    print(f"  v9 total: {len(v9_all)}")

    # === Step 1: Select chartqa_train 10K ===
    print(f"\nSelecting {args.chartqa_target} chartqa_train samples...")
    chartqa_selected = select_chartqa_train(v8_chartqa, teacher_map, args.chartqa_target)

    # Merge teacher reasoning where available
    teacher_merged = 0
    for i, s in enumerate(chartqa_selected):
        key = dedup_key(s)
        if key in teacher_map:
            # Use teacher's assistant_text (high-quality <think>...<answer> format)
            chartqa_selected[i] = {**s, 'assistant_text': teacher_map[key]['assistant_text']}
            teacher_merged += 1

    print(f"  Selected: {len(chartqa_selected)}")
    print(f"  Teacher reasoning merged: {teacher_merged} / {len(chartqa_selected)} ({100*teacher_merged/len(chartqa_selected):.1f}%)")
    print(f"  Remaining without teacher: {len(chartqa_selected) - teacher_merged}")

    # === Step 2: v8 rest 12K (all have reasoning) ===
    print(f"\nv8 rest sources: {len(v8_rest)} samples")
    no_reason = sum(1 for s in v8_rest if not has_reasoning(s))
    print(f"  Without reasoning: {no_reason} (will use template fallback)")

    # === Step 3: v9 supplement ===
    existing_keys = set(dedup_key(s) for s in chartqa_selected + v8_rest)
    print(f"\nSelecting v9 supplement (target {args.v9_target})...")
    v9_selected = select_v9_supplement(v9_all, existing_keys, args.v9_target)
    print(f"  Selected: {len(v9_selected)}")

    # === Combine ===
    final = chartqa_selected + v8_rest + v9_selected

    # Deduplicate
    seen = set()
    deduped = []
    for s in final:
        key = dedup_key(s)
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    random.shuffle(deduped)

    # === Stats ===
    total = len(deduped)
    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {total} samples")
    print(f"{'='*60}")

    # Source
    src_cnt = Counter(s.get('source', '?') for s in deduped)
    print("\nSource distribution:")
    for k, v in src_cnt.most_common():
        print(f"  {k:>20}: {v:>5} ({100*v/total:.1f}%)")

    # Reasoning coverage
    has_r = sum(1 for s in deduped if has_reasoning(s))
    has_teacher = sum(1 for s in deduped if s.get('assistant_text'))
    print(f"\nReasoning coverage: {has_r}/{total} ({100*has_r/total:.1f}%)")
    print(f"  Teacher assistant_text: {has_teacher}")
    print(f"  Self-gen reasoning: {has_r - has_teacher}")
    print(f"  No reasoning (template): {total - has_r}")

    # Answer types
    ans_cnt = Counter(classify_answer(s['answer']) for s in deduped)
    print(f"\nAnswer type distribution:")
    for k, v in ans_cnt.most_common():
        print(f"  {k:>10}: {v:>5} ({100*v/total:.1f}%)")

    # Question types
    q_cnt = Counter(classify_question(s['question']) for s in deduped)
    print(f"\nQuestion complexity distribution:")
    for k, v in q_cnt.most_common():
        print(f"  {k:>15}: {v:>5} ({100*v/total:.1f}%)")

    # Samples still needing teacher reasoning
    chartqa_no_teacher = [s for s in chartqa_selected if not s.get('assistant_text')]
    print(f"\n*** chartqa_train needing teacher distill: {len(chartqa_no_teacher)} ***")

    if args.stats:
        # Detailed comparison with eval benchmarks
        print(f"\n{'='*60}")
        print("Distribution alignment with eval benchmarks:")
        print(f"{'='*60}")
        eval_targets = {
            'multi_step': 40.8,
            'comparison': 14.5,
            'which_entity': 10.7,
            'counting': 9.2,
            'yesno': 2.4,
            'other': 21.3,
            'simple_value': 3.0,
        }
        print(f"  {'Type':>15} | {'Train%':>7} | {'Eval%':>7} | {'Gap':>7}")
        print(f"  {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        for qt in ['multi_step', 'comparison', 'which_entity', 'counting', 'yesno', 'other', 'simple_value']:
            train_pct = 100 * q_cnt.get(qt, 0) / total
            eval_pct = eval_targets.get(qt, 0)
            gap = train_pct - eval_pct
            print(f"  {qt:>15} | {train_pct:>6.1f}% | {eval_pct:>6.1f}% | {gap:>+6.1f}%")

    if not args.dry_run:
        # === Write output ===
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            for s in deduped:
                f.write(json.dumps(s, ensure_ascii=False, default=str) + '\n')
        print(f"\nWritten to {args.output}")

        # Also write the list of chartqa samples needing teacher distill
        if chartqa_no_teacher:
            distill_source = args.output.replace('.jsonl', '_need_teacher.jsonl')
            with open(distill_source, 'w') as f:
                for s in chartqa_no_teacher:
                    f.write(json.dumps(s, ensure_ascii=False, default=str) + '\n')
            print(f"Teacher distill source: {distill_source} ({len(chartqa_no_teacher)} samples)")
    else:
        print("\n[DRY RUN — no files written]")


if __name__ == '__main__':
    main()
