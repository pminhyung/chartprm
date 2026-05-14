#!/usr/bin/env python
"""Rule-based QA generation for GRPO training data.

Generates questions with guaranteed correct answers from CSV data.
No LLM needed — deterministic, arithmetic-verifiable answers.

Usage:
  python scripts/generate_qa_rule.py \
      --manifests data/charts_v2/owid/manifest.jsonl \
                  data/charts_v2/worldbank/manifest.jsonl \
                  data/charts_v2/synthetic/manifest.jsonl \
                  data/charts_v2/kaggle_like/manifest.jsonl \
                  data/charts_v2/scientific/manifest.jsonl \
                  data/charts_v2/additional/manifest.jsonl \
      --output data/charts_v2/rule_based_qa.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _safe_round(v, decimals=2):
    """Round, handling edge cases."""
    try:
        return round(float(v), decimals)
    except (ValueError, TypeError):
        return None


def generate_questions_from_csv(csv_path: str, max_per_chart: int = 5) -> list:
    """Generate rule-based QA pairs from CSV with guaranteed correct answers.

    Templates:
      1. max/min value
      2. difference between max and min
      3. average of a column
      4. sum of a column
      5. ratio of max to min
      6. value for specific entity
      7. which entity has highest/lowest
      8. percentage of total
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if len(df) < 2:
        return []

    # Identify columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols:
        return []

    # Filter out year-like columns from numeric
    value_cols = [c for c in numeric_cols
                  if not any(y in c.lower() for y in ("year", "date", "time", "index"))]
    if not value_cols:
        value_cols = numeric_cols

    # Get label column (first categorical or first column)
    label_col = cat_cols[0] if cat_cols else df.columns[0]

    questions = []

    for col in value_cols[:2]:  # up to 2 numeric columns
        vals = df[col].dropna()
        if len(vals) < 2:
            continue

        max_val = _safe_round(vals.max())
        min_val = _safe_round(vals.min())
        mean_val = _safe_round(vals.mean())
        sum_val = _safe_round(vals.sum())
        col_clean = col.replace("_", " ")

        # Template 1: What is the maximum value?
        if max_val is not None:
            questions.append({
                "question": f"What is the maximum {col_clean}?",
                "answer": max_val,
                "reasoning_steps": [f"Looking at the {col_clean} column, the maximum value is {max_val}."],
                "difficulty": "medium",
                "template": "max_value",
            })

        # Template 2: Difference between max and min
        if max_val is not None and min_val is not None:
            diff = _safe_round(max_val - min_val)
            if diff is not None and diff != 0:
                questions.append({
                    "question": f"What is the difference between the highest and lowest {col_clean}?",
                    "answer": diff,
                    "reasoning_steps": [
                        f"The maximum {col_clean} is {max_val}.",
                        f"The minimum {col_clean} is {min_val}.",
                        f"The difference is {max_val} - {min_val} = {diff}.",
                    ],
                    "difficulty": "medium",
                    "template": "diff_max_min",
                })

        # Template 3: Average
        if mean_val is not None:
            questions.append({
                "question": f"What is the average {col_clean}?",
                "answer": mean_val,
                "reasoning_steps": [
                    f"Sum of all {col_clean} values: {sum_val}.",
                    f"Number of entries: {len(vals)}.",
                    f"Average = {sum_val} / {len(vals)} = {mean_val}.",
                ],
                "difficulty": "hard",
                "template": "average",
            })

        # Template 4: Sum
        if sum_val is not None and len(vals) <= 10:
            questions.append({
                "question": f"What is the total {col_clean} across all entries?",
                "answer": sum_val,
                "reasoning_steps": [f"Adding all {col_clean} values: {sum_val}."],
                "difficulty": "medium",
                "template": "sum",
            })

        # Template 5: Ratio of max to min
        if max_val and min_val and min_val != 0:
            ratio = _safe_round(max_val / min_val)
            if ratio is not None and ratio != 1:
                questions.append({
                    "question": f"What is the ratio of the highest to the lowest {col_clean}?",
                    "answer": ratio,
                    "reasoning_steps": [
                        f"Highest: {max_val}.",
                        f"Lowest: {min_val}.",
                        f"Ratio = {max_val} / {min_val} = {ratio}.",
                    ],
                    "difficulty": "hard",
                    "template": "ratio",
                })

        # Template 6: Value for specific entity (text-answer-friendly: "which is highest")
        if label_col != col and len(df) >= 3:
            idx_max = vals.idxmax()
            entity_max = str(df.loc[idx_max, label_col])
            if entity_max and entity_max != "nan":
                questions.append({
                    "question": f"Which entry has the highest {col_clean}?",
                    "answer": entity_max,
                    "reasoning_steps": [
                        f"Looking at the {col_clean} column, {entity_max} has the highest value of {max_val}.",
                    ],
                    "difficulty": "medium",
                    "template": "which_highest",
                })

                idx_min = vals.idxmin()
                entity_min = str(df.loc[idx_min, label_col])
                if entity_min and entity_min != "nan" and entity_min != entity_max:
                    questions.append({
                        "question": f"Which entry has the lowest {col_clean}?",
                        "answer": entity_min,
                        "reasoning_steps": [
                            f"Looking at the {col_clean} column, {entity_min} has the lowest value of {min_val}.",
                        ],
                        "difficulty": "medium",
                        "template": "which_lowest",
                    })

        # Template 7: Percentage of total
        if sum_val and sum_val != 0 and label_col != col and len(df) >= 3:
            idx = random.choice(vals.index.tolist())
            val = _safe_round(df.loc[idx, col])
            entity = str(df.loc[idx, label_col])
            if val is not None and entity != "nan":
                pct = _safe_round(val / sum_val * 100)
                if pct is not None:
                    questions.append({
                        "question": f"What percentage of the total {col_clean} does {entity} account for?",
                        "answer": pct,
                        "reasoning_steps": [
                            f"{entity}'s {col_clean}: {val}.",
                            f"Total {col_clean}: {sum_val}.",
                            f"Percentage = ({val} / {sum_val}) × 100 = {pct}%.",
                        ],
                        "difficulty": "hard",
                        "template": "percentage",
                    })

        # Template 8: Difference between two specific entities
        if label_col != col and len(df) >= 3:
            idxs = random.sample(vals.index.tolist(), min(2, len(vals)))
            if len(idxs) == 2:
                e1 = str(df.loc[idxs[0], label_col])
                e2 = str(df.loc[idxs[1], label_col])
                v1 = _safe_round(df.loc[idxs[0], col])
                v2 = _safe_round(df.loc[idxs[1], col])
                if v1 is not None and v2 is not None and e1 != "nan" and e2 != "nan":
                    diff = _safe_round(abs(v1 - v2))
                    questions.append({
                        "question": f"What is the difference in {col_clean} between {e1} and {e2}?",
                        "answer": diff,
                        "reasoning_steps": [
                            f"{e1}: {v1}.",
                            f"{e2}: {v2}.",
                            f"Difference = |{v1} - {v2}| = {diff}.",
                        ],
                        "difficulty": "medium",
                        "template": "diff_entities",
                    })

    # Shuffle and limit
    random.shuffle(questions)
    return questions[:max_per_chart]


def main():
    parser = argparse.ArgumentParser(description="Rule-based QA generation")
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest JSONL files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max_per_chart", type=int, default=5, help="Max QA pairs per chart")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load all chart manifests
    charts = []
    for mpath in args.manifests:
        if not os.path.exists(mpath):
            print(f"  Skipping {mpath} (not found)")
            continue
        with open(mpath) as f:
            for line in f:
                try:
                    charts.append(json.loads(line))
                except:
                    pass
    print(f"Loaded {len(charts)} charts from {len(args.manifests)} manifests")

    # Resume
    done_slugs = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_slugs.add(json.loads(line).get("chart_slug", ""))
                except:
                    pass
    pending = [c for c in charts if c.get("slug", "") not in done_slugs]
    print(f"  Done: {len(done_slugs)}, Pending: {len(pending)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a")

    total_qa = 0
    errors = 0
    for chart in tqdm(pending, desc="Rule-based QA", unit="chart"):
        csv_path = chart.get("csv_path", "")
        if not csv_path or not os.path.exists(csv_path):
            errors += 1
            continue

        questions = generate_questions_from_csv(csv_path, args.max_per_chart)
        for qa in questions:
            qa.update({
                "csv_path": csv_path,
                "image_path": chart.get("image_path", ""),
                "source": chart.get("source", ""),
                "chart_slug": chart.get("slug", ""),
                "verified": True,
                "generation_method": "rule_based",
            })
            out_f.write(json.dumps(qa, ensure_ascii=False, default=str) + "\n")
            total_qa += 1

        if total_qa % 1000 == 0 and total_qa > 0:
            out_f.flush()

    out_f.close()

    # Stats
    print(f"\nGenerated {total_qa} QAs from {len(pending)} charts (errors={errors})")

    # Count by template
    templates = {}
    text_answers = 0
    with open(args.output) as f:
        for line in f:
            try:
                rec = json.loads(line)
                t = rec.get("template", "unknown")
                templates[t] = templates.get(t, 0) + 1
                ans = str(rec.get("answer", ""))
                try:
                    float(ans.replace(",", ""))
                except ValueError:
                    text_answers += 1
            except:
                pass

    total = sum(templates.values())
    print(f"\nTotal QAs: {total}")
    print(f"Text answers: {text_answers} ({text_answers/max(total,1)*100:.1f}%)")
    for t, c in sorted(templates.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
