#!/usr/bin/env python
"""Extended rule-based QA generation for v9 data pipeline.

Imports base templates from generate_qa_rule.py and adds:
  Block A: text answer templates (yes_no, trend, which_entity, count_above, title/label reading)
  Block B: CQA-Aug style simple templates (value_of, how_many_above, which_highest, difference, average)
  Block F: edge case templates (hypothetical, unanswerable, multi_choice, conversational)

All templates produce reasoning_steps for CoT generation.

Usage:
  # Block A: text-heavy QA
  python scripts/generate_qa_rule_v9.py \
      --manifests data/charts_v9/kaggle_ext/manifest.jsonl \
                  data/charts_v2/kaggle_like/manifest.jsonl \
                  data/charts_v2/additional/manifest.jsonl \
      --block a --output data/charts_v9/rule_qa_block_a.jsonl --limit 8000

  # Block B: simple template QA
  python scripts/generate_qa_rule_v9.py \
      --manifests data/charts_v9/kaggle_ext/manifest.jsonl \
                  data/charts_v2/kaggle_like/manifest.jsonl \
      --block b --output data/charts_v9/rule_qa_block_b.jsonl --limit 8000

  # Block C: scientific chart QA (rule-based portion)
  python scripts/generate_qa_rule_v9.py \
      --manifests data/charts_v9/scientific_ext/manifest.jsonl \
                  data/charts_v2/scientific/manifest.jsonl \
      --block c --output data/charts_v9/block_c_rule_qa.jsonl --limit 5000

  # Block F: edge case QA
  python scripts/generate_qa_rule_v9.py \
      --manifests data/charts_v9/kaggle_ext/manifest.jsonl \
                  data/charts_v2/kaggle_like/manifest.jsonl \
                  data/charts_v2/additional/manifest.jsonl \
      --block f --output data/charts_v9/rule_qa_block_f.jsonl --limit 4000
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_qa_rule import generate_questions_from_csv, _safe_round


# ═══════════════════════════════════════════
# Block A Templates: Text Answer + Yes/No
# ═══════════════════════════════════════════

def generate_block_a_questions(csv_path, chart_meta=None, max_per_chart=8):
    """Generate text-heavy QA with controlled ratios: 50% numeric, 30% text, 20% yes/no."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if len(df) < 2:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols:
        return []

    value_cols = [c for c in numeric_cols
                  if not any(y in c.lower() for y in ("year", "date", "time", "index"))]
    if not value_cols:
        value_cols = numeric_cols

    label_col = cat_cols[0] if cat_cols else df.columns[0]
    questions = []
    title = chart_meta.get("title", "") if chart_meta else ""

    for col in value_cols[:2]:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        col_clean = col.replace("_", " ")

        # ── Numeric templates (50%) ──

        # Value reading
        if len(df) >= 3 and label_col != col:
            idx = random.choice(vals.index.tolist())
            entity = str(df.loc[idx, label_col])
            val = _safe_round(df.loc[idx, col])
            if val is not None and entity != "nan":
                questions.append({
                    "question": f"What is the {col_clean} of {entity}?",
                    "answer": val,
                    "answer_type": "numeric",
                    "reasoning_steps": [
                        f"Looking at the data for {entity}.",
                        f"The {col_clean} of {entity} is {val}.",
                    ],
                    "difficulty": "easy",
                    "template": "value_reading",
                })

        # Difference
        if len(df) >= 3 and label_col != col:
            idxs = random.sample(vals.index.tolist(), min(2, len(vals)))
            if len(idxs) == 2:
                e1 = str(df.loc[idxs[0], label_col])
                e2 = str(df.loc[idxs[1], label_col])
                v1, v2 = _safe_round(df.loc[idxs[0], col]), _safe_round(df.loc[idxs[1], col])
                if v1 is not None and v2 is not None:
                    diff = _safe_round(abs(v1 - v2))
                    questions.append({
                        "question": f"What is the difference in {col_clean} between {e1} and {e2}?",
                        "answer": diff,
                        "answer_type": "numeric",
                        "reasoning_steps": [f"{e1}: {v1}.", f"{e2}: {v2}.", f"|{v1} - {v2}| = {diff}."],
                        "difficulty": "medium",
                        "template": "diff_entities",
                    })

        # Average
        mean_val = _safe_round(vals.mean())
        if mean_val is not None:
            questions.append({
                "question": f"What is the average {col_clean}?",
                "answer": mean_val,
                "answer_type": "numeric",
                "reasoning_steps": [f"Sum: {_safe_round(vals.sum())}.", f"Count: {len(vals)}.", f"Average = {mean_val}."],
                "difficulty": "hard",
                "template": "average",
            })

        # ── Text templates (30%) ──

        # Which entity has highest
        if label_col != col and len(df) >= 3:
            idx_max = vals.idxmax()
            entity_max = str(df.loc[idx_max, label_col])
            max_val = _safe_round(vals.max())
            if entity_max != "nan":
                questions.append({
                    "question": f"Which {label_col.lower().replace('_', ' ')} has the highest {col_clean}?",
                    "answer": entity_max,
                    "answer_type": "text",
                    "reasoning_steps": [
                        f"Comparing all {col_clean} values.",
                        f"{entity_max} has the highest value of {max_val}.",
                    ],
                    "difficulty": "medium",
                    "template": "which_entity_highest",
                })

            idx_min = vals.idxmin()
            entity_min = str(df.loc[idx_min, label_col])
            min_val = _safe_round(vals.min())
            if entity_min != "nan" and entity_min != entity_max:
                questions.append({
                    "question": f"Which {label_col.lower().replace('_', ' ')} has the lowest {col_clean}?",
                    "answer": entity_min,
                    "answer_type": "text",
                    "reasoning_steps": [
                        f"Comparing all {col_clean} values.",
                        f"{entity_min} has the lowest value of {min_val}.",
                    ],
                    "difficulty": "medium",
                    "template": "which_entity_lowest",
                })

        # Title reading
        if title:
            questions.append({
                "question": "What is the title of this chart?",
                "answer": title.split("(")[0].strip()[:80],
                "answer_type": "text",
                "reasoning_steps": [f"The chart title is '{title.split('(')[0].strip()[:80]}'."],
                "difficulty": "easy",
                "template": "title_reading",
            })

        # Label reading
        questions.append({
            "question": f"What metric is shown on the y-axis?",
            "answer": col,
            "answer_type": "text",
            "reasoning_steps": [f"The y-axis label is '{col}'."],
            "difficulty": "easy",
            "template": "label_reading",
        })

        # ── Yes/No templates (20%) ──

        # Comparison yes/no
        if label_col != col and len(df) >= 3:
            idxs = random.sample(vals.index.tolist(), min(2, len(vals)))
            if len(idxs) == 2:
                e1 = str(df.loc[idxs[0], label_col])
                e2 = str(df.loc[idxs[1], label_col])
                v1 = _safe_round(df.loc[idxs[0], col])
                v2 = _safe_round(df.loc[idxs[1], col])
                if v1 is not None and v2 is not None and e1 != "nan" and e2 != "nan":
                    is_greater = v1 > v2
                    questions.append({
                        "question": f"Is {e1}'s {col_clean} greater than {e2}'s?",
                        "answer": "Yes" if is_greater else "No",
                        "answer_type": "yesno",
                        "reasoning_steps": [
                            f"{e1}'s {col_clean}: {v1}.",
                            f"{e2}'s {col_clean}: {v2}.",
                            f"{v1} {'>' if is_greater else '<='} {v2}, so the answer is {'Yes' if is_greater else 'No'}.",
                        ],
                        "difficulty": "medium",
                        "template": "yes_no_comparison",
                    })

        # Trend direction (for time series)
        if len(vals) >= 4:
            half = len(vals) // 2
            first_half_mean = vals.iloc[:half].mean()
            second_half_mean = vals.iloc[half:].mean()
            is_increasing = second_half_mean > first_half_mean
            questions.append({
                "question": f"Is the overall {col_clean} trend increasing or decreasing?",
                "answer": "Increasing" if is_increasing else "Decreasing",
                "answer_type": "text",
                "reasoning_steps": [
                    f"First half average: {_safe_round(first_half_mean)}.",
                    f"Second half average: {_safe_round(second_half_mean)}.",
                    f"Since {_safe_round(second_half_mean)} {'>' if is_increasing else '<'} {_safe_round(first_half_mean)}, the trend is {'increasing' if is_increasing else 'decreasing'}.",
                ],
                "difficulty": "medium",
                "template": "trend_direction",
            })

        # Count above threshold
        if len(vals) >= 3:
            median = vals.median()
            threshold = _safe_round(median)
            if threshold is not None:
                count = int((vals > threshold).sum())
                questions.append({
                    "question": f"How many entries have {col_clean} above {threshold}?",
                    "answer": str(count),
                    "answer_type": "numeric",
                    "reasoning_steps": [
                        f"Threshold: {threshold}.",
                        f"Counting entries with {col_clean} > {threshold}.",
                        f"There are {count} entries above {threshold}.",
                    ],
                    "difficulty": "medium",
                    "template": "count_above",
                })

    # Enforce answer type ratios: 50% numeric, 30% text, 20% yes/no
    by_type = {"numeric": [], "text": [], "yesno": []}
    for q in questions:
        at = q.get("answer_type", "numeric")
        if at in by_type:
            by_type[at].append(q)
        else:
            by_type["numeric"].append(q)

    # Weighted selection: 50/30/20
    selected = []
    targets = {"numeric": int(max_per_chart * 0.50),
               "text": int(max_per_chart * 0.30),
               "yesno": max_per_chart - int(max_per_chart * 0.50) - int(max_per_chart * 0.30)}
    for at, target in targets.items():
        pool = by_type.get(at, [])
        random.shuffle(pool)
        selected.extend(pool[:target])

    # Fill remaining slots from any type
    used = set(id(q) for q in selected)
    remaining = [q for q in questions if id(q) not in used]
    random.shuffle(remaining)
    while len(selected) < max_per_chart and remaining:
        selected.append(remaining.pop())

    random.shuffle(selected)
    return selected


# ═══════════════════════════════════════════
# Block B Templates: Simple CQA-Aug style
# ═══════════════════════════════════════════

def generate_block_b_questions(csv_path, chart_meta=None, max_per_chart=6):
    """Generate CQA-Augmented style questions.

    Target ratios (from plan):
      value_of: 40%, difference: 20%, which_highest: 15%,
      how_many_above: 15%, average/sum: 10%
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if len(df) < 2:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols:
        return []

    value_cols = [c for c in numeric_cols
                  if not any(y in c.lower() for y in ("year", "date", "time", "index"))]
    if not value_cols:
        value_cols = numeric_cols

    label_col = cat_cols[0] if cat_cols else df.columns[0]

    # Collect all possible questions by template
    by_template = {"value_of": [], "how_many_above": [], "which_highest_lowest": [],
                   "difference_between": [], "average_simple": []}

    for col in value_cols[:2]:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        col_clean = col.replace("_", " ")

        # value_of — generate MULTIPLE (one per entity) for 40% target
        if label_col != col and len(df) >= 3:
            sample_idxs = random.sample(vals.index.tolist(), min(4, len(vals)))
            for idx in sample_idxs:
                entity = str(df.loc[idx, label_col])
                val = _safe_round(df.loc[idx, col])
                if val is not None and entity != "nan":
                    by_template["value_of"].append({
                        "question": f"What is the {col_clean} of {entity}?",
                        "answer": val,
                        "answer_type": "numeric",
                        "reasoning_steps": [f"The {col_clean} of {entity} is {val}."],
                        "difficulty": "easy",
                        "template": "value_of",
                    })

        # how_many_above
        if len(vals) >= 4:
            threshold = _safe_round(vals.quantile(random.choice([0.25, 0.5, 0.75])))
            if threshold is not None:
                above = random.choice([True, False])
                count = int((vals > threshold).sum() if above else (vals < threshold).sum())
                word = "above" if above else "below"
                by_template["how_many_above"].append({
                    "question": f"How many entries have {col_clean} {word} {threshold}?",
                    "answer": str(count),
                    "answer_type": "numeric",
                    "reasoning_steps": [f"Count entries with {col_clean} {word} {threshold}: {count}."],
                    "difficulty": "easy",
                    "template": "how_many_above",
                })

        # which_highest_lowest
        if label_col != col and len(df) >= 3:
            is_max = random.choice([True, False])
            idx_target = vals.idxmax() if is_max else vals.idxmin()
            entity = str(df.loc[idx_target, label_col])
            word = "highest" if is_max else "lowest"
            if entity != "nan":
                by_template["which_highest_lowest"].append({
                    "question": f"Which {label_col.lower().replace('_', ' ')} has the {word} {col_clean}?",
                    "answer": entity,
                    "answer_type": "text",
                    "reasoning_steps": [f"{entity} has the {word} {col_clean}."],
                    "difficulty": "easy",
                    "template": "which_highest_lowest",
                })

        # difference_between
        if label_col != col and len(df) >= 3:
            idxs = random.sample(vals.index.tolist(), min(2, len(vals)))
            if len(idxs) == 2:
                e1 = str(df.loc[idxs[0], label_col])
                e2 = str(df.loc[idxs[1], label_col])
                v1, v2 = _safe_round(df.loc[idxs[0], col]), _safe_round(df.loc[idxs[1], col])
                if v1 is not None and v2 is not None:
                    diff = _safe_round(abs(v1 - v2))
                    by_template["difference_between"].append({
                        "question": f"What is the difference between {e1} and {e2} in {col_clean}?",
                        "answer": diff,
                        "answer_type": "numeric",
                        "reasoning_steps": [f"{e1}: {v1}. {e2}: {v2}. Difference: {diff}."],
                        "difficulty": "medium",
                        "template": "difference_between",
                    })

        # average/sum
        mean_val = _safe_round(vals.mean())
        sum_val = _safe_round(vals.sum())
        if mean_val is not None:
            by_template["average_simple"].append({
                "question": f"What is the average {col_clean}?",
                "answer": mean_val,
                "answer_type": "numeric",
                "reasoning_steps": [f"Average of all {col_clean} values = {mean_val}."],
                "difficulty": "medium",
                "template": "average_simple",
            })

    # Weighted selection matching CQA-Aug ratios: value_of 40%, diff 20%, which 15%, how_many 15%, avg 10%
    weights = {"value_of": 0.40, "difference_between": 0.20, "which_highest_lowest": 0.15,
               "how_many_above": 0.15, "average_simple": 0.10}
    selected = []
    for tmpl, weight in weights.items():
        n = max(1, round(max_per_chart * weight))
        pool = by_template.get(tmpl, [])
        random.shuffle(pool)
        selected.extend(pool[:n])

    random.shuffle(selected)
    return selected[:max_per_chart]


# ═══════════════════════════════════════════
# Block F Templates: Edge Cases
# ═══════════════════════════════════════════

def generate_block_f_questions(csv_path, chart_meta=None, max_per_chart=4):
    """Generate edge case questions: hypothetical, unanswerable, multi-choice, conversational."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []

    if len(df) < 3:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not numeric_cols:
        return []

    value_cols = [c for c in numeric_cols
                  if not any(y in c.lower() for y in ("year", "date", "time", "index"))]
    if not value_cols:
        value_cols = numeric_cols

    label_col = cat_cols[0] if cat_cols else df.columns[0]
    questions = []

    for col in value_cols[:1]:
        vals = df[col].dropna()
        if len(vals) < 3:
            continue
        col_clean = col.replace("_", " ")

        # 1. Hypothetical (1K target)
        if label_col != col:
            idx = random.choice(vals.index.tolist())
            entity = str(df.loc[idx, label_col])
            val = _safe_round(df.loc[idx, col])
            pct = random.choice([10, 15, 20, 25, 30, 50])
            if val is not None and entity != "nan" and val != 0:
                new_val = _safe_round(val * (1 + pct / 100))
                questions.append({
                    "question": f"If {entity}'s {col_clean} increased by {pct}%, what would the new value be?",
                    "answer": new_val,
                    "answer_type": "numeric",
                    "reasoning_steps": [
                        f"Current {col_clean} of {entity}: {val}.",
                        f"Increase by {pct}%: {val} × {1 + pct/100} = {new_val}.",
                    ],
                    "difficulty": "medium",
                    "template": "hypothetical",
                })

        # 2. Unanswerable (1K target)
        if label_col != col:
            existing_entities = set(df[label_col].astype(str).tolist())
            fake_entities = [
                "Atlantis", "Narnia", "Wakanda", "Mordor", "Zamunda",
                "Freedonia", "Genovia", "Elbonia", "Latveria", "Sokovia",
            ]
            fake = random.choice([f for f in fake_entities if f not in existing_entities])
            questions.append({
                "question": f"What is the {col_clean} of {fake}?",
                "answer": "Cannot be determined",
                "answer_type": "unanswerable",
                "reasoning_steps": [
                    f"Looking for {fake} in the data.",
                    f"{fake} is not present in the chart.",
                    "The answer cannot be determined from the available data.",
                ],
                "difficulty": "medium",
                "template": "unanswerable",
            })

        # 3. Multi-choice (1K target)
        if label_col != col and len(df) >= 3:
            idx = random.choice(vals.index.tolist())
            entity = str(df.loc[idx, label_col])
            correct_val = _safe_round(df.loc[idx, col])
            if correct_val is not None and entity != "nan" and correct_val != 0:
                correct_pos = random.randint(0, 3)
                options = []
                for j in range(4):
                    if j == correct_pos:
                        options.append(correct_val)
                    else:
                        offset = random.uniform(0.1, 0.3) * abs(correct_val)
                        sign = random.choice([-1, 1])
                        options.append(_safe_round(correct_val + sign * offset))

                letters = ["A", "B", "C", "D"]
                opts_str = ", ".join(f"{letters[j]}) {options[j]}" for j in range(4))
                correct_letter = letters[correct_pos]

                questions.append({
                    "question": f"What is the {col_clean} of {entity}? Options: {opts_str}",
                    "answer": correct_letter,
                    "answer_type": "multichoice",
                    "reasoning_steps": [
                        f"The {col_clean} of {entity} is {correct_val}.",
                        f"Option {correct_letter}) {correct_val} matches.",
                    ],
                    "difficulty": "easy",
                    "template": "multi_choice",
                })

        # 4. Conversational 2-turn (1K target)
        if label_col != col and len(df) >= 4:
            idx_max = vals.idxmax()
            entity_max = str(df.loc[idx_max, label_col])
            max_val = _safe_round(vals.max())
            if entity_max != "nan":
                # Pick a second entity
                other_idxs = [i for i in vals.index if i != idx_max]
                if other_idxs:
                    idx2 = random.choice(other_idxs)
                    entity2 = str(df.loc[idx2, label_col])
                    val2 = _safe_round(df.loc[idx2, col])
                    if entity2 != "nan" and val2 is not None:
                        questions.append({
                            "question": (
                                f"Which {label_col.lower().replace('_', ' ')} has the highest {col_clean}? "
                                f"Follow up: What is {entity2}'s {col_clean}?"
                            ),
                            "answer": f"{entity_max}; {val2}",
                            "answer_type": "conversational",
                            "reasoning_steps": [
                                f"First question: {entity_max} has the highest {col_clean} at {max_val}.",
                                f"Follow up: {entity2}'s {col_clean} is {val2}.",
                            ],
                            "difficulty": "medium",
                            "template": "conversational",
                        })

    random.shuffle(questions)
    return questions[:max_per_chart]


# ═══════════════════════════════════════════
# Block C: Scientific QA (rule-based portion)
# ═══════════════════════════════════════════

def generate_block_c_questions_legacy(csv_path, chart_meta=None, max_per_chart=5):
    """LEGACY: Block C generator that assumes first column is entity.
    Kept as fallback for unknown chart types. Use generate_block_c_questions_v2 instead.
    """
    base_qs = generate_questions_from_csv(csv_path, max_per_chart=3)
    text_qs = generate_block_a_questions(csv_path, chart_meta, max_per_chart=3)
    combined = base_qs + text_qs
    random.shuffle(combined)
    return combined[:max_per_chart]


# ═══════════════════════════════════════════
# Block C v9.1: Per-chart-type dispatchers
# ═══════════════════════════════════════════
#
# Each scientific chart type has a dedicated generator that understands its
# schema. Rule QA is deterministic (answers computed from df) and produces a
# mix of text (ranking/which) and numeric (lookup/compute) answers, matching
# the CharXiv distribution (54% text / 45% numeric).
#
# Schemas (from manifest.jsonl + actual CSVs):
#   scatter_regression:  (x, y, Fitted_y) + r_squared, coeffs in meta
#   subplot_panel:       (panel, Epoch, Value) + panel_labels in meta
#   heatmap_corr:        NxN matrix, columns=var_names (rows i → var_names[i])
#   errorbar_comparison: (Method, <metric>, <metric>_std)
#   logscale_plot:       (x, y) with log-scale y
#   dual_axis:           (x, m1, m2)
#   confidence_band:     (x, Method, y, y_lower, y_upper)
#   violin_comparison:   (Method, <metric>) N methods × M samples each
#   bar_scientific/line/scatter/scatter_line (v2 pool): 2-col (x, y) numeric

def _safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _gen_scatter_regression_qa(df, meta, max_per_chart=4):
    qs = []
    if df.shape[1] < 3 or len(df) < 3:
        return qs
    x_col, y_col, fit_col = df.columns[0], df.columns[1], df.columns[2]
    # Text: trend direction (put first for text-ratio balance)
    try:
        slope = float(np.polyfit(df[x_col].astype(float), df[y_col].astype(float), 1)[0])
        direction = "increasing" if slope > 0 else "decreasing"
        qs.append({"question": f"Is {y_col} generally increasing or decreasing with {x_col}?",
                   "answer": direction, "answer_type": "text",
                   "template": "sci_scatter_trend", "difficulty": "easy",
                   "reasoning_steps": [f"Linear slope = {round(slope,3)}",
                                        f"Positive slope indicates {direction}"]})
    except Exception:
        pass
    # Text: fit quality as descriptive
    if meta and "r_squared" in meta:
        r2v = float(meta["r_squared"])
        if r2v >= 0.9:
            q_desc = "excellent"
        elif r2v >= 0.7:
            q_desc = "good"
        elif r2v >= 0.4:
            q_desc = "moderate"
        else:
            q_desc = "poor"
        qs.append({"question": "How well does the linear fit describe the data based on R-squared?",
                   "answer": q_desc, "answer_type": "text",
                   "template": "sci_scatter_fit_quality", "difficulty": "medium",
                   "reasoning_steps": [f"R-squared = {round(r2v,3)}", f"Interpreted as {q_desc} fit"]})
    # Numeric: lookup y at mid x
    idx = len(df) // 2
    xv = _safe_float(df.iloc[idx][x_col])
    yv = _safe_float(df.iloc[idx][y_col])
    fv = _safe_float(df.iloc[idx][fit_col])
    if xv is None or yv is None:
        return qs[:max_per_chart]
    xv, yv = round(xv, 2), round(yv, 2)
    qs.append({"question": f"What is the {y_col} when {x_col} is {xv}?",
               "answer": yv, "answer_type": "numeric",
               "template": "sci_scatter_lookup", "difficulty": "medium",
               "reasoning_steps": [f"Locate row where {x_col}={xv}", f"Read {y_col}={yv}"]})
    if fv is not None:
        res = round(yv - round(fv, 2), 2)
        qs.append({"question": f"What is the residual (observed minus fitted) at {x_col}={xv}?",
                   "answer": res, "answer_type": "numeric",
                   "template": "sci_residual", "difficulty": "hard",
                   "reasoning_steps": [f"Observed {y_col}={yv}", f"Fitted={round(fv,2)}",
                                        f"Residual = observed - fitted = {res}"]})
    if meta and "r_squared" in meta:
        r2 = round(float(meta["r_squared"]), 3)
        qs.append({"question": "What is the R-squared (coefficient of determination) of the fit?",
                   "answer": r2, "answer_type": "numeric",
                   "template": "sci_r2", "difficulty": "medium",
                   "reasoning_steps": [f"R-squared annotated on plot = {r2}"]})
    return qs[:max_per_chart]


def _gen_subplot_panel_qa(df, meta, max_per_chart=5):
    qs = []
    if "panel" not in df.columns or "Epoch" not in df.columns or "Value" not in df.columns:
        return qs
    panels = df["panel"].unique().tolist()
    if len(panels) < 2:
        return qs
    panel = random.choice(panels)
    sub = df[df["panel"] == panel]
    if len(sub) < 2:
        return qs
    # Text first: ranking across panels
    try:
        panel_maxes = df.groupby("panel")["Value"].max()
        best_panel = str(panel_maxes.idxmax())
        qs.append({"question": "Which panel reaches the highest peak value across all panels?",
                   "answer": best_panel, "answer_type": "text",
                   "template": "sci_panel_ranking", "difficulty": "hard",
                   "reasoning_steps": [f"Per-panel maxima: {dict(panel_maxes.round(3))}",
                                        f"Argmax of maxima = '{best_panel}'"]})
    except Exception:
        pass
    try:
        peak_idx = sub["Value"].idxmax()
        peak_epoch = round(float(df.loc[peak_idx, "Epoch"]), 1)
        peak_val = round(float(df.loc[peak_idx, "Value"]), 3)
    except Exception:
        return qs[:max_per_chart]
    qs.append({"question": f"At which epoch does the '{panel}' panel reach its peak value?",
               "answer": peak_epoch, "answer_type": "numeric",
               "template": "sci_panel_peak_epoch", "difficulty": "hard",
               "reasoning_steps": [f"Filter rows where panel='{panel}'",
                                    f"argmax over Value gives Epoch={peak_epoch}"]})
    qs.append({"question": f"What is the maximum value reached in the '{panel}' panel?",
               "answer": peak_val, "answer_type": "numeric",
               "template": "sci_panel_max", "difficulty": "medium",
               "reasoning_steps": [f"max(Value) over panel='{panel}' = {peak_val}"]})
    try:
        final_row = df[df["panel"] == panel].sort_values("Epoch").iloc[-1]
        fv = round(float(final_row["Value"]), 3)
        fe = round(float(final_row["Epoch"]), 1)
        qs.append({"question": f"What is the final '{panel}' value at the last epoch?",
                   "answer": fv, "answer_type": "numeric",
                   "template": "sci_panel_final", "difficulty": "medium",
                   "reasoning_steps": [f"Last epoch for '{panel}' = {fe}",
                                        f"Value at last epoch = {fv}"]})
    except Exception:
        pass
    try:
        stabilities = {}
        for p in panels:
            sp = df[df["panel"] == p].sort_values("Epoch")
            if len(sp) >= 4:
                stabilities[p] = float(sp["Value"].iloc[len(sp)//2:].std())
        if stabilities:
            stable_panel = min(stabilities, key=stabilities.get)
            qs.append({"question": "Which panel converges most stably (lowest variance in final half)?",
                       "answer": str(stable_panel), "answer_type": "text",
                       "template": "sci_panel_convergence", "difficulty": "very_hard",
                       "reasoning_steps": [f"Final-half std per panel: {dict((p,round(v,4)) for p,v in stabilities.items())}",
                                            f"Min std = '{stable_panel}'"]})
    except Exception:
        pass
    return qs[:max_per_chart]


def _gen_heatmap_corr_qa(df, meta, max_per_chart=5):
    qs = []
    var_names = (meta.get("var_names") if meta else None) or list(df.columns)
    n = len(var_names)
    if n < 3 or df.shape[0] < n or df.shape[1] < n:
        return qs
    try:
        mat = df.iloc[:n, :n].values.astype(float)
    except Exception:
        return qs
    # Text first: strongest pair
    upper = [(var_names[a], var_names[b], float(mat[a, b]))
             for a in range(n) for b in range(a + 1, n)]
    if upper:
        strongest = max(upper, key=lambda t: t[2])
        qs.append({"question": "Which pair of variables has the strongest positive correlation?",
                   "answer": f"{strongest[0]} and {strongest[1]}", "answer_type": "text",
                   "template": "sci_corr_strongest", "difficulty": "hard",
                   "reasoning_steps": [f"Max off-diagonal correlation = {round(strongest[2],3)}",
                                        f"Pair: {strongest[0]} and {strongest[1]}"]})
    # Text: most correlated with target
    target_i = random.randint(0, n - 1)
    target = var_names[target_i]
    col = mat[:, target_i].copy()
    col[target_i] = -2.0
    best = var_names[int(col.argmax())]
    qs.append({"question": f"Which variable is most positively correlated with {target}?",
               "answer": str(best), "answer_type": "text",
               "template": "sci_corr_target", "difficulty": "medium",
               "reasoning_steps": [f"Column for {target} (excluding self): argmax = {best}"]})
    # Numeric: specific cell
    i = 0
    j = random.randint(1, n - 1)
    v1, v2 = var_names[i], var_names[j]
    cell = round(float(mat[i, j]), 3)
    qs.append({"question": f"What is the correlation between {v1} and {v2}?",
               "answer": cell, "answer_type": "numeric",
               "template": "sci_corr_cell", "difficulty": "medium",
               "reasoning_steps": [f"Read matrix cell ({v1}, {v2}) = {cell}"]})
    # Text: sign
    sign = "positively" if cell > 0 else "negatively"
    qs.append({"question": f"Are {v1} and {v2} positively or negatively correlated?",
               "answer": f"{sign} correlated", "answer_type": "text",
               "template": "sci_corr_sign", "difficulty": "easy",
               "reasoning_steps": [f"Correlation = {cell} → {sign}"]})
    if upper:
        weakest = min(upper, key=lambda t: t[2])
        qs.append({"question": "What is the lowest correlation value between any two distinct variables?",
                   "answer": round(weakest[2], 3), "answer_type": "numeric",
                   "template": "sci_corr_min", "difficulty": "hard",
                   "reasoning_steps": [f"Min off-diagonal = {round(weakest[2],3)}"]})
    return qs[:max_per_chart]


def _gen_errorbar_comparison_qa(df, meta, max_per_chart=4):
    qs = []
    if df.shape[1] < 2 or len(df) < 2:
        return qs
    method_col = df.columns[0]
    metric_col = df.columns[1]
    std_col = df.columns[2] if len(df.columns) > 2 and str(df.columns[2]).endswith("_std") else None
    try:
        best_idx = df[metric_col].idxmax()
        worst_idx = df[metric_col].idxmin()
        best = str(df.loc[best_idx, method_col])
        worst = str(df.loc[worst_idx, method_col])
    except Exception:
        return qs
    # Text first
    qs.append({"question": f"Which method achieves the highest {metric_col}?",
               "answer": best, "answer_type": "text",
               "template": "sci_eb_best", "difficulty": "easy",
               "reasoning_steps": [f"argmax over {metric_col} = {best}"]})
    qs.append({"question": f"Which method achieves the lowest {metric_col}?",
               "answer": worst, "answer_type": "text",
               "template": "sci_eb_worst", "difficulty": "easy",
               "reasoning_steps": [f"argmin over {metric_col} = {worst}"]})
    if std_col:
        try:
            low_idx = df[std_col].idxmin()
            low = str(df.loc[low_idx, method_col])
            qs.append({"question": f"Which method has the smallest standard deviation in {metric_col}?",
                       "answer": low, "answer_type": "text",
                       "template": "sci_eb_low_std", "difficulty": "medium",
                       "reasoning_steps": [f"argmin over {std_col} = {low}"]})
        except Exception:
            pass
    try:
        diff = round(float(df.loc[best_idx, metric_col] - df.loc[worst_idx, metric_col]), 2)
        qs.append({"question": f"By how much does {best} outperform {worst} in {metric_col}?",
                   "answer": diff, "answer_type": "numeric",
                   "template": "sci_eb_diff", "difficulty": "hard",
                   "reasoning_steps": [f"{best} = {round(float(df.loc[best_idx,metric_col]),2)}",
                                        f"{worst} = {round(float(df.loc[worst_idx,metric_col]),2)}",
                                        f"Difference = {diff}"]})
    except Exception:
        pass
    try:
        idx = random.randint(0, len(df) - 1)
        m = str(df.iloc[idx][method_col])
        v = round(float(df.iloc[idx][metric_col]), 2)
        qs.append({"question": f"What is the {metric_col} of {m}?",
                   "answer": v, "answer_type": "numeric",
                   "template": "sci_eb_lookup", "difficulty": "easy",
                   "reasoning_steps": [f"Row {m}: {metric_col} = {v}"]})
    except Exception:
        pass
    return qs[:max_per_chart]


def _gen_logscale_plot_qa(df, meta, max_per_chart=3):
    """logscale_plot: (x, y) log y. Text-first ordering for CharXiv text ratio."""
    qs = []
    if df.shape[1] < 2 or len(df) < 3:
        return qs
    x_col, y_col = df.columns[0], df.columns[1]
    # Text: trend direction
    try:
        slope = float(np.polyfit(range(len(df)), df[y_col].astype(float), 1)[0])
        direction = "increasing" if slope > 0 else "decreasing"
        qs.append({"question": f"Is {y_col} increasing or decreasing overall?",
                   "answer": direction, "answer_type": "text",
                   "template": "sci_log_trend", "difficulty": "easy",
                   "reasoning_steps": [f"Overall linear slope = {round(slope,3)}", f"→ {direction}"]})
    except Exception:
        pass
    # Text: growth pattern
    try:
        y_vals = df[y_col].astype(float).values
        y_first = float(y_vals[0])
        y_last = float(y_vals[-1])
        if y_first > 0 and y_last > 0:
            ratio = y_last / y_first
            if ratio > 100:
                pattern = "exponential growth"
            elif ratio > 10:
                pattern = "rapid growth"
            elif ratio > 2:
                pattern = "moderate growth"
            elif ratio > 0.5:
                pattern = "stable"
            else:
                pattern = "rapid decay"
            qs.append({"question": f"What growth pattern best describes {y_col} across the range of {x_col}?",
                       "answer": pattern, "answer_type": "text",
                       "template": "sci_log_pattern", "difficulty": "medium",
                       "reasoning_steps": [f"Ratio last/first = {round(ratio,2)}", f"→ {pattern}"]})
    except Exception:
        pass
    # Numeric: lookup
    try:
        idx = len(df) // 2
        xv = round(float(df.iloc[idx][x_col]), 2)
        yv = round(float(df.iloc[idx][y_col]), 2)
        qs.append({"question": f"What is the {y_col} when {x_col} is {xv}?",
                   "answer": yv, "answer_type": "numeric",
                   "template": "sci_log_lookup", "difficulty": "medium",
                   "reasoning_steps": [f"Row at {x_col}={xv} → {y_col}={yv}"]})
    except Exception:
        pass
    try:
        y_first = float(df.iloc[0][y_col])
        y_last = float(df.iloc[-1][y_col])
        if y_first > 0 and y_last > 0:
            oom = round(float(np.log10(y_last / y_first)), 2)
            qs.append({"question": f"Over how many orders of magnitude does {y_col} change from first to last row?",
                       "answer": oom, "answer_type": "numeric",
                       "template": "sci_log_oom", "difficulty": "very_hard",
                       "reasoning_steps": [f"log10({round(y_last,4)} / {round(y_first,4)}) = {oom}"]})
    except Exception:
        pass
    return qs[:max_per_chart]


def _gen_dual_axis_qa(df, meta, max_per_chart=4):
    qs = []
    if df.shape[1] < 3 or len(df) < 3:
        return qs
    x_col, m1, m2 = df.columns[0], df.columns[1], df.columns[2]
    try:
        idx = len(df) // 2
        v1 = float(df.iloc[idx][m1])
        v2 = float(df.iloc[idx][m2])
        xv = round(float(df.iloc[idx][x_col]), 2)
    except Exception:
        return qs
    if v2 != 0:
        ratio = round(v1 / v2, 3)
        qs.append({"question": f"What is the ratio of {m1} to {m2} at {x_col}={xv}?",
                   "answer": ratio, "answer_type": "numeric",
                   "template": "sci_dual_ratio", "difficulty": "hard",
                   "reasoning_steps": [f"{m1}={round(v1,2)}, {m2}={round(v2,2)}",
                                        f"Ratio = {ratio}"]})
    higher = m1 if v1 > v2 else m2
    qs.append({"question": f"At {x_col}={xv}, is {m1} or {m2} higher?",
               "answer": higher, "answer_type": "text",
               "template": "sci_dual_which", "difficulty": "easy",
               "reasoning_steps": [f"{m1}={round(v1,2)}, {m2}={round(v2,2)}", f"→ {higher}"]})
    try:
        peak_idx = df[m1].idxmax()
        peak_x = round(float(df.loc[peak_idx, x_col]), 2)
        qs.append({"question": f"At which {x_col} does {m1} reach its maximum?",
                   "answer": peak_x, "answer_type": "numeric",
                   "template": "sci_dual_peak", "difficulty": "medium",
                   "reasoning_steps": [f"argmax over {m1} → {x_col}={peak_x}"]})
    except Exception:
        pass
    try:
        mean1 = round(float(df[m1].mean()), 2)
        mean2 = round(float(df[m2].mean()), 2)
        dominant = m1 if mean1 > mean2 else m2
        qs.append({"question": f"On average, which metric is larger in magnitude: {m1} or {m2}?",
                   "answer": dominant, "answer_type": "text",
                   "template": "sci_dual_dominant", "difficulty": "medium",
                   "reasoning_steps": [f"mean({m1})={mean1}, mean({m2})={mean2}", f"→ {dominant}"]})
    except Exception:
        pass
    return qs[:max_per_chart]


def _gen_confidence_band_qa(df, meta, max_per_chart=4):
    qs = []
    needed = {"x", "Method", "y", "y_lower", "y_upper"}
    if not needed.issubset(set(df.columns)):
        return qs
    methods = df["Method"].unique().tolist()
    if len(methods) < 1:
        return qs
    m = random.choice(methods)
    sub = df[df["Method"] == m]
    if len(sub) < 2:
        return qs
    try:
        mid_idx = sub.index[len(sub) // 2]
        xv = round(float(df.loc[mid_idx, "x"]), 2)
        width = round(float(df.loc[mid_idx, "y_upper"] - df.loc[mid_idx, "y_lower"]), 3)
        yv = round(float(df.loc[mid_idx, "y"]), 3)
    except Exception:
        return qs
    qs.append({"question": f"What is the confidence interval width for {m} at x={xv}?",
               "answer": width, "answer_type": "numeric",
               "template": "sci_cb_width", "difficulty": "hard",
               "reasoning_steps": [f"y_upper={round(float(df.loc[mid_idx,'y_upper']),3)}",
                                    f"y_lower={round(float(df.loc[mid_idx,'y_lower']),3)}",
                                    f"width = upper - lower = {width}"]})
    qs.append({"question": f"What is the y value of {m} at x={xv}?",
               "answer": yv, "answer_type": "numeric",
               "template": "sci_cb_lookup", "difficulty": "medium",
               "reasoning_steps": [f"Row (Method={m}, x={xv}) → y={yv}"]})
    # CI width average (works for single-method too: answer is the only method's mean)
    try:
        df2 = df.copy()
        df2["_width"] = df2["y_upper"] - df2["y_lower"]
        mean_w = round(float(df2[df2["Method"] == m]["_width"].mean()), 3)
        qs.append({"question": f"What is the average confidence interval width for {m}?",
                   "answer": mean_w, "answer_type": "numeric",
                   "template": "sci_cb_mean_width", "difficulty": "hard",
                   "reasoning_steps": [f"(y_upper - y_lower) averaged over rows of {m} = {mean_w}"]})
    except Exception:
        pass
    # Max y for single method
    try:
        sub_y = df[df["Method"] == m]["y"]
        peak_y = round(float(sub_y.max()), 3)
        qs.append({"question": f"What is the maximum y value reached by {m}?",
                   "answer": peak_y, "answer_type": "numeric",
                   "template": "sci_cb_max_y", "difficulty": "medium",
                   "reasoning_steps": [f"max(y) for {m} = {peak_y}"]})
    except Exception:
        pass
    # Only add multi-method ranking questions if 2+ methods exist
    if len(methods) >= 2:
        try:
            df2 = df.copy()
            df2["_width"] = df2["y_upper"] - df2["y_lower"]
            widths = df2.groupby("Method")["_width"].mean()
            widest = str(widths.idxmax())
            qs.append({"question": "Which method has the widest average confidence interval?",
                       "answer": widest, "answer_type": "text",
                       "template": "sci_cb_widest", "difficulty": "hard",
                       "reasoning_steps": [f"Mean CI width per method: {dict(widths.round(3))}",
                                            f"Argmax → {widest}"]})
        except Exception:
            pass
        try:
            means = df.groupby("Method")["y"].mean()
            best = str(means.idxmax())
            qs.append({"question": "Which method achieves the highest average y value?",
                       "answer": best, "answer_type": "text",
                       "template": "sci_cb_best", "difficulty": "medium",
                       "reasoning_steps": [f"Mean y per method: {dict(means.round(3))}",
                                            f"Argmax → {best}"]})
        except Exception:
            pass
    return qs[:max_per_chart]


def _gen_violin_comparison_qa(df, meta, max_per_chart=5):
    qs = []
    if df.shape[1] < 2 or len(df) < 5:
        return qs
    method_col = df.columns[0]
    metric_col = df.columns[1]
    methods = df[method_col].unique().tolist()
    if len(methods) < 2:
        return qs
    m = random.choice(methods)
    vals = df[df[method_col] == m][metric_col]
    if len(vals) < 2:
        return qs
    try:
        med = round(float(vals.median()), 3)
        rng = round(float(vals.max() - vals.min()), 3)
    except Exception:
        return qs
    qs.append({"question": f"What is the median {metric_col} of {m}?",
               "answer": med, "answer_type": "numeric",
               "template": "sci_violin_median", "difficulty": "medium",
               "reasoning_steps": [f"median({m}) = {med}"]})
    qs.append({"question": f"What is the range (max minus min) of {metric_col} for {m}?",
               "answer": rng, "answer_type": "numeric",
               "template": "sci_violin_range", "difficulty": "hard",
               "reasoning_steps": [f"max={round(float(vals.max()),3)}, min={round(float(vals.min()),3)}",
                                    f"range = {rng}"]})
    try:
        medians = df.groupby(method_col)[metric_col].median()
        highest = str(medians.idxmax())
        qs.append({"question": f"Which method has the highest median {metric_col}?",
                   "answer": highest, "answer_type": "text",
                   "template": "sci_violin_highest_median", "difficulty": "medium",
                   "reasoning_steps": [f"Medians per method: {dict(medians.round(3))}",
                                        f"Argmax → {highest}"]})
    except Exception:
        pass
    try:
        stds = df.groupby(method_col)[metric_col].std()
        widest = str(stds.idxmax())
        qs.append({"question": f"Which method has the widest spread in {metric_col}?",
                   "answer": widest, "answer_type": "text",
                   "template": "sci_violin_widest", "difficulty": "hard",
                   "reasoning_steps": [f"Stds per method: {dict(stds.round(3))}",
                                        f"Argmax → {widest}"]})
    except Exception:
        pass
    try:
        q75 = float(vals.quantile(0.75))
        q25 = float(vals.quantile(0.25))
        iqr = round(q75 - q25, 3)
        qs.append({"question": f"What is the interquartile range (IQR) of {metric_col} for {m}?",
                   "answer": iqr, "answer_type": "numeric",
                   "template": "sci_violin_iqr", "difficulty": "very_hard",
                   "reasoning_steps": [f"Q75={round(q75,3)}, Q25={round(q25,3)}",
                                        f"IQR = Q75 - Q25 = {iqr}"]})
    except Exception:
        pass
    return qs[:max_per_chart]


def _gen_v2_simple_qa(df, meta, max_per_chart=4):
    """v2 pool: bar_scientific, line, scatter, scatter_line. 2-col (x, y) numeric.
    Text-first ordering to improve overall text answer ratio for CharXiv alignment.
    """
    qs = []
    if df.shape[1] < 2 or len(df) < 3:
        return qs
    x_col, y_col = df.columns[0], df.columns[1]
    # Text: overall trend
    try:
        slope = float(np.polyfit(range(len(df)), df[y_col].astype(float), 1)[0])
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "flat"
        qs.append({"question": f"Is the overall trend of {y_col} increasing, decreasing, or flat?",
                   "answer": direction, "answer_type": "text",
                   "template": "v2_trend", "difficulty": "easy",
                   "reasoning_steps": [f"Overall linear slope = {round(slope,3)} → {direction}"]})
    except Exception:
        pass
    # Text: monotonicity
    try:
        diffs = np.diff(df[y_col].astype(float).values)
        if all(d >= 0 for d in diffs):
            mono = "monotonically increasing"
        elif all(d <= 0 for d in diffs):
            mono = "monotonically decreasing"
        else:
            mono = "non-monotonic"
        qs.append({"question": f"Is {y_col} monotonic as {x_col} increases?",
                   "answer": mono, "answer_type": "text",
                   "template": "v2_monotonic", "difficulty": "medium",
                   "reasoning_steps": [f"Examine consecutive differences", f"→ {mono}"]})
    except Exception:
        pass
    # Numeric: peak x
    try:
        peak_idx = df[y_col].idxmax()
        peak_x = round(float(df.loc[peak_idx, x_col]), 2)
        peak_y = round(float(df.loc[peak_idx, y_col]), 2)
        qs.append({"question": f"At which {x_col} does {y_col} reach its maximum value?",
                   "answer": peak_x, "answer_type": "numeric",
                   "template": "v2_peak_x", "difficulty": "medium",
                   "reasoning_steps": [f"argmax over {y_col} → {x_col}={peak_x}"]})
        qs.append({"question": f"What is the maximum value of {y_col}?",
                   "answer": peak_y, "answer_type": "numeric",
                   "template": "v2_max_y", "difficulty": "easy",
                   "reasoning_steps": [f"max({y_col}) = {peak_y}"]})
    except Exception:
        pass
    try:
        mean_y = round(float(df[y_col].mean()), 2)
        qs.append({"question": f"What is the average value of {y_col}?",
                   "answer": mean_y, "answer_type": "numeric",
                   "template": "v2_mean", "difficulty": "hard",
                   "reasoning_steps": [f"mean({y_col}) = {mean_y}"]})
    except Exception:
        pass
    return qs[:max_per_chart]


# Dispatcher registry
SCI_DISPATCHERS = {
    "scatter_regression":  _gen_scatter_regression_qa,
    "subplot_panel":       _gen_subplot_panel_qa,
    "heatmap_corr":        _gen_heatmap_corr_qa,
    "errorbar_comparison": _gen_errorbar_comparison_qa,
    "logscale_plot":       _gen_logscale_plot_qa,
    "dual_axis":           _gen_dual_axis_qa,
    "confidence_band":     _gen_confidence_band_qa,
    "violin_comparison":   _gen_violin_comparison_qa,
    "bar_scientific":      _gen_v2_simple_qa,
    "line":                _gen_v2_simple_qa,
    "scatter":             _gen_v2_simple_qa,
    "scatter_line":        _gen_v2_simple_qa,
}


def generate_block_c_questions(csv_path, chart_meta=None, max_per_chart=5):
    """Block C v9.1: per-chart-type dispatcher.

    Routes to a dedicated dispatcher based on chart_meta['chart_type'].
    Falls back to legacy generator for unknown types.
    """
    if chart_meta is None:
        return generate_block_c_questions_legacy(csv_path, chart_meta, max_per_chart)
    ct = chart_meta.get("chart_type", "")
    dispatcher = SCI_DISPATCHERS.get(ct)
    if dispatcher is None:
        return generate_block_c_questions_legacy(csv_path, chart_meta, max_per_chart)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    try:
        return dispatcher(df, chart_meta, max_per_chart=max_per_chart)
    except Exception:
        return []


# ═══════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════

BLOCK_GENERATORS = {
    "a": generate_block_a_questions,
    "b": generate_block_b_questions,
    "c": generate_block_c_questions,
    "f": generate_block_f_questions,
}

BLOCK_MAX_PER_CHART = {
    "a": 8,
    "b": 4,
    "c": 5,
    "f": 4,
}


def main():
    parser = argparse.ArgumentParser(description="Extended rule-based QA v9")
    parser.add_argument("--manifests", nargs="+", required=True, help="Manifest JSONL files")
    parser.add_argument("--block", choices=["a", "b", "c", "f"], required=True, help="Block type")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max total QA pairs to generate")
    parser.add_argument("--max_per_chart", type=int, default=None, help="Override max QA per chart")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    gen_fn = BLOCK_GENERATORS[args.block]
    max_pc = args.max_per_chart or BLOCK_MAX_PER_CHART[args.block]

    # Load manifests
    charts = []
    for mpath in args.manifests:
        if not os.path.exists(mpath):
            print(f"  Skipping {mpath} (not found)")
            continue
        with open(mpath) as f:
            for line in f:
                try:
                    charts.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(charts)} charts from {len(args.manifests)} manifests")

    # Resume
    done_slugs = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    done_slugs.add(json.loads(line).get("chart_slug", ""))
                except Exception:
                    pass
    pending = [c for c in charts if c.get("slug", "") not in done_slugs]
    print(f"  Done slugs: {len(done_slugs)}, Pending charts: {len(pending)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a")

    total_qa = 0
    errors = 0
    limit = args.limit or float("inf")

    for chart in tqdm(pending, desc=f"Block-{args.block} QA", unit="chart"):
        if total_qa >= limit:
            break

        csv_path = chart.get("csv_path", "")
        if not csv_path or not os.path.exists(csv_path):
            errors += 1
            continue

        questions = gen_fn(csv_path, chart_meta=chart, max_per_chart=max_pc)
        for qi, qa in enumerate(questions):
            if total_qa >= limit:
                break
            qa.update({
                "csv_path": csv_path,
                "image_path": chart.get("image_path", ""),
                "source": f"block_{args.block}",
                "chart_slug": f"b{args.block}_{chart.get('slug', '')}_{qi}",
                "verified": True,
                "generation_method": "rule_based_v9",
            })
            out_f.write(json.dumps(qa, ensure_ascii=False, default=str) + "\n")
            total_qa += 1

        if total_qa % 1000 == 0 and total_qa > 0:
            out_f.flush()

    out_f.close()

    # Stats
    print(f"\nGenerated {total_qa} QAs from {len(pending)} charts (errors={errors})")

    templates = {}
    answer_types = {}
    with open(args.output) as f:
        for line in f:
            try:
                rec = json.loads(line)
                t = rec.get("template", "unknown")
                templates[t] = templates.get(t, 0) + 1
                at = rec.get("answer_type", "numeric")
                answer_types[at] = answer_types.get(at, 0) + 1
            except Exception:
                pass

    total = sum(templates.values())
    print(f"\nTotal QAs: {total}")
    print(f"\nAnswer types:")
    for at, c in sorted(answer_types.items(), key=lambda x: -x[1]):
        print(f"  {at}: {c} ({c/max(total,1)*100:.1f}%)")
    print(f"\nTemplates:")
    for t, c in sorted(templates.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({c/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
