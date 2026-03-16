"""
ChartVCR: Chart-Verifiable Causal Reward

THE MAIN CONTRIBUTION. This file implements the complete reward function
compatible with BigCharts-R1's GRPO trainer interface.

Reward function signature: def fn(completions, solution, csv_path, **kwargs) -> list[float]
Answer format: <thinking>...</thinking><answer>...</answer> (BigCharts-R1 format)

Pilot study decision: rule-based only (no LLM verifier).
"""
import re
import math
import os
import pandas as pd
from typing import Optional, List, Set

import sys
# Insert chartvr root (3 levels up from code/rewards/chart_vcr_reward.py)
_chartvr_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _chartvr_root not in sys.path:
    sys.path.insert(0, _chartvr_root)

from code.rewards.sentence_parser import parse_cot_to_sentences, extract_numbers
from code.rewards.causal_attribution import (
    compute_causal_rewards, value_accuracy_score, find_closest_table_value
)


def _load_table_values(csv_path: str) -> Set[float]:
    """Extract all numeric values from a chart's data table."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()

    values = set()
    for col in df.columns:
        for val in df[col]:
            try:
                values.add(float(val))
            except (ValueError, TypeError):
                pass
    return values


def _extract_answer(content: str) -> Optional[str]:
    """Extract answer from BigCharts-R1 format: <answer>...</answer>"""
    match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_thinking(content: str) -> str:
    """Extract thinking content from <thinking>...</thinking>"""
    match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def _compute_accuracy_reward(content: str, gold: str) -> float:
    """Continuous relaxed accuracy (same as BigCharts-R1 error_magnitude_reward)."""
    answer = _extract_answer(content)
    if answer is None:
        return 0.0

    def _to_float(text):
        try:
            # Do NOT divide by 100 for percentages — ChartQA convention
            return float(text.rstrip("%"))
        except ValueError:
            return None

    pred_f = _to_float(answer)
    gold_f = _to_float(gold)

    if pred_f is not None and gold_f is not None:
        if gold_f == 0:
            return 1.0 if pred_f == 0 else 0.0
        relative_change = abs(pred_f - gold_f) / abs(gold_f)
        return 1.0 / (1.0 + relative_change)

    return 1.0 if answer.strip().lower() == gold.strip().lower() else 0.0


def _compute_format_reward(content: str) -> float:
    """Format compliance reward (same as BigCharts-R1)."""
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    if re.fullmatch(pattern, content.strip(), re.DOTALL):
        return 1.0
    return 0.0


def _compute_process_reward(content: str, csv_path: str, sigma: float = 0.10) -> float:
    """
    Process reward with causal attribution.
    THIS IS THE CORE NOVELTY.

    Uses rule-based verification only (per pilot study decision).
    """
    table_values = _load_table_values(csv_path)
    if not table_values:
        return 0.0

    # Extract thinking content for analysis
    thinking = _extract_thinking(content)
    if not thinking:
        return 0.0

    # Parse CoT into sentences
    parsed = parse_cot_to_sentences(thinking)
    if not parsed:
        return 0.0

    # Phase 1: Verify each sentence (rule-based)
    sentence_data = []
    for sent in parsed:
        if len(sent.numbers) == 0:
            raw_label = "not_verifiable"
            comp_correct = None
        elif sent.has_computation and sent.computation_result:
            a, b, op, stated = sent.computation_result
            try:
                if op == '+':
                    expected = a + b
                elif op == '-':
                    expected = a - b
                elif op == '*':
                    expected = a * b
                elif op == '/':
                    expected = a / b if b != 0 else float('inf')
                else:
                    expected = None

                if expected is not None and not math.isinf(expected):
                    comp_correct = abs(stated - expected) / max(abs(expected), 1e-10) < 0.05
                else:
                    comp_correct = None
            except Exception:
                comp_correct = None

            # Use actual operands from computation, not first 2 numbers in sentence
            operands = [a, b]
            in_table = any(
                find_closest_table_value(n, table_values)[1] > 0.5
                for n in operands
            )

            if comp_correct is True and in_table:
                raw_label = "correct"
            elif comp_correct is False:
                raw_label = "incorrect"
            else:
                raw_label = "not_verifiable"
        elif len(sent.numbers) >= 1:
            best_score = max(
                find_closest_table_value(n, table_values)[1]
                for n in sent.numbers
            )
            if best_score > 0.7:
                raw_label = "correct"
            elif best_score < 0.3:
                raw_label = "incorrect"
            else:
                raw_label = "not_verifiable"
        else:
            raw_label = "not_verifiable"

        sentence_data.append({
            "text": sent.text,
            "index": sent.index,
            "numbers": sent.numbers,
            "raw_label": raw_label,
            "computation_correct": comp_correct if sent.has_computation else None
        })

    # Phase 2: Causal attribution (rule-based)
    attributed = compute_causal_rewards(sentence_data, table_values, sigma=sigma)

    # Phase 3: Aggregate sentence rewards (exclude not_verifiable)
    valid_rewards = [
        s.sentence_reward for s in attributed
        if s.causal_label != "not_verifiable"
    ]

    if not valid_rewards:
        return 0.0

    return sum(valid_rewards) / len(valid_rewards)


# ============================================================
# BigCharts-R1 GRPO Interface Functions
# ============================================================

def chart_vcr_reward(completions, solution, csv_path=None,
                     w_accuracy=0.5, w_process=0.3, w_format=0.2,
                     sigma=0.10, **kwargs) -> List[float]:
    """
    Complete ChartVCR reward function for BigCharts-R1 GRPO trainer.

    R = w_accuracy × R_acc + w_process × R_proc + w_format × R_fmt

    Args:
        completions: List of completions, each is [{"role": "assistant", "content": "..."}]
        solution: List of gold answers (str), repeated per num_generations
        csv_path: List of CSV file paths (str), repeated per num_generations
        **kwargs: Other fields from dataset (ignored)

    Returns:
        List[float]: Reward for each completion
    """
    contents = [completion[0].get("content", "") if completion else "" for completion in completions]
    rewards = []
    csv_paths = csv_path if csv_path else [None] * len(contents)

    for content, sol, csv in zip(contents, solution, csv_paths):
        r_acc = _compute_accuracy_reward(content, sol)
        r_fmt = _compute_format_reward(content)

        if csv and os.path.exists(csv):
            r_proc = _compute_process_reward(content, csv, sigma=sigma)
        else:
            r_proc = 0.0

        total = w_accuracy * r_acc + w_process * r_proc + w_format * r_fmt

        if math.isnan(total) or math.isinf(total):
            total = 0.0

        rewards.append(total)

    return rewards


def chart_vcr_accuracy_only(completions, solution, **kwargs) -> List[float]:
    """Accuracy-only reward (for ablation A1a — same as BigCharts-R1 baseline)."""
    contents = [completion[0]["content"] for completion in completions]
    return [_compute_accuracy_reward(c, s) for c, s in zip(contents, solution)]


def chart_vcr_no_format(completions, solution, csv_path=None, **kwargs) -> List[float]:
    """Accuracy + Process reward, no format (for ablation A1b)."""
    return chart_vcr_reward(completions, solution, csv_path,
                            w_accuracy=0.55, w_process=0.45, w_format=0.0, **kwargs)


def chart_vcr_equal_penalty(completions, solution, csv_path=None,
                            sigma=0.10, **kwargs) -> List[float]:
    """
    ChartVCR with equal penalty (no causal distinction) — for ablation A2b.
    All incorrect sentences get logic_score=0.0, regardless of source/propagated.
    """
    contents = [completion[0].get("content", "") if completion else "" for completion in completions]
    rewards = []
    csv_paths = csv_path if csv_path else [None] * len(contents)

    for content, sol, csv in zip(contents, solution, csv_paths):
        r_acc = _compute_accuracy_reward(content, sol)
        r_fmt = _compute_format_reward(content)

        if csv and os.path.exists(csv):
            # Modified process reward: no causal distinction
            table_values = _load_table_values(csv)
            thinking = _extract_thinking(content)
            parsed = parse_cot_to_sentences(thinking) if thinking else []

            if parsed and table_values:
                sentence_scores = []
                for sent in parsed:
                    if len(sent.numbers) == 0:
                        continue
                    best = max(
                        find_closest_table_value(n, table_values)[1]
                        for n in sent.numbers
                    )
                    sentence_scores.append(best)

                r_proc = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
            else:
                r_proc = 0.0
        else:
            r_proc = 0.0

        total = 0.5 * r_acc + 0.3 * r_proc + 0.2 * r_fmt
        rewards.append(total if not (math.isnan(total) or math.isinf(total)) else 0.0)

    return rewards
