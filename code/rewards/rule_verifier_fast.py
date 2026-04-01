"""
Pure rule-based process reward verifier. No LLM calls. ~1ms/completion.

Two components:
  (1) Value Grounding: numbers in reasoning that match CSV values
  (2) Arithmetic Verification: "a op b = c" recalculated in Python
"""
import re
import pandas as pd
from typing import Set

NUM_RE = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
CALC_RE = rf'({NUM_RE})\s*([+\-*/×÷])\s*({NUM_RE})\s*[=≈]\s*({NUM_RE})'

# Years to exclude from value matching (false positives for OWID data)
YEAR_MIN, YEAR_MAX = 1900, 2030


def _parse_num(s: str) -> float:
    return float(s.replace(',', ''))


def _is_likely_year(n: float) -> bool:
    return n == int(n) and YEAR_MIN <= n <= YEAR_MAX


def get_table_values(csv_path: str) -> Set[float]:
    """Extract all numeric values from CSV."""
    try:
        df = pd.read_csv(csv_path)
        vals = set()
        for col in df.columns:
            for v in df[col]:
                try:
                    vals.add(float(v))
                except (ValueError, TypeError):
                    pass
        return vals
    except Exception:
        return set()


def compute_process_reward_fast(response: str, csv_path: str) -> float:
    """
    Pure rule-based process reward. ~1ms/completion.

    Returns float in [0, 1].

    Value Grounding (weight 0.6): fraction of reasoning numbers found in CSV.
    Arithmetic (weight 0.4): fraction of correct calculations.
    """
    table_values = get_table_values(csv_path)
    if not table_values:
        return 0.5  # no CSV data -> neutral

    # Extract reasoning: try multiple formats
    # Format 1: <think>reasoning</think> (full tags)
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1)
    # Format 2: reasoning</think>content (opening tag in prompt, not completion)
    elif '</think>' in response:
        reasoning = response[:response.index('</think>')]
    else:
        # No think tags — use everything before <answer> or full response
        answer_match = re.search(r'<answer>', response)
        reasoning = response[:answer_match.start()] if answer_match else response

    if not reasoning.strip():
        return 0.0

    # Extract numbers from reasoning
    numbers = []
    for m in re.findall(NUM_RE, reasoning):
        try:
            n = _parse_num(m)
            if not _is_likely_year(n):
                numbers.append(n)
        except ValueError:
            pass

    if not numbers:
        return 0.0  # no numbers in reasoning -> worst

    # (1) Value Grounding
    matched = 0
    for n in numbers:
        if abs(n) < 1e-10:
            # Zero is common and not meaningful for matching
            continue
        best_score = max(
            (1.0 / (1.0 + abs(n - tv) / max(abs(tv), 1e-10))
             for tv in table_values),
            default=0.0
        )
        if best_score > 0.8:  # ~20% tolerance
            matched += 1
    grounding = matched / len(numbers) if numbers else 0.0

    # (2) Arithmetic Verification
    calc_matches = re.findall(CALC_RE, reasoning)
    if calc_matches:
        correct = 0
        for m in calc_matches:
            try:
                a = _parse_num(m[0])
                op = m[1].replace('×', '*').replace('÷', '/')
                b = _parse_num(m[2])
                stated = _parse_num(m[3])

                if op == '+':
                    expected = a + b
                elif op == '-':
                    expected = a - b
                elif op == '*':
                    expected = a * b
                elif op == '/' and b != 0:
                    expected = a / b
                else:
                    correct += 1  # unverifiable -> no penalty
                    continue

                if expected != 0:
                    if abs(stated - expected) / abs(expected) < 0.05:
                        correct += 1
                elif abs(stated) < 0.01:
                    correct += 1
            except Exception:
                correct += 1  # parse failure -> no penalty
        arithmetic = correct / len(calc_matches)
    else:
        arithmetic = 0.5  # no calculations -> neutral

    return 0.6 * grounding + 0.4 * arithmetic
