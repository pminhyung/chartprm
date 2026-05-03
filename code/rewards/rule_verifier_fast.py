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


def compute_process_components_fast(response: str, csv_path: str):
    """Return (grounding, arithmetic) components in [0, 1].

    Used by compute_process_reward_fast and VAPV component ablations
    (reward_conditional_v2_value_only / _arith_only).

    Signal conventions when CSV absent or reasoning empty: (0.5, 0.5) neutral.
    """
    table_values = get_table_values(csv_path)
    if not table_values:
        return 0.5, 0.5

    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1)
    elif '</think>' in response:
        reasoning = response[:response.index('</think>')]
    else:
        answer_match = re.search(r'<answer>', response)
        reasoning = response[:answer_match.start()] if answer_match else response

    if not reasoning.strip():
        return 0.0, 0.0

    numbers = []
    for m in re.findall(NUM_RE, reasoning):
        try:
            n = _parse_num(m)
            if not _is_likely_year(n):
                numbers.append(n)
        except ValueError:
            pass

    if not numbers:
        grounding = 0.0
    else:
        matched = 0
        denom = 0
        for n in numbers:
            if abs(n) < 1e-10:
                continue
            denom += 1
            best_score = max(
                (1.0 / (1.0 + abs(n - tv) / max(abs(tv), 1e-10))
                 for tv in table_values),
                default=0.0,
            )
            if best_score > 0.8:
                matched += 1
        grounding = matched / denom if denom else 0.0

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
                    correct += 1
                    continue
                if expected != 0:
                    if abs(stated - expected) / abs(expected) < 0.05:
                        correct += 1
                elif abs(stated) < 0.01:
                    correct += 1
            except Exception:
                correct += 1
        arithmetic = correct / len(calc_matches)
    else:
        arithmetic = 0.5

    return grounding, arithmetic


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


def get_table_entities(csv_path: str):
    """Extract entity tokens from CSV — column headers + first-column row labels.

    Used by anchored grounding: a numeric mention only counts when an entity
    token appears in its ±20-token window. Empty set → anchoring disabled
    (function returns 1.0 for all in-table values, matching legacy behavior).
    """
    try:
        df = pd.read_csv(csv_path)
        ents = set()
        for c in df.columns:
            if isinstance(c, str):
                w = re.split(r'[\s_/\-]+', c.strip().lower())
                ents.update(t for t in w if t and not t.isdigit() and len(t) > 1)
        if len(df.columns) > 0:
            for v in df.iloc[:, 0]:
                if isinstance(v, str):
                    w = re.split(r'[\s_/\-]+', v.strip().lower())
                    ents.update(t for t in w if t and not t.isdigit() and len(t) > 1)
        return ents
    except Exception:
        return set()


def _split_reasoning(response: str) -> str:
    """Pull out reasoning span from response (mirrors v1/v2 logic)."""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        return think_match.group(1)
    if '</think>' in response:
        return response[:response.index('</think>')]
    answer_match = re.search(r'<answer>', response)
    return response[:answer_match.start()] if answer_match else response


def compute_anchored_step_credit(response: str, csv_path: str):
    """V2 grounding: numeric mention counts only if (value ∈ CSV) AND
    (an entity token appears in ±20-token window around the number).

    Denominator = ALL non-zero numeric mentions (not only matched ones).
    Indiscriminate number quoting therefore inflates denominator faster than
    numerator → score self-dilutes. Returns (anchored_credit, arithmetic).

    No CSV → (0.0, 0.0) — caller must gate.
    No entities found → fall back to value-only matching (still
    denominator-aware), so anchoring never makes things worse than v2.
    """
    table_values = get_table_values(csv_path)
    if not table_values:
        return 0.0, 0.0

    entities = get_table_entities(csv_path)
    reasoning = _split_reasoning(response)
    if not reasoning.strip():
        return 0.0, 0.0

    # Tokenize reasoning once for window lookup
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_'\-]*|[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
                        reasoning)
    tok_lower = [t.lower() for t in tokens]

    # Identify numeric tokens with their positions
    num_positions = []
    for i, t in enumerate(tokens):
        try:
            n = _parse_num(t)
            if abs(n) < 1e-10 or _is_likely_year(n):
                continue
            num_positions.append((i, n))
        except (ValueError, AttributeError):
            pass

    if not num_positions:
        return 0.0, 0.0

    matched = 0
    for i, n in num_positions:
        in_table = any(
            (1.0 / (1.0 + abs(n - tv) / max(abs(tv), 1e-10))) > 0.9
            for tv in table_values
        )
        if not in_table:
            continue
        if entities:
            lo, hi = max(0, i - 20), min(len(tok_lower), i + 21)
            window = tok_lower[lo:hi]
            if any(t in entities for t in window):
                matched += 1
        else:
            matched += 1
    grounding = matched / len(num_positions)

    # Arithmetic: same recipe as v2
    calc_matches = re.findall(CALC_RE, reasoning)
    if calc_matches:
        correct, total = 0, 0
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
                    continue
                total += 1
                if expected != 0:
                    if abs(stated - expected) / abs(expected) < 0.05:
                        correct += 1
                elif abs(stated) < 0.01:
                    correct += 1
            except Exception:
                continue
        arithmetic = correct / total if total > 0 else 0.0
    else:
        arithmetic = 0.0
    return grounding, arithmetic


def compute_vapv_v2(response: str, csv_path: str, max_completion_length: int = 4096):
    """VAPV V2 process reward (rule-based, ~1ms).

    Returns (process_reward, debug_info_dict) — process ∈ [0, 1].

    Recipe:
      grounding = anchored_step_credit (denominator-aware, can't be gamed by
                  indiscriminate citation; entity ± window required)
      arithmetic = recomputed from CALC_RE matches
      raw = 0.6·grounding + 0.4·arithmetic
      length_factor = (1 - (think_len / max_completion_length) ** 2)  ≥ 0
      process = raw · length_factor

    No CSV → process = 0.0 (caller should gate to outcome-only).
    """
    grounding, arithmetic = compute_anchored_step_credit(response, csv_path)
    raw = 0.6 * grounding + 0.4 * arithmetic
    reasoning = _split_reasoning(response)
    think_len = len(reasoning.split())  # word-token approximation
    ratio = min(1.0, think_len / max(max_completion_length, 1))
    length_factor = max(0.0, 1.0 - ratio * ratio)
    process = raw * length_factor
    return process, {
        "grounding": grounding,
        "arithmetic": arithmetic,
        "raw_process": raw,
        "think_len_words": think_len,
        "length_factor": length_factor,
    }


def compute_process_reward_v2(response: str, csv_path: str) -> float:
    """Tightened rule-based process reward v2. ~1ms/completion.

    Changes from v1:
    - Grounding threshold: 0.8 → 0.9 (~10% tolerance)
    - Arithmetic default: 0.5 → 0.0 (no calcs = no bonus)
    - No CSV fallback: 0.5 → 0.0
    - Parse failures: no penalty → skip (no credit)
    """
    table_values = get_table_values(csv_path)
    if not table_values:
        return 0.0

    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1)
    elif '</think>' in response:
        reasoning = response[:response.index('</think>')]
    else:
        answer_match = re.search(r'<answer>', response)
        reasoning = response[:answer_match.start()] if answer_match else response

    if not reasoning.strip():
        return 0.0

    numbers = []
    for m in re.findall(NUM_RE, reasoning):
        try:
            n = _parse_num(m)
            if not _is_likely_year(n):
                numbers.append(n)
        except ValueError:
            pass

    if not numbers:
        return 0.0

    # (1) Value Grounding — tighter threshold
    non_zero = [n for n in numbers if abs(n) >= 1e-10]
    if not non_zero:
        grounding = 0.0
    else:
        matched = 0
        for n in non_zero:
            best_score = max(
                (1.0 / (1.0 + abs(n - tv) / max(abs(tv), 1e-10))
                 for tv in table_values),
                default=0.0
            )
            if best_score > 0.9:  # ~10% tolerance (was 0.8)
                matched += 1
        grounding = matched / len(non_zero)

    # (2) Arithmetic — strict default
    calc_matches = re.findall(CALC_RE, reasoning)
    if calc_matches:
        correct = 0
        total = 0
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
                    continue  # skip unverifiable

                total += 1
                if expected != 0:
                    if abs(stated - expected) / abs(expected) < 0.05:
                        correct += 1
                elif abs(stated) < 0.01:
                    correct += 1
            except Exception:
                continue  # skip parse failures
        arithmetic = correct / total if total > 0 else 0.0
    else:
        arithmetic = 0.0  # no calculations → no bonus (was 0.5)

    return 0.6 * grounding + 0.4 * arithmetic
