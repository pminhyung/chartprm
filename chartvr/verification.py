"""QA arithmetic verification — single canonical implementation."""
import re

# Unified calculation regex
NUM_RE = r'[-+]?\d+(?:,?\d{3})*(?:\.\d+)?'
CALC_RE = re.compile(
    rf'({NUM_RE})\s*([+\-*/×÷])\s*({NUM_RE})\s*[=≈]\s*({NUM_RE})'
)


def _parse_num(s: str) -> float:
    return float(s.replace(',', ''))


def verify_qa(qa: dict) -> bool:
    """Verify QA by re-checking arithmetic in reasoning steps.
    Returns True if all computations are correct and answer matches."""
    steps = qa.get("reasoning_steps", qa.get("steps", []))
    if not steps:
        return False

    all_correct = True
    last_computed = None
    comps_found = 0

    for step in steps:
        for m in CALC_RE.finditer(step):
            try:
                a = _parse_num(m.group(1))
                op = m.group(2).replace('×', '*').replace('÷', '/')
                b = _parse_num(m.group(3))
                stated = _parse_num(m.group(4))

                if op == '+': actual = a + b
                elif op == '-': actual = a - b
                elif op == '*': actual = a * b
                elif op == '/' and b != 0: actual = a / b
                else: continue

                comps_found += 1
                last_computed = stated

                if actual != 0 and abs(stated - actual) / abs(actual) > 0.05:
                    all_correct = False
                elif actual == 0 and abs(stated) >= 0.01:
                    all_correct = False
            except Exception:
                continue

    # Check answer is numeric and within range
    try:
        av = float(str(qa["answer"]).replace(",", "").replace("%", ""))
    except (ValueError, TypeError):
        return False

    if abs(av) > 10000 or (abs(av) < 0.001 and av != 0) or av == 0.0:
        return False

    # Check final answer matches last computation
    if last_computed is not None:
        if av != 0 and abs(last_computed - av) / max(abs(av), 1e-10) > 0.05:
            return False
        elif av == 0 and abs(last_computed) >= 0.01:
            return False

    return comps_found > 0 and all_correct


def verify_answer(qa: dict, csv_path: str) -> dict:
    """Verify QA answer by re-extracting values from CSV and recomputing.
    Returns qa with 'verified' and 'verify_reason' fields."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        qa["verified"] = False
        qa["verify_reason"] = "csv_parse_error"
        return qa

    steps = qa.get("reasoning_steps", [])
    if not steps:
        qa["verified"] = False
        qa["verify_reason"] = "no_steps"
        return qa

    all_correct = True
    comps_found = 0
    last_computed = None

    for step in steps:
        for m in CALC_RE.finditer(step):
            try:
                a = _parse_num(m.group(1))
                op = m.group(2).replace('×', '*').replace('÷', '/')
                b = _parse_num(m.group(3))
                stated = _parse_num(m.group(4))

                if op == '+': actual = a + b
                elif op == '-': actual = a - b
                elif op == '*': actual = a * b
                elif op == '/' and b != 0: actual = a / b
                else: continue

                comps_found += 1
                last_computed = stated

                if actual == 0:
                    if abs(stated) >= 0.01:
                        all_correct = False
                else:
                    if abs(stated - actual) / abs(actual) > 0.05:
                        all_correct = False
            except (ValueError, ZeroDivisionError):
                continue

    try:
        answer_val = float(str(qa["answer"]).replace(',', '').replace('%', ''))
    except (ValueError, TypeError):
        qa["verified"] = False
        qa["verify_reason"] = "non_numeric_answer"
        return qa

    if last_computed is not None:
        if answer_val == 0:
            if abs(last_computed) >= 0.01:
                qa["verified"] = False
                qa["verify_reason"] = "final_answer_vs_computation_mismatch"
                return qa
        else:
            if abs(last_computed - answer_val) / max(abs(answer_val), 1e-10) > 0.05:
                qa["verified"] = False
                qa["verify_reason"] = "final_answer_vs_computation_mismatch"
                return qa

    if comps_found == 0:
        qa["verified"] = False
        qa["verify_reason"] = "no_computations_in_steps"
    elif all_correct:
        qa["verified"] = True
        qa["verify_reason"] = "computations_verified"
    else:
        qa["verified"] = False
        qa["verify_reason"] = "computation_mismatch"

    return qa
