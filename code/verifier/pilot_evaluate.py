"""
Phase 1.4: Evaluate verifier accuracy by comparing verifier labels with
rule-based ground truth (computation checks + table lookups).

Since human labeling is expensive, we use rule-based GT as a proxy:
- For computation sentences: Python eval determines correctness
- For extraction sentences: CSV lookup determines correctness

CHECKPOINT CONDITIONS:
- Overall accuracy >= 80%: PROCEED with LLM verifier
- Accuracy 70-80%: Try improved prompt, retest
- Accuracy < 70%: FALLBACK to rule-based only
"""
import json
import os
import sys
import re
import pandas as pd

sys.path.insert(0, os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

from code.rewards.sentence_parser import extract_numbers, extract_computation
from code.rewards.causal_attribution import find_closest_table_value

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")


def load_table_values(csv_path):
    """Extract all numeric values from a CSV."""
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


def compute_rule_based_label(sentence_data):
    """
    Compute rule-based ground truth label for a sentence.
    Returns: 'correct', 'incorrect', or 'not_verifiable'
    """
    sentence = sentence_data["sentence"]
    numbers = sentence_data["numbers"]
    label_type = sentence_data["label_type"]
    csv_path = sentence_data["csv_path"]

    if label_type == "computation":
        # Check arithmetic correctness
        comp = extract_computation(sentence)
        if comp is None:
            return "not_verifiable"

        a, b, op, stated_result = comp
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
                return "not_verifiable"

            rel_error = abs(stated_result - expected) / max(abs(expected), 1e-10)
            if rel_error < 0.05:
                return "correct"
            else:
                return "incorrect"
        except Exception:
            return "not_verifiable"

    elif label_type == "extraction":
        # Check if extracted numbers match the table
        table_values = load_table_values(csv_path)
        if not table_values:
            return "not_verifiable"

        # Check each number against table
        for n in numbers:
            _, score = find_closest_table_value(n, table_values)
            if score > 0.7:
                return "correct"

        # No number matched the table well
        # But it could be a computed value, not an extraction
        return "not_verifiable"  # Conservative: don't label as incorrect

    return "not_verifiable"


def evaluate_pilot():
    """Compare verifier labels with rule-based ground truth."""
    verifier_path = os.path.join(BASE, "results/pilot/verifier_labels.json")

    with open(verifier_path) as f:
        verifier_data = json.load(f)

    # Compute rule-based labels
    y_true = []
    y_pred = []
    details = []

    for s in verifier_data:
        if s["verifier_label"] == "skip":
            continue

        rule_label = compute_rule_based_label(s)
        if rule_label == "not_verifiable":
            continue  # Can't evaluate verifier on ambiguous cases

        y_true.append(rule_label)
        y_pred.append(s["verifier_label"])
        details.append({
            "id": f"{s['trace_idx']}_{s['sentence_idx']}",
            "sentence": s["sentence"][:100],
            "rule_label": rule_label,
            "verifier_label": s["verifier_label"],
            "match": rule_label == s["verifier_label"],
        })

    if not y_true:
        print("ERROR: No evaluable sentences found")
        return "rule_based_only"

    # Overall accuracy
    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Binary: error detection (correct vs incorrect)
    binary_true = [1 if t == "incorrect" else 0 for t in y_true]
    binary_pred = [1 if p == "incorrect" else 0 for p in y_pred]
    binary_acc = sum(1 for t, p in zip(binary_true, binary_pred) if t == p) / max(len(binary_true), 1)

    print(f"\n{'=' * 50}")
    print(f"PILOT STUDY RESULTS")
    print(f"{'=' * 50}")
    print(f"Total compared: {len(y_true)}")
    print(f"3-class accuracy: {overall_acc:.1%}")
    print(f"Binary error detection accuracy: {binary_acc:.1%}")

    # Breakdown
    from collections import Counter
    print(f"\nRule-based labels: {Counter(y_true)}")
    print(f"Verifier labels:  {Counter(y_pred)}")

    # Confusion matrix
    print(f"\nMismatches:")
    mismatches = [d for d in details if not d["match"]]
    for m in mismatches[:10]:
        print(f"  [{m['id']}] rule={m['rule_label']}, verifier={m['verifier_label']}: {m['sentence']}")

    # DECISION
    if overall_acc >= 0.80:
        print(f"\n✅ DECISION: PROCEED with LLM verifier (accuracy={overall_acc:.1%})")
        decision = "llm_verifier"
    elif overall_acc >= 0.70:
        print(f"\n⚠️ DECISION: Try improved prompt, then retest (accuracy={overall_acc:.1%})")
        decision = "retry_prompt"
    else:
        print(f"\n❌ DECISION: FALLBACK to rule-based only (accuracy={overall_acc:.1%})")
        decision = "rule_based_only"

    # Save decision
    decision_path = os.path.join(BASE, "results/pilot/decision.json")
    with open(decision_path, "w") as f:
        json.dump({
            "overall_accuracy": overall_acc,
            "binary_accuracy": binary_acc,
            "decision": decision,
            "n_samples": len(y_true),
            "n_correct_rule": sum(1 for t in y_true if t == "correct"),
            "n_incorrect_rule": sum(1 for t in y_true if t == "incorrect"),
            "details": details,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDecision saved to: {decision_path}")
    return decision


if __name__ == "__main__":
    evaluate_pilot()
