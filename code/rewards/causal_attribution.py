"""
Rule-based causal error attribution.
NO LLM calls. Pure Python logic.

Phase 1 (LLM/rule): classify each sentence as correct/incorrect/not_verifiable
Phase 2 (this code): refine incorrect → source_error or propagated_error
"""
import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class AttributedSentence:
    text: str
    index: int
    numbers: List[float]
    raw_label: str          # From Phase 1: correct/incorrect/not_verifiable
    causal_label: str       # Refined: correct/source_error/propagated_error/not_verifiable
    logic_score: float      # 1.0 if logic correct, 0.0 if not
    input_quality: float    # 0.0~1.0 based on upstream number accuracy
    sentence_reward: float  # = logic_score × input_quality


def numbers_match(a: float, b: float, tol: float = 0.05) -> bool:
    """Check if two numbers are approximately equal."""
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / abs(b) < tol


def value_accuracy_score(model_value: float, table_value: float, sigma: float = 0.10) -> float:
    """
    Continuous accuracy score using Gaussian decay.
    sigma=0.10 means 10% relative error gives score ~0.61
    """
    if table_value == 0:
        return 1.0 if abs(model_value) < 0.01 else 0.0

    relative_error = abs(model_value - table_value) / abs(table_value)
    return math.exp(-0.5 * (relative_error / sigma) ** 2)


def find_closest_table_value(number: float, table_values: Set[float]) -> Tuple[float, float]:
    """Find the closest value in the table and return (closest_val, accuracy_score)."""
    if not table_values:
        return (0.0, 0.0)

    best_score = 0.0
    best_val = 0.0
    for tv in table_values:
        score = value_accuracy_score(number, tv)
        if score > best_score:
            best_score = score
            best_val = tv

    return (best_val, best_score)


def compute_causal_rewards(
    sentences: List[dict],
    table_values: Set[float],
    sigma: float = 0.10
) -> List[AttributedSentence]:
    """
    Main causal reward computation.

    Algorithm:
    1. Track which numbers are "tainted" (produced by incorrect sentences)
    2. For each incorrect sentence:
       - If all its input numbers are fresh (from chart): source_error
       - If any input number is tainted: propagated_error
    3. Compute:
       - logic_score: for source_error → 0.0; for propagated → check computation
       - input_quality: Gaussian score of input numbers vs table
       - sentence_reward: logic_score × input_quality
    """

    tainted_numbers: Set[float] = set()
    number_quality: Dict[float, float] = {}

    results = []

    for sent in sentences:
        numbers = sent["numbers"]
        raw_label = sent["raw_label"]

        if raw_label == "not_verifiable":
            results.append(AttributedSentence(
                text=sent["text"], index=sent["index"], numbers=numbers,
                raw_label="not_verifiable", causal_label="not_verifiable",
                logic_score=1.0, input_quality=1.0, sentence_reward=-1.0
            ))
            continue

        # Compute input_quality: how good are the numbers this sentence uses?
        if numbers:
            input_scores = []
            for n in numbers:
                n_rounded = round(n, 2)

                is_tainted = any(
                    numbers_match(n, tn, 0.05) for tn in tainted_numbers
                )

                if is_tainted:
                    matching_quality = min(
                        (number_quality.get(round(tn, 2), 0.0)
                         for tn in tainted_numbers
                         if numbers_match(n, tn, 0.05)),
                        default=0.0
                    )
                    input_scores.append(matching_quality)
                else:
                    _, score = find_closest_table_value(n, table_values)
                    input_scores.append(score)
                    number_quality[n_rounded] = score

            input_quality = min(input_scores)  # Conservative: weakest link
        else:
            input_quality = 1.0

        if raw_label == "correct":
            logic_score = 1.0
            causal = "correct"
        else:
            # raw_label == "incorrect"
            uses_tainted = any(
                any(numbers_match(n, tn, 0.05) for tn in tainted_numbers)
                for n in numbers
            )

            if uses_tainted:
                causal = "propagated_error"
                if sent.get("computation_correct") is True:
                    logic_score = 1.0
                elif sent.get("computation_correct") is False:
                    logic_score = 0.0
                else:
                    logic_score = 0.5
            else:
                causal = "source_error"
                logic_score = 0.0

            # Mark all numbers this sentence produces as tainted
            for n in numbers:
                tainted_numbers.add(round(n, 2))

        sentence_reward = logic_score * input_quality

        results.append(AttributedSentence(
            text=sent["text"], index=sent["index"], numbers=numbers,
            raw_label=raw_label, causal_label=causal,
            logic_score=logic_score, input_quality=input_quality,
            sentence_reward=sentence_reward
        ))

    return results
