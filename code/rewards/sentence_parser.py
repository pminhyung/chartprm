"""
Core utility: parse CoT into sentences and extract numbers.
Used by both the pilot study AND the actual reward function.
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParsedSentence:
    text: str
    index: int
    numbers: List[float]
    has_computation: bool
    computation_result: Optional[Tuple[float, float, str, float]]
    # (operand1, operand2, operator, stated_result)


def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return numbers


def extract_computation(text: str) -> Optional[Tuple[float, float, str, float]]:
    """
    Extract explicit arithmetic: "45.3 - 38.1 = 7.2"
    Returns (a, b, op, result) or None.
    """
    pattern = (
        r'([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        r'\s*([+\-*/×÷])\s*'
        r'([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
        r'\s*[=≈]\s*'
        r'([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
    )
    match = re.search(pattern, text)
    if match:
        try:
            a = float(match.group(1).replace(",", ""))
            op = match.group(2)
            if op == '×':
                op = '*'
            if op == '÷':
                op = '/'
            b = float(match.group(3).replace(",", ""))
            result = float(match.group(4).replace(",", ""))
            return (a, b, op, result)
        except ValueError:
            return None
    return None


def parse_cot_to_sentences(response: str) -> List[ParsedSentence]:
    """
    Split CoT into sentences and annotate each.

    Splitting strategy:
    - Primary: extract <think> content if present
    - Split on "\\n\\n" (BigCharts-R1 step delimiter)
    - Fallback: split on "\\n" if too few segments
    """
    # Extract think content if present
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        content = think_match.group(1).strip()
    else:
        content = response.strip()

    # Split into sentences
    sentences = content.split("\n\n")
    if len(sentences) < 3:
        sentences = content.split("\n")

    # Filter empty/trivial sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    parsed = []
    for i, sent in enumerate(sentences):
        numbers = extract_numbers(sent)
        computation = extract_computation(sent)
        parsed.append(ParsedSentence(
            text=sent,
            index=i,
            numbers=numbers,
            has_computation=computation is not None,
            computation_result=computation
        ))

    return parsed
