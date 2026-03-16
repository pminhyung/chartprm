"""
ChartVCR Reasoning Chain Analysis (v2 — audited).

Splits model reasoning into verifiable sentences,
extracts chart-relevant numbers, checks arithmetic,
and applies causal error attribution.

Fixes from audit:
1. Filter list indices ("1. Lamb" → 1 is not a chart value)
2. Filter years (1900-2099)
3. Filter ordinals ("first", "second")
4. Better sentence splitting (handle inline reasoning without \\n)
5. Strip <think>/<answer> tags
6. Skip sentences where ALL numbers are filtered
"""
import re
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field


# ═══════════════════════════════════════════
# Number extraction with context filtering
# ═══════════════════════════════════════════

# Raw number pattern — allows 1-6 digit integers + decimals + comma-separated
_NUM_PATTERN = re.compile(r'[-+]?\d{1,6}(?:,\d{3})*(?:\.\d+)?')

# Arithmetic pattern: "a op b = c"
_CALC_PATTERN = re.compile(
    r'([-+]?\d{1,6}(?:,\d{3})*(?:\.\d+)?)'
    r'\s*([+\-*/×÷])\s*'
    r'([-+]?\d{1,6}(?:,\d{3})*(?:\.\d+)?)'
    r'\s*[=≈]\s*'
    r'([-+]?\d{1,6}(?:,\d{3})*(?:\.\d+)?)'
)

# Patterns to EXCLUDE numbers from verification
_LIST_INDEX_RE = re.compile(r'(?:^|\n)\s*(\d{1,2})\.\s+[A-Z]')  # "1. Lamb", "2. Corn"
_PARENTHETICAL_INDEX_RE = re.compile(r'\((\d{1,2})\)')  # "(1)", "(2)"
_YEAR_RANGE = (1900, 2099)
_STEP_LABEL_RE = re.compile(r'(?:step|Step|STEP)\s+(\d+)')  # "Step 1", "Step 2"


def extract_chart_numbers(sentence: str) -> List[float]:
    """
    Extract numbers that are likely chart values, filtering out:
    - List indices ("1. Lamb")
    - Years (1900-2099)
    - Step labels ("Step 1")
    - Parenthetical indices ("(1)")
    - Counting words in context ("three countries")
    """
    # Find all number matches with positions
    raw_matches = list(_NUM_PATTERN.finditer(sentence))
    if not raw_matches:
        return []

    # Collect indices to exclude
    exclude_positions = set()

    # 1. List indices: "1. Lamb", "2. Corn"
    for m in _LIST_INDEX_RE.finditer(sentence):
        exclude_positions.add(m.start(1))

    # 2. Parenthetical indices: "(1)", "(2)"
    for m in _PARENTHETICAL_INDEX_RE.finditer(sentence):
        exclude_positions.add(m.start(1))

    # 3. Step labels: "Step 1"
    for m in _STEP_LABEL_RE.finditer(sentence):
        exclude_positions.add(m.start(1))

    # Filter numbers
    chart_nums = []
    for m in raw_matches:
        try:
            val = float(m.group().replace(',', ''))
        except ValueError:
            continue

        # Skip if at excluded position
        if m.start() in exclude_positions:
            continue

        # Skip years (4-digit integers 1900-2099)
        if _YEAR_RANGE[0] <= val <= _YEAR_RANGE[1] and val == int(val):
            # Keep only if it has explicit value context (%, $, "value", "total")
            context = sentence[max(0, m.start()-10):m.end()+10].lower()
            if not any(kw in context for kw in ['%', '$', 'value', 'total', 'average', 'sum']):
                continue

        # Skip list indices: "1. Lamb", "2. Corn" and similar
        after = sentence[m.end():m.end()+4]
        if val == int(val) and 1 <= val <= 50 and re.match(r'\.\s+[A-Z]', after):
            continue

        # Skip parenthetical numbering: "(1)", "(2)"
        before_char = sentence[m.start()-1:m.start()] if m.start() > 0 else ''
        after_char = sentence[m.end():m.end()+1]
        if before_char == '(' and after_char == ')':
            continue

        # Skip counting words: "three countries" → if sentence has "is X" or "are X" for small ints
        if val == int(val) and 1 <= val <= 10:
            before_words = sentence[max(0, m.start()-15):m.start()].lower()
            after_words = sentence[m.end():m.end()+20].lower()
            # "there are 3 countries" — 3 is a count, not chart value
            if any(w in before_words for w in ['are ', 'is ', 'has ', 'have ', 'only ']):
                if any(w in after_words for w in [' item', ' bar', ' color', ' countr', ' categor', ' segment', ' line', ' group']):
                    continue

        chart_nums.append(val)

    return chart_nums


# ═══════════════════════════════════════════
# Sentence splitting
# ═══════════════════════════════════════════

def split_reasoning(response: str) -> List[str]:
    """
    Split reasoning response into verifiable sentences.
    Handles: <think> tags, newlines, periods.
    Removes: tags, empty lines, very short fragments.
    """
    # Extract thinking content if present
    think_m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_m:
        text = think_m.group(1).strip()
    else:
        # Remove answer tags and use full response as reasoning
        text = re.sub(r'</?(?:think|answer|/think|/answer)>', '', response).strip()
        # If there's an <answer> tag, take everything before it
        ans_pos = text.find('<answer>')
        if ans_pos > 0:
            text = text[:ans_pos].strip()

    # Split on newlines first
    lines = text.split('\n')

    # For very long single lines, also split on ". " (sentence boundary)
    sentences = []
    for line in lines:
        line = line.strip()
        if not line or len(line) <= 5:
            continue
        # Skip tag-only lines
        if line in ('</think>', '<think>', '</answer>', '<answer>'):
            continue
        # If line is very long (>200 chars), split on sentence boundaries
        if len(line) > 200:
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
            for p in parts:
                p = p.strip()
                if len(p) > 5:
                    sentences.append(p)
        else:
            sentences.append(line)

    return sentences


# ═══════════════════════════════════════════
# Gaussian scoring
# ═══════════════════════════════════════════

def gaussian_score(model_val: float, table_val: float, sigma: float = 0.10) -> float:
    """Continuous accuracy: Gaussian decay based on relative error."""
    if table_val == 0:
        return 1.0 if abs(model_val) < 0.01 else 0.0
    rel_err = abs(model_val - table_val) / abs(table_val)
    return math.exp(-0.5 * (rel_err / sigma) ** 2)


def best_table_match(val: float, table_vals: Set[float], sigma: float = 0.10) -> Tuple[float, Optional[float]]:
    """Find best matching table value. Returns (score, matched_value)."""
    if not table_vals:
        return 0.0, None
    best_score = 0.0
    best_match = None
    for tv in table_vals:
        s = gaussian_score(val, tv, sigma)
        if s > best_score:
            best_score = s
            best_match = tv
    return best_score, best_match


# ═══════════════════════════════════════════
# Sentence-level analysis
# ═══════════════════════════════════════════

@dataclass
class SentenceAnalysis:
    text: str
    index: int
    raw_numbers: List[float]          # All numbers in sentence
    chart_numbers: List[float]        # Numbers after filtering
    filtered_numbers: List[float]     # Numbers that were filtered out
    input_quality: float = 1.0
    logic_score: float = 1.0
    sentence_reward: float = 0.0
    label: str = "skip"               # skip | correct | source_error | propagated_error
    arithmetic: Optional[str] = None  # Description of arithmetic check
    number_details: List[str] = field(default_factory=list)


def analyze_sentence(
    sent: str,
    index: int,
    table_vals: Set[float],
    tainted: Set[float],
    sigma: float = 0.10,
) -> SentenceAnalysis:
    """Analyze a single sentence for chart value accuracy and arithmetic."""
    raw_nums = [float(m.replace(',', '')) for m in _NUM_PATTERN.findall(sent)]
    chart_nums = extract_chart_numbers(sent)
    filtered = [n for n in raw_nums if n not in chart_nums]

    result = SentenceAnalysis(
        text=sent,
        index=index,
        raw_numbers=raw_nums,
        chart_numbers=chart_nums,
        filtered_numbers=filtered,
    )

    # Skip if no chart-relevant numbers
    if not chart_nums:
        result.label = "skip"
        return result

    # Input quality: how well do chart numbers match the table?
    iq_list = []
    uses_tainted = False
    for n in chart_nums:
        # Check taint
        if any(abs(n - t) / max(abs(t), 1e-10) < 0.05 for t in tainted):
            uses_tainted = True
            iq_list.append(0.3)
            result.number_details.append(f"{n} → TAINTED (from prior error)")
        else:
            score, matched = best_table_match(n, table_vals, sigma)
            iq_list.append(score)
            if score > 0.7:
                result.number_details.append(f"{n} ✅ matches {matched} (score={score:.3f})")
            elif score > 0.3:
                result.number_details.append(f"{n} ⚠️ partial match {matched} (score={score:.3f})")
            else:
                result.number_details.append(f"{n} ❌ no match (best={matched}, score={score:.3f})")

    # Use mean instead of min: one bad number shouldn't zero out the whole sentence
    # But still penalize heavily if majority are bad
    if iq_list:
        mean_iq = sum(iq_list) / len(iq_list)
        min_iq = min(iq_list)
        # Blend: 70% mean + 30% min (penalizes but doesn't zero out)
        result.input_quality = 0.7 * mean_iq + 0.3 * min_iq
    else:
        result.input_quality = 1.0

    # Logic score: check arithmetic
    cm = _CALC_PATTERN.search(sent)
    if cm:
        try:
            a = float(cm.group(1).replace(',', ''))
            op = cm.group(2).replace('×', '*').replace('÷', '/')
            b = float(cm.group(3).replace(',', ''))
            stated = float(cm.group(4).replace(',', ''))
            if op in '+-*/' and (op != '/' or b != 0):
                expected = eval(f"{a}{op}{b}")
                if abs(stated - expected) / max(abs(expected), 1e-10) < 0.05:
                    result.logic_score = 1.0
                    result.arithmetic = f"✅ {a}{op}{b} = {stated} (expected {expected:.4f})"
                else:
                    result.logic_score = 0.0
                    result.arithmetic = f"❌ {a}{op}{b} = {stated} (expected {expected:.4f})"
        except Exception:
            result.logic_score = 1.0
            result.arithmetic = "⚠️ parse error"
    else:
        result.arithmetic = None  # No arithmetic to check

    # Sentence reward
    result.sentence_reward = result.logic_score * result.input_quality

    # Label with causal attribution
    # Use 0.3 threshold (not 0.5) — less aggressive taint triggering
    if result.sentence_reward >= 0.3:
        result.label = "correct"
    elif uses_tainted:
        result.label = "propagated_error"
    else:
        result.label = "source_error"

    return result


# ═══════════════════════════════════════════
# Full process reward computation
# ═══════════════════════════════════════════

def compute_process_reward(
    response: str,
    csv_path: str,
    sigma: float = 0.10,
) -> Tuple[float, List[SentenceAnalysis]]:
    """
    Compute ChartVCR process reward with full analysis.
    Returns (reward_score, list_of_sentence_analyses).
    """
    import pandas as pd

    # Load table values
    table_vals = set()
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            for col in df.columns:
                for v in df[col]:
                    try:
                        table_vals.add(float(v))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    if not table_vals:
        return 0.0, []

    # Split reasoning into sentences
    sentences = split_reasoning(response)
    if not sentences:
        return 0.0, []

    # Analyze each sentence
    tainted: Set[float] = set()
    analyses = []

    for i, sent in enumerate(sentences):
        analysis = analyze_sentence(sent, i, table_vals, tainted, sigma)
        analyses.append(analysis)

        # Taint numbers from source errors (not propagated)
        if analysis.label == "source_error":
            for n in analysis.chart_numbers:
                tainted.add(n)

    # Aggregate: average of verified sentences only
    verified = [a.sentence_reward for a in analyses if a.label != "skip"]
    reward = sum(verified) / len(verified) if verified else 0.0

    return reward, analyses
