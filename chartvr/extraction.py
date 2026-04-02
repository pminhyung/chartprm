"""Answer extraction and accuracy metrics (v3).

Single canonical implementation — all other files should import from here.
"""
import re


def relaxed_accuracy(pred, gold):
    """ChartQA relaxed accuracy with unit stripping (v3).
    Returns 1.0 if match within 5% tolerance, 0.0 otherwise."""
    units_re = r'\s*(billion|million|thousand|trillion|percent|k|K|M|B|T|tons?|t|kg|lbs?|GiB|MB|GB)\b'
    p = re.sub(units_re, '', pred.strip(), flags=re.I).strip()
    g = re.sub(units_re, '', gold.strip(), flags=re.I).strip()
    p = re.sub(r'[,%$]', '', p).rstrip('.')
    g = re.sub(r'[,%$]', '', g).rstrip('.')
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower().strip() == g.lower().strip() else 0.0


def cerm_accuracy(pred: str, gold: str) -> float:
    """BigCharts-R1 style continuous accuracy. Returns float [0, 1].
    Smoother than binary relaxed_accuracy — used as GRPO reward."""
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        relative_change = abs(pf - gf) / abs(gf)
        return 1.0 / (1.0 + relative_change)
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


def _normalize(answer):
    """Normalize extracted answer: strip units, handle ranges, Yes/No."""
    answer = answer.replace("**", "").strip()
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()
    if len(answer) < 80 and answer.endswith('.'):
        answer = answer[:-1].strip()
    prefixes = [
        "Final answer:", "The answer is", "Answer:", "Therefore,",
        "So the answer is", "Thus,", "Hence,", "In conclusion,",
        "There are", "There is", "The value is", "The difference is",
        "It is", "The total is", "The average is",
    ]
    for p in prefixes:
        if answer.lower().startswith(p.lower()):
            answer = answer[len(p):].strip()
            break
    # Handle "X to Y" ranges — extract last value
    range_match = re.match(r'^(\d{4})\s+to\s+(\d{4})$', answer.strip())
    if range_match:
        return range_match.group(2)
    # Handle comma-separated lists — take first item
    if len(answer) > 10 and ',' in answer:
        first = answer.split(',')[0].strip()
        if first.lower() in ('yes', 'no'):
            return first
        if len(answer) > 30:
            return first
    ans_lower = answer.lower()
    if 'no' in ans_lower.split()[:3] or ans_lower.startswith('no'):
        return 'No'
    if 'yes' in ans_lower.split()[:3] or ans_lower.startswith('yes'):
        return 'Yes'
    # Strip units from answer tag content
    answer = re.sub(r'\s*(billion|million|thousand|trillion|percent)\b', '', answer, flags=re.I).strip()
    answer = re.sub(r'\s*[kKMBT]$', '', answer).strip()
    if len(answer) > 15:
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', answer)
        if nums:
            return nums[-1].replace(',', '')
    return answer


def extract_answer(response):
    """Extract answer from model response (v3).

    Priority: \\boxed{} → <answer>...</answer> → unclosed <answer> →
              post-</think> content → last line.
    """
    if not response:
        return ""
    m = re.search(r'\\boxed\{(.*?)\}', response)
    if m:
        return _normalize(m.group(1).strip())
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return _normalize(m.group(1).strip())
    # Handle unclosed <answer> tag (e.g. when stop=["</answer>"] truncates it)
    m = re.search(r'<answer>(.*?)$', response, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return _normalize(m.group(1).strip())
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            m2 = re.search(r'<answer>(.*?)</answer>', after, re.DOTALL | re.IGNORECASE)
            if m2:
                return _normalize(m2.group(1).strip())
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                return _normalize(lines[-1])
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return _normalize(lines[-1]) if lines else ""
