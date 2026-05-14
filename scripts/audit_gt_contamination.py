"""Scan SFT jsonl files for Qwen thinking-mode self-correction contamination.

Counts per-source how many samples have "Wait, let", "Correction during",
etc. patterns inside reasoning_steps / assistant_text / answer fields.

Usage:
    python scripts/audit_gt_contamination.py data/sft_hq_lite.jsonl [more.jsonl ...]
"""
import json
import re
import sys
from collections import Counter, defaultdict

PATTERNS = [
    re.compile(r"wait,\s*let", re.IGNORECASE),
    re.compile(r"wait,\s*re-?read", re.IGNORECASE),
    re.compile(r"wait[,.]\s+re-?verif", re.IGNORECASE),
    re.compile(r"correction\s*(during|:)", re.IGNORECASE),
    re.compile(r"let me re-?read", re.IGNORECASE),
    re.compile(r"let me re-?verify", re.IGNORECASE),
    re.compile(r"actually[,.]?\s+let", re.IGNORECASE),
    re.compile(r"hmm[,.]", re.IGNORECASE),
]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks — Wait patterns inside are normal."""
    return _THINK_RE.sub("", text)


def extract_text(rec: dict) -> str:
    """Concatenate every field where Wait contamination might live.

    <think> blocks are stripped: self-correction inside thinking is expected
    for teacher-distilled data and should not count as contamination.
    """
    parts: list[str] = []
    # reasoning_steps (list or str)
    rs = rec.get("reasoning_steps", "")
    if isinstance(rs, list):
        parts.append(" ".join(str(s) for s in rs))
    elif isinstance(rs, str):
        parts.append(rs)
    # assistant_text (teacher-distilled)
    at = rec.get("assistant_text", "")
    if isinstance(at, str):
        parts.append(at)
    # question + answer as a safety net
    parts.append(str(rec.get("question", "")))
    parts.append(str(rec.get("answer", "")))
    return _strip_think("\n".join(parts))


def count_hits(text: str) -> int:
    return sum(1 for p in PATTERNS if p.search(text))


def audit(path: str) -> dict:
    by_source: dict[str, Counter] = defaultdict(Counter)
    total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            src = rec.get("source", "?")
            text = extract_text(rec)
            hits = count_hits(text)
            by_source[src]["total"] += 1
            if hits > 0:
                by_source[src]["contaminated"] += 1
    return {"total": total, "by_source": by_source}


def print_report(path: str, report: dict) -> None:
    print(f"\n=== {path} ===")
    print(f"Total records: {report['total']}")
    print(f"{'source':<20s} {'total':>7s} {'bad':>6s} {'%':>6s}")
    print("-" * 42)
    grand_total = 0
    grand_bad = 0
    for src in sorted(report["by_source"].keys()):
        c = report["by_source"][src]
        t = c["total"]
        b = c.get("contaminated", 0)
        pct = b * 100 / t if t else 0
        grand_total += t
        grand_bad += b
        print(f"{src:<20s} {t:>7d} {b:>6d} {pct:>5.1f}%")
    print("-" * 42)
    gpct = grand_bad * 100 / grand_total if grand_total else 0
    print(f"{'TOTAL':<20s} {grand_total:>7d} {grand_bad:>6d} {gpct:>5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        print_report(p, audit(p))
