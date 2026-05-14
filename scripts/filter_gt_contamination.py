"""Filter Qwen self-correction patterns out of an SFT jsonl.

Writes a cleaned copy and a side jsonl of the removed samples. Applies the
same pattern set as audit_gt_contamination.py.

Usage:
    python scripts/filter_gt_contamination.py INPUT.jsonl OUTPUT.jsonl [REMOVED.jsonl]
"""
import json
import re
import sys
from collections import Counter

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


def has_contamination(rec: dict) -> bool:
    parts: list[str] = []
    rs = rec.get("reasoning_steps", "")
    if isinstance(rs, list):
        parts.append(" ".join(str(s) for s in rs))
    elif isinstance(rs, str):
        parts.append(rs)
    at = rec.get("assistant_text", "")
    if isinstance(at, str):
        parts.append(at)
    text = "\n".join(parts)
    return any(p.search(text) for p in PATTERNS)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    removed_path = sys.argv[3] if len(sys.argv) >= 4 else out + ".removed"
    kept = 0
    removed = 0
    by_source_removed: Counter = Counter()
    with (
        open(inp) as f_in,
        open(out, "w") as f_out,
        open(removed_path, "w") as f_rm,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if has_contamination(rec):
                removed += 1
                by_source_removed[rec.get("source", "?")] += 1
                f_rm.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                kept += 1
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"input : {inp}")
    print(f"output: {out}  ({kept} kept)")
    print(f"removed: {removed_path}  ({removed} removed)")
    print("\nRemoved by source:")
    for src, n in by_source_removed.most_common():
        print(f"  {src:<20s} {n}")


if __name__ == "__main__":
    main()
