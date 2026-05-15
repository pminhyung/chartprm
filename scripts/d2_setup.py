"""D2 setup — pick 100 samples from segmented_v3.jsonl for the D2 pilot.

Guide §3.2: take the first 100 valid samples (n_steps > 0).
Anti-contamination: input is already pHash-clean training-side data.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN = BASE / "data/d1_pilot/segmented_v3.jsonl"
OUT = BASE / "data/d2_pilot/samples.jsonl"
N_PILOT = 100


def main():
    data = [json.loads(l) for l in open(IN) if l.strip()]
    valid = [d for d in data if d.get("n_steps", 0) > 0]
    print(f"[d2-setup] input={len(data)} valid={len(valid)} pick={N_PILOT}")
    if len(valid) < N_PILOT:
        print(f"[d2-setup] WARN: only {len(valid)} valid samples, taking all")
        picked = valid
    else:
        picked = valid[:N_PILOT]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for s in picked:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[d2-setup] wrote {len(picked)} -> {OUT}")

    # Quick stats
    from collections import Counter
    src = Counter(s["source"] for s in picked)
    print(f"[d2-setup] source mix: {dict(src)}")
    total_steps = sum(s["n_steps"] for s in picked)
    print(f"[d2-setup] total steps to score: {total_steps}  (K=8 continuations each = {total_steps*8} generations)")


if __name__ == "__main__":
    sys.exit(main() or 0)
