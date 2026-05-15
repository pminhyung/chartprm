"""D1 report — coherent rate + failure modes + decision."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
ANN_JSONL = BASE / "data/d1_pilot/annotations.jsonl"
SEG_JSONL = BASE / "data/d1_pilot/segmented.jsonl"
REPORT_MD = BASE / "data/d1_pilot/report.md"


def main():
    if not ANN_JSONL.exists():
        print(f"missing: {ANN_JSONL}")
        return 1
    anns = [json.loads(l) for l in open(ANN_JSONL) if l.strip()]
    if not anns:
        print("annotations.jsonl is empty")
        return 1

    total = sum(len(a["segments"]) for a in anns)
    coherent = sum(1 for a in anns for s in a["segments"] if s["coherent"] == "Y")
    rate = coherent / total if total else 0.0

    failures = Counter(s["failure"] for a in anns for s in a["segments"] if s["coherent"] == "N")
    types = Counter(s["step_type"] for a in anns for s in a["segments"])
    by_source_total = Counter()
    by_source_coh = Counter()
    for a in anns:
        for s in a["segments"]:
            by_source_total[a["source"]] += 1
            if s["coherent"] == "Y":
                by_source_coh[a["source"]] += 1
    by_source = {k: f"{by_source_coh[k]}/{v} = {by_source_coh[k]/v:.1%}" for k, v in by_source_total.items()}

    # Step distribution from segmented.jsonl (informational)
    segs = [json.loads(l) for l in open(SEG_JSONL) if l.strip()]
    valid = [r for r in segs if r["n_steps"] > 0]
    n_steps = [r["n_steps"] for r in valid]
    step_lens = [l for r in valid for l in r["step_token_lens"]]

    if rate >= 0.85:
        decision = "PASS"
        next_action = "Proceed to D2 (MC labeling + perception verifier pilot)."
    elif rate >= 0.75:
        decision = "CONDITIONAL"
        next_action = "Patch dominant failure mode then re-annotate 10 sample. See guide §2.6."
    else:
        decision = "FAIL"
        next_action = "Escalate to LLM-based segmenter (e.g., GPT-4o-mini) or revisit framing."

    import numpy as np
    med_steps = float(np.median(n_steps)) if n_steps else 0
    med_len = float(np.median(step_lens)) if step_lens else 0
    p10 = float(np.percentile(step_lens, 10)) if step_lens else 0
    p90 = float(np.percentile(step_lens, 90)) if step_lens else 0

    md = f"""# D1 Segmentation Pilot Report

## Summary

| Metric | Value |
|---|---|
| Annotated samples | {len(anns)} |
| Total segments | {total} |
| Coherent segments | {coherent} |
| **Coherent rate** | **{rate:.1%}** |
| **Decision** | **{decision}** |

## By source

| Source | Coherent / Total | Rate |
|---|---|---|
"""
    for k, v in by_source.items():
        md += f"| {k} | {v} | |\n"

    md += f"""
## Failure mode breakdown (when coherent=N)

| Mode | Count |
|---|---|
"""
    for k, v in failures.most_common():
        md += f"| {k} | {v} |\n"

    md += f"""
## Step type distribution (informational)

| Type | Count |
|---|---|
"""
    for k, v in types.most_common():
        md += f"| {k} | {v} |\n"

    md += f"""
## Step distribution (over {len(valid)} valid samples)

- steps/sample : median={med_steps:.1f}
- step tokens : median={med_len:.0f}  p10={p10:.0f}  p90={p90:.0f}

## Next action

{next_action}
"""

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(md)
    print(md)
    print(f"\n[saved] {REPORT_MD}")


if __name__ == "__main__":
    sys.exit(main() or 0)
