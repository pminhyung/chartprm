"""D1 automated validation — guide §2.5 spec.

6 sanity checks over segmented traces. No human annotation required.

Checks (target):
  1. length_median_in_range     : 150 ≤ median step tokens ≤ 500
  2. length_p90_under_cap       : p90 step tokens ≤ 700
  3. steps_per_sample_in_range  : 3 ≤ median steps/sample ≤ 12
  4. sentence_integrity_ok      : ≥75% segments end with terminal punctuation
  5. tiny_fragment_ok           : ≤10% segments < 30 tokens
  6. single_step_ok             : ≤15% samples with n_steps ≤ 1

Pass: 6/6  Conditional: 4-5/6  Fail: <4/6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
DEFAULT_IN = BASE / "data/d1_pilot/segmented_v2.jsonl"
REPORT_JSON = BASE / "data/d1_pilot/auto_validate.json"


def ends_with_terminal(text: str) -> bool:
    text = text.rstrip()
    return bool(re.search(r'[.!?]["\')\]]?$', text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN), help="segmented_v2.jsonl path")
    args = ap.parse_args()
    in_path = Path(args.input)

    data = [json.loads(l) for l in open(in_path) if l.strip()]
    valid = [r for r in data if r.get("n_steps", 0) > 0]
    print(f"[validate] input: {in_path}  total={len(data)}  valid={len(valid)}")

    if not valid:
        print("no valid records; aborting")
        return 1

    all_lens = [l for d in valid for l in d["step_token_lens"]]
    all_n_steps = [d["n_steps"] for d in data]  # include zero-step records in single-step calc
    median_len = float(np.median(all_lens))
    p90_len = float(np.percentile(all_lens, 90))
    median_steps = float(np.median([n for n in all_n_steps if n > 0]))

    print(f"\n[Length] median={median_len:.0f}  p90={p90_len:.0f}   (target: median 150-500, p90<700)")
    print(f"[Steps/sample] median={median_steps:.1f}   (target: 3-12)")

    terminals = [ends_with_terminal(step) for d in valid for step in d["steps"]]
    terminal_rate = sum(terminals) / len(terminals)
    print(f"[Sentence integrity] {terminal_rate*100:.1f}% segments end with .!?  (target ≥75%)")

    tiny = [l for l in all_lens if l < 30]
    tiny_rate = len(tiny) / len(all_lens)
    print(f"[Tiny fragments] {len(tiny)}/{len(all_lens)} (<30 tok) = {tiny_rate*100:.1f}%   (target ≤10%)")

    single_step_rate = sum(1 for n in all_n_steps if n <= 1) / len(all_n_steps)
    print(f"[Single-step samples] {single_step_rate*100:.1f}%   (target ≤15%)")

    checks = {
        "length_median_in_range": 150 <= median_len <= 500,
        "length_p90_under_cap": p90_len <= 700,
        "steps_per_sample_in_range": 3 <= median_steps <= 12,
        "sentence_integrity_ok": terminal_rate >= 0.75,
        "tiny_fragment_ok": tiny_rate <= 0.10,
        "single_step_ok": single_step_rate <= 0.15,
    }
    passed = sum(checks.values())
    print(f"\n[Auto sanity] {passed}/6 pass")
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")

    if passed == 6:
        verdict = "PASS"
    elif passed >= 4:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"
    print(f"\nVerdict: {verdict}  (PASS=6/6, CONDITIONAL=4-5/6, FAIL=<4/6)")

    # Per-source breakdown (informational)
    src_dist = Counter(r["source"] for r in valid)
    print(f"\nSource breakdown: {dict(src_dist)}")
    for src in src_dist:
        sub = [r for r in valid if r["source"] == src]
        ns = [r["n_steps"] for r in sub]
        ls = [l for r in sub for l in r["step_token_lens"]]
        print(f"  {src}: n_steps median={np.median(ns):.0f} max={max(ns)}  step_tokens median={np.median(ls):.0f} p90={np.percentile(ls,90):.0f}")

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({
        "input": str(in_path),
        "n_samples": len(data),
        "n_valid": len(valid),
        "length_median": median_len,
        "length_p90": p90_len,
        "steps_per_sample_median": median_steps,
        "sentence_integrity_rate": terminal_rate,
        "tiny_fragment_rate": tiny_rate,
        "single_step_rate": single_step_rate,
        "checks": checks,
        "passed": passed,
        "verdict": verdict,
    }, indent=2))
    print(f"\n[saved] {REPORT_JSON}")
    return 0 if verdict == "PASS" else (1 if verdict == "CONDITIONAL" else 2)


if __name__ == "__main__":
    sys.exit(main() or 0)
