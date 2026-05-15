"""D2 4-cell distribution + Pattern A/B/C detector — main decision gate.

For each baseline, compute (drift × outcome) 2×2 distribution:
- grounded_correct: low drift  + correct outcome  (good)
- shortcut:        high drift + correct outcome  (lucky)
- careful_flawed:  low drift  + wrong outcome
- hallucinated:    high drift + wrong outcome

Patterns (guide §3 decision gate):
- A: all baselines shortcut ≥15% & grounded_correct diff <10pp  → Phase 2 GO
- A-strong: shortcut ≥15% & grounded_correct diff ≥10pp         → GO + sub-thesis
- B: distributions ~equal, shortcut <15%                        → MC-only pivot
- C: ChartGemma grounded_correct ≥ zero-shot +10pp              → reconsider
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartvr.extraction import extract_answer, relaxed_accuracy  # noqa: E402

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
DEFAULT_DRIFT_THRESHOLD = 0.5  # perception_agg >= threshold → low drift


def extract_final(content: str, reasoning: str = "") -> str:
    if content and content.strip():
        return extract_answer(content) or content.strip()
    return extract_answer(reasoning) or ""


def assign_cell(perception_agg: float | None, outcome: int) -> str:
    if perception_agg is None:
        return "unknown"
    low_drift = perception_agg >= DEFAULT_DRIFT_THRESHOLD
    if outcome == 1 and low_drift:
        return "grounded_correct"
    if outcome == 1 and not low_drift:
        return "shortcut"
    if outcome == 0 and low_drift:
        return "careful_flawed"
    return "hallucinated"


def analyze_baseline(name: str, perception_path: Path) -> dict:
    if not perception_path.exists():
        return {"error": f"missing {perception_path}"}
    samples = [json.loads(l) for l in open(perception_path) if l.strip()]
    cells = []
    valid = 0
    for s in samples:
        # Determine outcome via raw_response (baseline inference output) or content (orig trace)
        raw = s.get("raw_response", "") or s.get("content", "") or s.get("reasoning_content", "")
        final = extract_final(s.get("content", ""), s.get("reasoning_content", "")) \
                if s.get("content") or s.get("reasoning_content") else extract_answer(raw) or raw.strip()
        outcome = int(relaxed_accuracy(final, s["gold_answer"]))

        p_scores = [x["mean_score"] for x in s.get("step_perception_scores", [])
                    if x is not None and x.get("mean_score") is not None]
        perception_agg = float(np.mean(p_scores)) if p_scores else None
        if perception_agg is not None:
            valid += 1
        cells.append(assign_cell(perception_agg, outcome))
    n = len(cells)
    if n == 0:
        return {"error": "no samples"}
    breakdown = {
        "n_samples": n,
        "valid_perception": valid,
        "grounded_correct": cells.count("grounded_correct") / n * 100,
        "shortcut": cells.count("shortcut") / n * 100,
        "careful_flawed": cells.count("careful_flawed") / n * 100,
        "hallucinated": cells.count("hallucinated") / n * 100,
        "unknown": cells.count("unknown") / n * 100,
    }
    outc = cells.count("grounded_correct") + cells.count("shortcut")
    breakdown["outcome_correct_rate"] = outc / n * 100
    valid_known = max(n - cells.count("unknown"), 1)
    breakdown["drift_rate_within_correct"] = (
        cells.count("shortcut") / max(cells.count("grounded_correct") + cells.count("shortcut"), 1) * 100
    )
    return breakdown


def detect_pattern(results: dict) -> str:
    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) < 2:
        return "INSUFFICIENT_DATA"
    shortcut_rates = [v["shortcut"] for v in valid.values()]
    grounded_rates = [v["grounded_correct"] for v in valid.values()]
    universal_shortcut = all(s >= 15 for s in shortcut_rates)
    grounded_diff = max(grounded_rates) - min(grounded_rates)

    # Pattern C check: ChartGemma grounded > zero-shot 4b + 10pp?
    zs = valid.get("zeroshot_4b", {})
    cg = valid.get("chartgemma_12b", {})
    if zs and cg:
        if cg.get("grounded_correct", 0) - zs.get("grounded_correct", 0) >= 10:
            return "C  (ChartGemma grounded ≥ zero-shot +10pp → 차별화 약화)"

    if universal_shortcut and grounded_diff < 10:
        return "A  (drift universal, framing live → Phase 2 GO)"
    if universal_shortcut and grounded_diff >= 10:
        return "A-strong  (drift universal AND model-dependent → Phase 2 GO + sub-thesis)"
    if not universal_shortcut and grounded_diff < 10:
        return "B  (drift not universal → MC-only Math-Shepherd pivot)"
    return "AMBIGUOUS  (mixed evidence — need closer inspection)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", default=str(BASE / "data/d2_pilot/baseline_zeroshot_4b_perception.jsonl"))
    ap.add_argument("--r1", default=str(BASE / "data/d2_pilot/baseline_chart_r1_perception.jsonl"))
    ap.add_argument("--cg", default=str(BASE / "data/d2_pilot/baseline_chartgemma_perception.jsonl"))
    ap.add_argument("--output", default=str(BASE / "data/d2_pilot/4cell_distribution.json"))
    ap.add_argument("--drift_threshold", type=float, default=DEFAULT_DRIFT_THRESHOLD)
    args = ap.parse_args()
    globals()["DEFAULT_DRIFT_THRESHOLD"] = args.drift_threshold

    baselines = [
        ("zeroshot_4b", Path(args.zs)),
        ("chart_r1_7b", Path(args.r1)),
        ("chartgemma_12b", Path(args.cg)),
    ]
    results = {}
    for name, path in baselines:
        try:
            results[name] = analyze_baseline(name, path)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # Print table
    headers = ["Model", "N", "Grounded+Correct", "Shortcut", "CarefulFlawed", "Hallucinated", "Unknown", "Outcome%", "DriftWithinCorrect%"]
    print(f"\n{headers[0]:<18}{headers[1]:>5}{headers[2]:>20}{headers[3]:>12}{headers[4]:>16}{headers[5]:>15}{headers[6]:>10}{headers[7]:>12}{headers[8]:>22}")
    print("=" * 130)
    for name, b in results.items():
        if "error" in b:
            print(f"{name:<18}  ERROR: {b['error']}")
            continue
        print(f"{name:<18}{b['n_samples']:>5}{b['grounded_correct']:>19.1f}%{b['shortcut']:>11.1f}%"
              f"{b['careful_flawed']:>15.1f}%{b['hallucinated']:>14.1f}%{b['unknown']:>9.1f}%"
              f"{b['outcome_correct_rate']:>11.1f}%{b['drift_rate_within_correct']:>21.1f}%")
    pattern = detect_pattern(results)
    print(f"\n=== Pattern detected: {pattern} ===\n")

    out = {"drift_threshold": DEFAULT_DRIFT_THRESHOLD, "baselines": results, "pattern": pattern}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
