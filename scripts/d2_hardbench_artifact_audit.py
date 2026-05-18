"""Hard-bench artifact audit — eval/judge noise detection.

Checks:
1. Inference errors per (model, bench): how many samples errored out?
2. Outcome scoring coverage: how many parsed vs judge NULL?
3. Judge format consistency: 'Verdict: Correct/Incorrect' / 'Yes/No' parseable?
4. Pred-Gold format mismatch (chartqa_pro): pred number ≈ gold number 케이스 검출
5. Empty content/reasoning: how many samples had no usable trace?
6. Perception NOT_FOUND rate per cell
7. Sample 3 careful_flawed examples per (model, bench) for manual eyeballing
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
HB = BASE / "data/d2_hardbench"

MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]


def numbers_in(s):
    return re.findall(r"-?\d+(?:\.\d+)?", str(s))


def audit_per_combo(model: str, bench: str) -> dict:
    inf_p = HB / f"inference/{model}/{bench}.jsonl"
    seg_p = HB / f"segmented/{model}/{bench}.jsonl"
    perc_p = HB / f"perception/{model}/{bench}.jsonl"
    outc_p = HB / f"outcome/outcome_{model}_{bench}.jsonl"
    if not all(p.exists() for p in [inf_p, seg_p, perc_p, outc_p]):
        return {"error": "missing files"}

    inf_recs = [json.loads(l) for l in open(inf_p) if l.strip()]
    seg_recs = [json.loads(l) for l in open(seg_p) if l.strip()]
    perc_recs = [json.loads(l) for l in open(perc_p) if l.strip()]
    outc_recs = [json.loads(l) for l in open(outc_p) if l.strip()]

    n = len(inf_recs)
    # 1. Inference errors
    inf_errors = sum(1 for r in inf_recs if r.get("error"))
    inf_empty = sum(1 for r in inf_recs if not (r.get("content","").strip() or r.get("reasoning_content","").strip()))

    # 5. Empty steps in segmentation
    empty_steps = sum(1 for r in seg_recs if r.get("n_steps", 0) == 0)

    # 6. Perception NOT_FOUND rate
    perc_nf = []
    for r in perc_recs:
        rates = [s.get("not_found_rate") for s in r.get("step_perception_scores", []) if s is not None]
        rates = [x for x in rates if x is not None]
        if rates:
            perc_nf.append(sum(rates)/len(rates))
    mean_nf = sum(perc_nf)/max(len(perc_nf), 1)

    # 2. Outcome coverage
    judge_null = sum(1 for r in outc_recs if r.get("outcome") is None)
    judge_pass = sum(1 for r in outc_recs if r.get("outcome") == 1)
    judge_fail = sum(1 for r in outc_recs if r.get("outcome") == 0)
    judge_parse_fail = judge_null

    # 3. Judge raw format check (sample 5 raws)
    judge_raws_sample = [r.get("judge_raw","")[:100] for r in outc_recs[:5] if "judge_raw" in r]

    # 4. Pred-Gold format mismatch (chartqa_pro)
    fmt_suspicious = 0
    if bench == "chartqa_pro":
        for r in outc_recs:
            if r.get("outcome") == 0:  # outcome 0 cases
                pred_nums = numbers_in(r.get("prediction",""))
                gold_nums = numbers_in(r.get("gold",""))
                if pred_nums and gold_nums:
                    for pn in pred_nums:
                        for gn in gold_nums:
                            try:
                                if abs(float(pn)-float(gn))/max(abs(float(gn)),1e-10) <= 0.05:
                                    fmt_suspicious += 1
                                    break
                            except: pass
                        else: continue
                        break

    # 7. 3 careful_flawed examples (perception >=0.5, outcome=0)
    outc_by_id = {r["id"]: r for r in outc_recs}
    cf_examples = []
    import numpy as np
    for r in perc_recs:
        p_scores = [s["mean_score"] for s in r.get("step_perception_scores", [])
                    if s is not None and s.get("mean_score") is not None]
        if not p_scores: continue
        p_agg = float(np.mean(p_scores))
        out_r = outc_by_id.get(r["id"])
        if out_r and out_r.get("outcome") == 0 and p_agg >= 0.7:
            cf_examples.append({
                "id": r["id"],
                "p_agg": round(p_agg, 3),
                "question": r["question"][:200],
                "gold": out_r["gold"][:200],
                "pred": out_r["prediction"][:300],
                "judge_raw": out_r.get("judge_raw","")[:200],
            })
        if len(cf_examples) >= 3: break

    return {
        "n": n,
        "inference_errors": inf_errors,
        "inference_empty_traces": inf_empty,
        "segmentation_empty": empty_steps,
        "perception_mean_NOT_FOUND_rate": round(mean_nf*100, 1),
        "judge_pass": judge_pass,
        "judge_fail": judge_fail,
        "judge_null": judge_null,
        "judge_raws_sample": judge_raws_sample,
        "chartqa_pro_fmt_suspicious_false_neg": fmt_suspicious,
        "careful_flawed_examples": cf_examples,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(HB / "reports/artifact_audit.json"))
    args = ap.parse_args()

    results = {}
    print(f"\n{'='*80}\nARTIFACT AUDIT (per model × bench)\n{'='*80}\n")
    for model in MODELS:
        for bench in BENCHES:
            key = f"{model}__{bench}"
            r = audit_per_combo(model, bench)
            results[key] = r
            if "error" in r:
                print(f"{key:<50} ERROR: {r['error']}")
                continue
            print(f"\n--- {key} ---")
            print(f"  N={r['n']}")
            print(f"  inference errors: {r['inference_errors']}")
            print(f"  inference empty traces: {r['inference_empty_traces']}")
            print(f"  segmentation empty (no steps): {r['segmentation_empty']}")
            print(f"  perception NOT_FOUND rate: {r['perception_mean_NOT_FOUND_rate']}%")
            print(f"  outcome judge: pass={r['judge_pass']}  fail={r['judge_fail']}  null/parse_fail={r['judge_null']}")
            if r.get("chartqa_pro_fmt_suspicious_false_neg", 0) > 0:
                print(f"  chartqa_pro fmt_suspicious_false_neg: {r['chartqa_pro_fmt_suspicious_false_neg']}/{r['judge_fail']} fails")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n-> {out_path}\n")


if __name__ == "__main__":
    sys.exit(main() or 0)
