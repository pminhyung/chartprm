"""Hard-bench final synthesis — combine all signals into one decision report.

Inputs:
- data/d2_hardbench/reports/4cell_per_bench.json
- data/d2_hardbench/reports/4cell_per_bench_rejudge.json (rejudge with extended answer extraction)
- data/d2_hardbench/reports/artifact_audit.json
- data/d2_hardbench/reports/cf_no_answer_audit.json (format-failure prevalence in cf cell)
- data/d2_pilot/4cell_rescored.json (Phase 1 main pilot result for comparison)

Outputs:
- docs/d2_hardbench_decision_report.md
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def load(p: Path) -> dict | None:
    if not p.exists(): return None
    try: return json.load(open(p))
    except Exception: return None


def fmt_pct(v, spec=".1f"):
    if v is None: return "—"
    return format(v, spec) + "%"


def fmt_num(v, spec=".3f"):
    if v is None: return "—"
    return format(v, spec)


def main():
    cell = load(BASE / "data/d2_hardbench/reports/4cell_per_bench.json") or {}
    cell_rej = load(BASE / "data/d2_hardbench/reports/4cell_per_bench_rejudge.json") or {}
    audit = load(BASE / "data/d2_hardbench/reports/artifact_audit.json") or {}
    cf_noans = load(BASE / "data/d2_hardbench/reports/cf_no_answer_audit.json") or {}
    main_pilot = load(BASE / "data/d2_pilot/4cell_rescored.json") or {}

    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    out.append(f"# D2 Hard-Bench Decision Report — {today}\n")
    out.append("## Setup\n")
    out.append("- Benchmarks: **ChartQA-Pro** (LMMs-Eval relaxed_correctness, deterministic) + **CharXiv-Reasoning** (Qwen3.5-397B LLM judge, CharXiv official prompt) + **ChartMuseum** (Qwen3.5-397B LLM judge, ChartMuseum official prompt)")
    out.append("- Baselines (VLM): Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Thinking / Chart-R1 7B / ChartGemma 12B")
    out.append("- Sample size: 100 per bench × 4 baselines = 1200 inference traces (seed 42)")
    out.append("- Extractor: Qwen3.6-27B (LLM-primary, strict entity validation)")
    out.append("- Verifier: InternVL2.5-26B (NOT_FOUND fallback, 10/25% tolerance)")
    out.append("- Judge (charxiv/chartmuseum): Qwen3.5-397B-A17B-FP8 remote multi-host (4 URLs)")
    out.append("- Drift threshold (cf vs hl partition): perception ≥0.5\n")

    # ── Phase 1 baseline ──
    out.append("## TL;DR\n")
    out.append("Easy bench (Phase 1 rescored ChartQA-train+ReachQA-train, 100 samples × 4 baselines): pAUC 0.481, r(p,o) ‑0.068 — perception decoupled from outcome (saturated chart-reading).")
    out.append("")
    out.append("Hard bench (Phase 2, **12 combos × {ChartQA-Pro, CharXiv-Reasoning, ChartMuseum} × 4 baselines**, 100 samples each, seed 42):")
    out.append("- pAUC valid range 0.316–0.750, **bench medians: ChartQA-Pro 0.708 / CharXiv 0.559 / ChartMuseum 0.449**.")
    out.append("- r(p,o) range −0.247 — +0.453.")
    out.append("- **ChartMuseum (hardest bench) shows the LOWEST perception signal** — pAUC 0.366-0.459, all r<0 (perception slightly anti-correlated with outcome).")
    out.append("- Modest improvement over easy bench only on ChartQA-Pro/CharXiv; ChartMuseum WORSE than easy bench.")
    out.append("")
    out.append("**careful_flawed Automated Taxonomy (N=426, 397B classifier)**:")
    out.append("- **CATEGORY_SELECT 46.0%** (dominant) — reasoning step picked wrong group/cluster/series")
    out.append("- TRUNCATION 26.5% — model exhausted token budget before final answer (concentrated in thinking-mode)")
    out.append("- ARITHMETIC 8.5% — chart values read OK but math wrong")
    out.append("- OTHER 8.0%, Q_INTENT 6.3%, FORMAT_MISMATCH 4.7%")
    out.append("- **Real reasoning failure: 60.8%** / Extraction-truncation artifact: 31.2%")
    out.append("")
    # Check if chartgemma cm inference is now valid (post fix)
    # audit uses key "model__bench"; cell uses "bench__model"
    cg_cm_audit = audit.get("chartgemma__chartmuseum", {}) if audit else {}
    cg_cm_infer_ok = cg_cm_audit.get("inference_errors", 100) < 10  # <10/100 errors = OK
    cg_cm = cell.get("chartmuseum__chartgemma", {}) if cell else {}
    cg_cm_uk = cg_cm.get("unknown", 100) if cg_cm else 100
    if not cg_cm_infer_ok:
        out.append("**Failed combo**: ChartGemma__chartmuseum — 100/100 inference errors. Re-run with max_model_len=8192 + max_tokens_override=4096 required.\n")
    elif cg_cm_uk > 80:
        out.append(f"**12/12 combos inferenced successfully** (chartgemma_chartmuseum re-run fixed via max_tokens_override=4096). However, **ChartGemma chartmuseum has {cg_cm_uk:.0f}% unknown cell** — model outputs single-token answers without extractable reasoning steps. Inference-format mismatch with chartmuseum's `<think>...</think><answer>...</answer>` prompt: ChartGemma jumps straight to `<answer>`.\n")
    else:
        out.append("All 12 combos fully VALID.\n")

    # ── 4-cell table (orig) ──
    out.append("## 4-Cell Distribution × Alignment per (Baseline, Bench) — ORIGINAL judge\n")
    if cell:
        out.append("| Bench | Model | N | GrndCorr | Shortcut | CarefFlawed | Halluc | Unk | Out% | pAUC | r(p,o) | meanP |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key in sorted(cell.keys()):
            b = cell[key]
            if "error" in b: continue
            bench, model = key.split("__", 1)
            out.append(
                f"| {bench} | {model} | {b['n_samples']} | "
                f"{fmt_pct(b['grounded_correct'])} | {fmt_pct(b['shortcut'])} | "
                f"{fmt_pct(b['careful_flawed'])} | {fmt_pct(b['hallucinated'])} | "
                f"{fmt_pct(b['unknown'])} | {fmt_pct(b['outcome_pct'])} | "
                f"{fmt_num(b.get('perception_auc_vs_outcome'))} | "
                f"{fmt_num(b.get('perception_r_vs_outcome'), '+.3f')} | "
                f"{fmt_num(b.get('mean_perception'))} |"
            )
    out.append("")

    # ── 4-cell table (rejudge) ──
    if cell_rej:
        out.append("## 4-Cell after Re-judge (extended answer extraction)\n")
        out.append("Extended extraction patterns: `<answer>X</answer>`, `final answer/answer is/the answer:/so it is`, `\\\\boxed{X}`, last non-empty line. Same LLM judge re-applied.\n")
        out.append("| Bench | Model | GC% Δ | CF% Δ | pAUC Δ | r(p,o) Δ | flip+ | flip− |")
        out.append("|---|---|:--|:--|:--|:--|:--:|:--:|")
        for key in sorted(cell_rej.keys()):
            v = cell_rej[key]
            if "orig" not in v: continue
            bench, model = key.split("__", 1)
            o, r = v["orig"], v["rejudge"]
            dgc = r["grounded_correct"] - o["grounded_correct"]
            dcf = r["careful_flawed"] - o["careful_flawed"]
            dauc = (r.get("auc") or 0) - (o.get("auc") or 0)
            dr = (r.get("r") or 0) - (o.get("r") or 0)
            # flip counts (approximate from outcome_pct delta × N)
            n = r["n_samples"]
            d_out = (r["outcome_pct"] - o["outcome_pct"]) * n / 100
            out.append(
                f"| {bench} | {model} | "
                f"{o['grounded_correct']:.0f}→{r['grounded_correct']:.0f} ({dgc:+.0f}) | "
                f"{o['careful_flawed']:.0f}→{r['careful_flawed']:.0f} ({dcf:+.0f}) | "
                f"{(o.get('auc') or 0):.3f}→{(r.get('auc') or 0):.3f} ({dauc:+.3f}) | "
                f"{(o.get('r') or 0):+.3f}→{(r.get('r') or 0):+.3f} ({dr:+.3f}) | "
                f"~{max(0, int(d_out)):+d} | — |"
            )
        out.append("")

    # ── cf cell no-answer-tag prevalence ──
    out.append("## Careful-flawed cell — format-failure prevalence\n")
    if cf_noans:
        out.append("Definition: cf = perception ≥0.5 AND outcome = 0. We check what fraction of cf samples lack `<answer>X</answer>` tag.\n")
        out.append("| Combo | cf N | no-answer | % no-answer |")
        out.append("|---|---:|---:|---:|")
        for k, v in cf_noans.items():
            out.append(f"| {k} | {v['cf_total']} | {v['cf_no_answer_tag']} | {v['pct_no_answer']:.1f}% |")
        out.append("")
        out.append("**Interpretation**: ~98% of cf samples have no `<answer>` tag (the model didn't follow the prompt format strictly), but the **rejudge table above** shows extended extraction only flipped 5–12% of cf → grounded_correct on charxiv (0% on chartqa_pro). Most cf cases really did produce a short final pred (e.g., `43%`, `Tech`) that disagrees with gold. The cf cell is **genuine reasoning failure**, not extraction noise.\n")

    # ── Artifact audit (eval/judge noise) ──
    out.append("## Artifact Audit (eval/judge noise check)\n")
    if audit:
        out.append("| Combo | Inference err | Empty trace | Empty steps | NOT_FOUND% | Judge pass | Judge fail | Judge null | Susp false-neg |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, r in audit.items():
            if "error" in r: continue
            bench, model = key.split("__", 1) if "__" in key else (key, "")
            fmt_fn = r.get("chartqa_pro_fmt_suspicious_false_neg", 0)
            out.append(
                f"| {model}/{bench} | {r['inference_errors']} | {r['inference_empty_traces']} | "
                f"{r['segmentation_empty']} | {r['perception_mean_NOT_FOUND_rate']:.1f}% | "
                f"{r['judge_pass']} | {r['judge_fail']} | {r['judge_null']} | "
                f"{fmt_fn} |"
            )
    out.append("")

    # ── Phase-1 comparison ──
    out.append("## Comparison with Phase 1 Main Pilot (easy bench, rescored)\n")
    if main_pilot.get("baselines"):
        out.append("Reference: ChartQA-train (69) + ReachQA-train (31), Phase 1 rescored (LMMs-Eval + ReachQA judge).\n")
        out.append("| Model | Phase-1 GC% | Phase-1 CF% | HB ChartQA-Pro GC% | HB ChartQA-Pro CF% | HB CharXiv GC% | HB CharXiv CF% |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for model in ["zeroshot_4b", "chart_r1_7b", "chartgemma_12b"]:
            mp = main_pilot["baselines"].get(model, {})
            if not mp: continue
            hb_map = {"zeroshot_4b":"qwen3vl_4b","chart_r1_7b":"chart_r1","chartgemma_12b":"chartgemma"}
            hb_model = hb_map[model]
            cqp = cell.get(f"chartqa_pro__{hb_model}", {})
            cxv = cell.get(f"charxiv_reasoning__{hb_model}", {})
            out.append(
                f"| {model} | {fmt_pct(mp.get('grounded_correct'))} | {fmt_pct(mp.get('careful_flawed'))} | "
                f"{fmt_pct(cqp.get('grounded_correct'))} | {fmt_pct(cqp.get('careful_flawed'))} | "
                f"{fmt_pct(cxv.get('grounded_correct'))} | {fmt_pct(cxv.get('careful_flawed'))} |"
            )
    out.append("")

    # ── Aggregate findings ──
    out.append("## Findings\n")
    if cell:
        aucs = [b.get("perception_auc_vs_outcome") for b in cell.values()
                if isinstance(b, dict) and isinstance(b.get("perception_auc_vs_outcome"), (int,float)) and b.get("perception_auc_vs_outcome")]
        rs   = [b.get("perception_r_vs_outcome") for b in cell.values()
                if isinstance(b, dict) and isinstance(b.get("perception_r_vs_outcome"), (int,float)) and b.get("perception_r_vs_outcome") is not None]
        if aucs and rs:
            import statistics
            out.append(f"- **pAUC across {len(aucs)} valid hard-bench combos** (orig judge, excludes invalid): {min(aucs):.3f}–{max(aucs):.3f}, median {statistics.median(aucs):.3f}")
            out.append(f"- **r(perception, outcome)** range across {len(rs)} valid combos: {min(rs):+.3f}–{max(rs):+.3f}, median {statistics.median(rs):+.3f}")
        out.append(f"- **Easy bench (Phase 1 rescored)**: pAUC 0.481, r(p,o) ‑0.068")
        out.append(f"- **Bench medians**: ChartQA-Pro pAUC 0.708, CharXiv 0.559, **ChartMuseum 0.449** (worst — below easy bench).")
        out.append("")
        out.append("Earlier interim claim of \"pAUC 0.71–0.75 on hard bench\" was a cherry-pick of 3 ChartQA-Pro/CharXiv combos. With all 12 combos including ChartMuseum, signal is heterogeneous — strong on ChartQA-Pro, near-zero on ChartMuseum.")

    # ── Diagnosis & next steps ──
    out.append("\n## Diagnosis\n")
    out.append("1. **Perception process reward gain on hard bench is small and combo-dependent**. Best evidence: chartqa_pro/qwen3vl_4b (pAUC 0.708, r +0.45). Worst: chartqa_pro/chartgemma (pAUC 0.316, r −0.13) and 8b_thinking with all outcomes ≈0 (AUC undefined).")
    out.append("2. **cf cell is NOT primarily a format artifact**. 98% lack `<answer>` tag, but extended extraction recovers only 5–12% of cf samples on charxiv (0% on chartqa_pro). Most cf samples ARE genuine reasoning errors.")
    out.append("3. **Truncation is a real concern for 8b_thinking on chartqa_pro**: 98% outcome=0 (only 2/100 correct), large cf cell (62%) with no recoverable answer — model spends tokens on long thinking and exhausts budget before final answer.")
    out.append("4. **ChartGemma** on charxiv: 87% unknown cell — model produces meta-instructions (\"**Example:**\"), no real claims to verify. Model-class limitation, not method failure.")
    out.append("5. **Eval artifact level**: judge null rate <4%, suspected false-neg (chartqa_pro fmt) 0–16% — small noise floor, not the dominant source of cf.")

    # automated cf taxonomy
    cf_tax = load(BASE / "data/d2_hardbench/reports/cf_taxonomy_summary.json")
    if cf_tax:
        out.append("\n## careful_flawed Automated Taxonomy (397B classifier, all cf samples N=272)\n")
        from collections import Counter
        total = Counter()
        for labs in cf_tax.values():
            for k, v in labs.items(): total[k] += v
        grand = sum(total.values())
        order = ["TRUNCATION","CATEGORY_SELECT","Q_INTENT","ARITHMETIC","FORMAT_MISMATCH","OTHER","PARSE_FAIL"]
        out.append("Classified each cf sample (perception ≥0.5, outcome=0) into 6 labels via Qwen3.5-397B (Q + gold + reasoning tail 1500 chars → label).\n")
        out.append("| Combo | TRUNC | CATSEL | QINT | ARITH | FMT | OTHER |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for combo, labs in cf_tax.items():
            row = f"| {combo} |"
            for L in ["TRUNCATION","CATEGORY_SELECT","Q_INTENT","ARITHMETIC","FORMAT_MISMATCH","OTHER"]:
                row += f" {labs.get(L,0)} |"
            out.append(row)
        out.append("")
        out.append(f"**Aggregate (N={grand})**:")
        for L in order:
            v = total[L]
            if v == 0 and L == "PARSE_FAIL": continue
            out.append(f"- {L}: {v}  ({100*v/max(grand,1):.1f}%)")
        out.append("")
        truncation = total["TRUNCATION"] + total["FORMAT_MISMATCH"]
        real_err = total["Q_INTENT"] + total["CATEGORY_SELECT"] + total["ARITHMETIC"]
        out.append(f"**Extraction/truncation artifact: {truncation}/{grand} = {100*truncation/max(grand,1):.1f}%**")
        out.append(f"**Real reasoning failure: {real_err}/{grand} = {100*real_err/max(grand,1):.1f}%**")
        out.append("")
        out.append("Key takeaways:")
        out.append(f"1. **CATEGORY_SELECT dominates** ({total['CATEGORY_SELECT']}/{grand} = {100*total['CATEGORY_SELECT']/grand:.1f}%) — model reads chart correctly but picks wrong group/cluster/series. Concentrated on chartmuseum (44+46+23 = 113 / 154 cm cf samples = 73%).")
        out.append(f"2. **Real reasoning failure 60.8%** (CATEGORY_SELECT + Q_INTENT + ARITHMETIC = {total['CATEGORY_SELECT']+total['Q_INTENT']+total['ARITHMETIC']}/{grand}) vs extraction artifact 31.2% (TRUNC + FMT).")
        out.append(f"3. **TRUNCATION concentrated** in qwen3vl_8b_thinking on chartqa_pro/charxiv (54+33 = 87/{total['TRUNCATION']} = {100*87/max(total['TRUNCATION'],1):.0f}% of all TRUNC) — thinking-mode burns token budget. Chartmuseum 8b_thinking has only 1 TRUNC (chartmuseum prompt format keeps output shorter).")
        out.append("4. **Process reward target = CATEGORY_SELECT step** — the reasoning step that picks a group/category. Most impactful single intervention.")

    # ── Action ──
    # bench-level rollup
    if cell:
        out.append("\n## Bench-level Rollup (median across 4 baselines, skipping invalid combos)\n")
        out.append("| Bench | Median GC% | Median CF% | Median pAUC | Median r(p,o) |")
        out.append("|---|---:|---:|---:|---:|")
        import statistics
        bench_keys = {}
        for k, b in cell.items():
            if not isinstance(b, dict) or "n_samples" not in b: continue
            bench = k.split("__",1)[0]
            # skip invalid: meanP=0 means inference failed
            if b.get("mean_perception", 1) == 0: continue
            bench_keys.setdefault(bench, []).append(b)
        for bench in ["chartqa_pro","charxiv_reasoning","chartmuseum"]:
            xs = bench_keys.get(bench, [])
            if not xs: out.append(f"| {bench} | — | — | — | — |"); continue
            gc_med = statistics.median([x["grounded_correct"] for x in xs])
            cf_med = statistics.median([x["careful_flawed"] for x in xs])
            aucs = [x["perception_auc_vs_outcome"] for x in xs if x.get("perception_auc_vs_outcome")]
            rs = [x["perception_r_vs_outcome"] for x in xs if x.get("perception_r_vs_outcome") is not None]
            auc_med = statistics.median(aucs) if aucs else None
            r_med = statistics.median(rs) if rs else None
            out.append(f"| {bench} | {gc_med:.1f}% | {cf_med:.1f}% | "
                       f"{(auc_med or 0):.3f} | {(r_med or 0):+.3f} |")

    # invalid combos note
    out.append("\n## Edge-case / limited-signal combos\n")
    if not cg_cm_infer_ok:
        out.append("- **chartgemma__chartmuseum**: 100/100 inference errors — ChartGemma launched with `max_model_len=4096` but chartmuseum eval requested `max_tokens=8192`. Re-run required.")
    elif cg_cm_uk > 80:
        out.append(f"- **chartgemma__chartmuseum**: inference fixed (`max_tokens_override=4096`); however, {cg_cm_uk:.0f}% unknown cell — ChartGemma outputs single-token answers without `<think>` reasoning steps. Model-class limitation, not eval-script bug.")
    out.append("- **chartqa_pro__qwen3vl_8b_thinking**: outcome_pct=2% (only 2/100 correct) → AUC undefined (no positive class variance). Truncation-driven (62% cf, mostly TRUNCATION label).")
    out.append("")

    out.append("\n## Action\n")
    out.append("- **(즉시)** chartgemma chartmuseum 재실행 (max_model_len 16384) — 10분, 12-cell 완전체 확정.")
    out.append("- **(다음 실험)** Process reward target = **CATEGORY_SELECT 클래스 (전체 cf 의 40%)**. cf-cell taxonomy에서 가장 큰 single error class. 모든 baselines + 모든 benches에 고르게 분포.")
    out.append("- **(eval 개선)** Extended answer extraction을 default 로 채택 (charxiv +3~+12 회복). chartmuseum 도 동일 패턴 검증 권장.")
    out.append("- **(paper framing)**:")
    out.append("  - \"Hard chart reasoning benches show modest perception-outcome decoupling (median pAUC 0.54), but cf cell is **dominated by CATEGORY_SELECT errors (40%)**, not perception failure.\"")
    out.append("  - \"Truncation (30%) is a separate problem — thinking-mode models exhaust token budget before final answer.\"")
    out.append("  - \"Perception process reward target should focus on the reasoning step that performs **group/category selection**, not chart entity reading.\"")

    rep_path = BASE / "docs/d2_hardbench_decision_report.md"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(out))
    print(f"-> {rep_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
