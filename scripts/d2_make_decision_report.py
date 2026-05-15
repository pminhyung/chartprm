"""Generate docs/d1d2_decision_report.md from collected results.

Pulls:
- data/d1_pilot/verifier_reliability.summary.json
- data/d2_pilot/alignment_v2.json
- data/d2_pilot/4cell_distribution.json
"""
import json
import os
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def load_optional(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt_pct(v: float | None, spec: str = ".1f") -> str:
    if v is None:
        return "—"
    return format(v, spec) + "%"


def fmt_num(v: float | None, spec: str = ".3f") -> str:
    if v is None:
        return "na"
    return format(v, spec)


def main():
    rel = load_optional(BASE / "data/d1_pilot/verifier_reliability.summary.json")
    align = load_optional(BASE / "data/d2_pilot/alignment_v2.json")
    cell = load_optional(BASE / "data/d2_pilot/4cell_distribution.json")

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# D1+D2 Decision Report — {today} (open-weights, Spec A+B+C+D patched)\n"]

    # Setup section
    lines.append("## Setup\n")
    lines.append("- Verifier: **InternVL2.5-26B** (cross-family, NOT_FOUND fallback, 10%/25% tolerance)")
    lines.append("- Extractor: **Qwen3.6-27B** (LLM-primary with strict entity validation)")
    lines.append("- Baselines: zero-shot Qwen3.5-VL-4B (archived), Chart-R1 7B, ChartGemma 12B")
    lines.append("- Sample pool: 100 D2 pilot samples (ChartQA+ReachQA train, pHash-clean)\n")

    # Day 1 — Reliability
    lines.append("## Day 1.A — Verifier reliability (CSV-based)\n")
    if rel:
        lines.append(f"- N queries: {rel['n_total']}")
        lines.append(f"- OK (parsed): {rel['ok']} ({fmt_pct(rel['coverage_pct'])})")
        lines.append(f"- NOT_FOUND : {rel['not_found']} ({fmt_pct(rel['not_found_pct'])})")
        lines.append(f"- Agreement (on parsed): **{fmt_pct(rel['agreement_pct'])}**")
        lines.append(f"- Criteria: {rel.get('criteria','—')}")
        lines.append(f"- **VERDICT: {'PASS' if rel['pass'] else 'FAIL'}**\n")
    else:
        lines.append("- (pending)\n")

    # Day 1 — Alignment
    lines.append("## Day 1.B — Patched pipeline alignment (4-metric vs outcome)\n")
    if align:
        a = align
        lines.append(f"- N samples: {a['n']}, outcome correct rate: {fmt_pct(a['outcome_correct_rate']*100)}")
        lines.append(f"- Mean NOT_FOUND rate per sample: {fmt_pct(a.get('not_found_rate_mean',0)*100)}\n")
        lines.append("| Metric | Value | Target | Pass |")
        lines.append("|---|---:|---|:---:|")
        c = a["checks"]
        lines.append(f"| Perception ROC AUC vs outcome | {fmt_num(a['perception_auc_vs_outcome'])} | ≥0.65 | {'PASS' if c['perception_auc_vs_outcome'] else 'FAIL'} |")
        lines.append(f"| MC ROC AUC vs outcome         | {fmt_num(a['mc_auc_vs_outcome'])} | ≥0.70 | {'PASS' if c['mc_auc_vs_outcome'] else 'FAIL'} |")
        lines.append(f"| r(perception, mc)             | {fmt_num(a['perception_mc_correlation'])} | 0.3-0.7 | {'PASS' if c['complementarity_in_range'] else 'FAIL'} |")
        lines.append(f"| AUC lift (mc+p vs mc)         | {fmt_num(a['auc_lift'], '+.3f')} | ≥+0.02 | {'PASS' if c['auc_lift_positive'] else 'FAIL'} |")
        lines.append(f"\n- **passed: {a['passed']}/4**\n")
        lines.append("- NOTE: 4-cell distribution이 main decision metric. 통과 못 해도 Day 2로 진행 (decoupling 자체가 thesis).\n")
    else:
        lines.append("- (pending)\n")

    # Day 2 — 4-cell distribution
    lines.append("## Day 2 — 4-cell distribution (drift × outcome)\n")
    if cell:
        thr = cell.get("drift_threshold", 0.5)
        lines.append(f"_Drift threshold (perception_agg ≥ X → low drift): **{thr}**_\n")
        lines.append("| Model | N | Grounded+Correct | Shortcut | CarefulFlawed | Hallucinated | Unknown | Outcome% | DriftWithinCorrect% |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, b in cell.get("baselines", {}).items():
            if "error" in b:
                lines.append(f"| {name} | — | ERROR: {b['error']} | | | | | | |")
                continue
            lines.append(
                f"| {name} | {b['n_samples']} | {fmt_pct(b['grounded_correct'])} | {fmt_pct(b['shortcut'])} | "
                f"{fmt_pct(b['careful_flawed'])} | {fmt_pct(b['hallucinated'])} | {fmt_pct(b['unknown'])} | "
                f"{fmt_pct(b['outcome_correct_rate'])} | {fmt_pct(b['drift_rate_within_correct'])} |"
            )
        lines.append(f"\n### Pattern detected: **{cell.get('pattern','—')}**\n")
    else:
        lines.append("- (pending)\n")

    # Decision
    lines.append("## Decision\n")
    if cell and "pattern" in cell:
        p = cell["pattern"]
        if p.startswith("A"):
            lines.append("**Phase 2 GO** — drift framing supported by data, proceed to Week 2 PRM training data construction (D8-D10).")
            if "strong" in p:
                lines.append("Sub-thesis '*drift varies by training paradigm*'를 paper §1에 추가 권장.")
        elif p.startswith("B"):
            lines.append("**MC-only pivot** — perception term 폐기 또는 w_p=0.1 minor weight. Main contribution: Math-Shepherd chart-domain application.")
        elif p.startswith("C"):
            lines.append("**Reconsider** — ChartGemma의 SFT-grounded effect가 step-level grounding으로 cascade되어 우리 차별화 약화. RL+PRM 특화 angle 또는 step-level grounded SFT data 합성 angle로 framing 재정의 권장.")
        else:
            lines.append(f"**TBD** (pattern: {p})")
    else:
        lines.append("- (4-cell distribution 미완 — 결정 대기)\n")

    # Reproducibility
    lines.append("\n## Reproducibility\n")
    lines.append("**Data**:")
    lines.append("- `data/d2_pilot/claims_v2.jsonl` — main 100 sample LLM-primary claims")
    lines.append("- `data/d2_pilot/perception_v2.jsonl` — InternVL verifier scores")
    lines.append("- `data/d2_pilot/alignment_v2.json` — 4-metric alignment")
    lines.append("- `data/d2_pilot/baseline_{zeroshot_4b,chart_r1,chartgemma}_perception.jsonl` — per-baseline perception")
    lines.append("- `data/d2_pilot/4cell_distribution.json` — Pattern A/B/C source\n")
    lines.append("**Scripts**:")
    lines.append("- `scripts/d2_claim_extract_v2.py` (LLM-primary, multi-host async, resume)")
    lines.append("- `scripts/d2_perception_verify_v2.py` (NOT_FOUND fallback, multi-host)")
    lines.append("- `scripts/d2_alignment_v2.py` / `scripts/d2_4cell_distribution.py`")
    lines.append("- `scripts/d1_verifier_reliability_csv.py` (CSV-based reliability)\n")
    lines.append("**vLLM servers** (10 GPU: 2-11):")
    lines.append("- 8100/8101 InternVL2.5-26B verifier (TP=2 × 2 hosts)")
    lines.append("- 8200 Qwen3.6-27B extractor (TP=4)")
    lines.append("- 8300 Chart-R1 7B, 8301 ChartGemma 12B (TP=1 each)\n")

    out = "\n".join(lines)
    out_path = BASE / "docs/d1d2_decision_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out)
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
