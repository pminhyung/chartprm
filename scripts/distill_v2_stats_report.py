"""Emit a Markdown report comparing the v8 distill v2 corpus across stages:
raw distill → format-compliance build → quality filter.

Produces ``docs/v8_distill_v2_corpus_stats.md`` so the user has a durable,
reviewable record without scraping pipeline logs.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import quantiles


def count_jsonl(path: str) -> int:
    if not Path(path).exists():
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def family_breakdown(path: str) -> Counter:
    c: Counter = Counter()
    if not Path(path).exists():
        return c
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c[d.get("_format_family", "?")] += 1
    return c


def drop_reason_breakdown(path: str) -> Counter:
    c: Counter = Counter()
    if not Path(path).exists():
        return c
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c[d.get("_drop_reason", "?")] += 1
    return c


def token_length_stats(path: str, sample_size: int = 1000) -> dict:
    """Approximate `assistant_text` token length (4 chars/token heuristic — fast,
    no tokenizer needed). Reports p50/p75/p90/p95/p99/max."""
    if not Path(path).exists():
        return {}
    lens = []
    with open(path) as f:
        import random

        random.seed(0)
        lines = f.readlines()
        if len(lines) > sample_size:
            lines = random.sample(lines, sample_size)
        for line in lines:
            d = json.loads(line)
            txt = d.get("assistant_text", "") or ""
            lens.append(len(txt) // 4)  # rough chars→tokens
    if not lens:
        return {}
    lens.sort()
    n = len(lens)
    return {
        "n_sampled": n,
        "p50": lens[int(n * 0.50)],
        "p75": lens[int(n * 0.75)],
        "p90": lens[int(n * 0.90)],
        "p95": lens[int(n * 0.95)],
        "p99": lens[int(n * 0.99)],
        "max": lens[-1],
    }


def sample_dropped(path: str, n: int = 3) -> list[tuple[str, dict]]:
    """Return up to n rows per drop reason for qualitative inspection."""
    seen: dict[str, list[dict]] = {}
    if not Path(path).exists():
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            r = d.get("_drop_reason", "?")
            if r not in seen:
                seen[r] = []
            if len(seen[r]) < n:
                seen[r].append(d)
    out = []
    for r, rows in sorted(seen.items()):
        for d in rows:
            out.append((r, d))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_raw", default="data/sft_v8_improved_v2.jsonl")
    p.add_argument("--distill", default="data/sft_v8_distill_v2.jsonl")
    p.add_argument("--distill_failed", default="data/sft_v8_distill_v2_failed.jsonl")
    p.add_argument("--distill_retry_failed", default="data/sft_v8_distill_v2_failed_retry.jsonl")
    p.add_argument("--build_kept", default="data/sft_v8_distill_v2_sft.jsonl")
    p.add_argument("--build_dropped", default="data/sft_v8_distill_v2_sft_dropped.jsonl")
    p.add_argument("--filter_kept", default="data/sft_v8_distill_v2_sft_filtered.jsonl")
    p.add_argument("--filter_dropped", default="data/sft_v8_distill_v2_sft_filtered_dropped.jsonl")
    p.add_argument("--output", default="docs/v8_distill_v2_corpus_stats.md")
    args = p.parse_args()

    n_raw_input = count_jsonl(args.input_raw)
    n_distill = count_jsonl(args.distill)
    n_distill_failed = count_jsonl(args.distill_failed)
    n_retry_failed = count_jsonl(args.distill_retry_failed)
    n_build_kept = count_jsonl(args.build_kept)
    n_build_dropped = count_jsonl(args.build_dropped)
    n_filter_kept = count_jsonl(args.filter_kept)
    n_filter_dropped = count_jsonl(args.filter_dropped)

    fam_distill = family_breakdown(args.distill)
    fam_build = family_breakdown(args.build_kept)
    fam_filter = family_breakdown(args.filter_kept)

    build_drops = drop_reason_breakdown(args.build_dropped)
    filter_drops = drop_reason_breakdown(args.filter_dropped)

    tok_build = token_length_stats(args.build_kept)
    tok_filter = token_length_stats(args.filter_kept)

    samples = sample_dropped(args.filter_dropped, n=2)

    out = []
    out.append("# v8 SFT Distill v2 — Corpus Statistics\n")
    out.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    out.append("## Pipeline funnel\n")
    out.append("| Stage | Count | Note |")
    out.append("|---|---:|---|")
    out.append(f"| Input rows (`sft_v8_improved_v2`) | {n_raw_input} | source v8 corpus |")
    out.append(f"| Distillation kept | {n_distill} | Qwen3.5-397B teacher (1st pass + retry) |")
    out.append(f"| Distill 1st-pass failed | {n_distill_failed} | HTTP timeout (recoverable) |")
    out.append(f"| Distill retry failed | {n_retry_failed} | timeout > 1800 s — truly unrecoverable |")
    out.append(f"| Format-compliance build kept | {n_build_kept} | drop {n_build_dropped}: missing `<answer>` etc. |")
    out.append(f"| **Quality-filter kept (SFT input)** | **{n_filter_kept}** | drop {n_filter_dropped}: R1-R4 |")
    out.append("")

    out.append("## Family distribution\n")
    out.append("| Family | After distill | After build | After filter | Δ filter (%) |")
    out.append("|---|---:|---:|---:|---:|")
    families = sorted(set(fam_distill) | set(fam_build) | set(fam_filter))
    for f in families:
        d = fam_distill.get(f, 0)
        b = fam_build.get(f, 0)
        fl = fam_filter.get(f, 0)
        delta = (b - fl) * 100 / max(1, b) if b else 0
        out.append(f"| {f} | {d} | {b} | {fl} | {delta:.1f}% |")
    out.append("")

    out.append("## Drop reasons\n")
    out.append("### Format-compliance build")
    if build_drops:
        out.append("| Reason | Count |")
        out.append("|---|---:|")
        for k, v in sorted(build_drops.items(), key=lambda x: -x[1]):
            out.append(f"| `{k}` | {v} |")
    else:
        out.append("(none)")
    out.append("")
    out.append("### Quality filter (R1-R4)")
    if filter_drops:
        out.append("| Reason | Count | Description |")
        out.append("|---|---:|---|")
        descriptions = {
            "R1_reasoning_too_short": "len(_teacher_reasoning) < 150 chars",
            "R2_cm_answer_body_long": "ChartMuseum `_clean_answer` > 200 chars",
            "R3_cm_answer_body_multiline": "newline inside ChartMuseum answer body",
            "R4a_numeric_mismatch": "answer_type=numeric AND teacher mismatch (>5%)",
            "R4b_mc_value_form_mismatch": "value-form multichoice AND teacher wrong",
        }
        for k, v in sorted(filter_drops.items(), key=lambda x: -x[1]):
            out.append(f"| `{k}` | {v} | {descriptions.get(k, '')} |")
    else:
        out.append("(none)")
    out.append("")

    out.append("## `assistant_text` length distribution (approx tokens, char/4)\n")
    out.append("| Stage | n_sampled | p50 | p75 | p90 | p95 | p99 | max |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage, t in [("after build", tok_build), ("after filter", tok_filter)]:
        if t:
            out.append(
                f"| {stage} | {t['n_sampled']} | {t['p50']} | {t['p75']} | "
                f"{t['p90']} | {t['p95']} | {t['p99']} | {t['max']} |"
            )
    out.append("")

    if samples:
        out.append("## Sample dropped rows (quality filter)\n")
        for reason, d in samples:
            gold = str(d.get("answer", ""))[:80]
            pred = str(d.get("_teacher_pred", ""))[:80]
            answer_type = d.get("answer_type", "")
            is_mc = "yes" if d.get("_is_multichoice") else "no"
            out.append(f"- **`{reason}`** | `answer_type={answer_type}`, `mc={is_mc}`")
            out.append(f"  - gold: `{gold}`")
            out.append(f"  - teacher pred: `{pred}`")
        out.append("")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"[stats] report written → {args.output}")
    print(f"  filter: {n_build_kept} → {n_filter_kept} (-{n_build_kept-n_filter_kept}, {(n_build_kept-n_filter_kept)*100/max(1,n_build_kept):.1f}%)")


if __name__ == "__main__":
    main()
