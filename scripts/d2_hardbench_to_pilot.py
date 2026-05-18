"""Convert eval_standard.py inference output → pilot-format jsonl (segmented).

Input: data/d2_hardbench/inference/<model>/<bench>.jsonl
Output: data/d2_hardbench/segmented/<model>/<bench>.jsonl
  with schema {id, source, image_path, question, gold_answer, content,
               reasoning_content, raw_response, steps, n_steps, bench, scoring}

Uses d2_segment_baseline.py logic (no <think>, fallback whole text).
Tokenizer per-baseline: 4B uses Qwen3VL-4B, 7B uses Qwen2.5-VL-7B, etc.
"""
from __future__ import annotations

import argparse, json, os, re, sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

TOKENIZER_MAP = {
    "qwen3vl_4b":            BASE / "models/qwen3vl-4b-instruct",
    "qwen3vl_8b_thinking":   BASE / "models/qwen3vl-8b-thinking",
    "chart_r1":              BASE / "models/chart_r1_7b",
    "chartgemma":            BASE / "models/chartgemma_12b",
}


def segment_steps(text: str, tokenizer, target_min: int = 60, target_max: int = 560) -> list[str]:
    if not text or not text.strip(): return []
    text = re.sub(r"(?i)\b(final answer|the answer is)\s*:?.*$", "", text, flags=re.DOTALL).strip()
    if not text: return []
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if not chunks: chunks = [text]
    merged, buf = [], ""
    for c in chunks:
        candidate = (buf + "\n\n" + c).strip() if buf else c
        tlen = len(tokenizer.encode(candidate, add_special_tokens=False))
        if tlen < target_min:
            buf = candidate
        else:
            if buf: merged.append(buf)
            buf = c
    if buf: merged.append(buf)
    final = []
    for seg in merged:
        seglen = len(tokenizer.encode(seg, add_special_tokens=False))
        if seglen <= target_max:
            final.append(seg); continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", seg)
        cur = ""
        for p in parts:
            cand = (cur + " " + p).strip() if cur else p
            if len(tokenizer.encode(cand, add_special_tokens=False)) > target_max and cur:
                final.append(cur); cur = p
            else:
                cur = cand
        if cur: final.append(cur)
    return [s for s in final if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="qwen3vl_4b / chart_r1 / chartgemma / qwen3vl_8b_thinking")
    ap.add_argument("--bench", required=True, help="chartqa_pro / charxiv_reasoning / chartmuseum")
    ap.add_argument("--target_min", type=int, default=60)
    ap.add_argument("--target_max", type=int, default=560)
    args = ap.parse_args()

    inp = BASE / f"data/d2_hardbench/inference/{args.model}/{args.bench}.jsonl"
    out = BASE / f"data/d2_hardbench/segmented/{args.model}/{args.bench}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        print(f"missing input: {inp}"); return 1
    tok_path = TOKENIZER_MAP.get(args.model)
    if not tok_path or not tok_path.exists():
        # fallback to Qwen 4B
        tok_path = BASE / "models/qwen3vl-4b-instruct"
    print(f"[hb_to_pilot] {args.model}:{args.bench}  tokenizer={tok_path.name}")
    tok = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)

    recs = [json.loads(l) for l in open(inp) if l.strip()]
    written = 0
    with open(out, "w") as f:
        for r in recs:
            content = r.get("content", "") or ""
            reasoning = r.get("reasoning_content", "") or ""
            # For thinking models: reasoning_content has the trace
            # For non-thinking: content has it
            trace = reasoning if reasoning else content
            if r.get("error"):
                steps = []
            else:
                steps = segment_steps(trace, tok, args.target_min, args.target_max)
            out_rec = {
                "id": r["sample_id"],
                "source": r["bench"],
                "bench": r["bench"],
                "scoring": r.get("scoring", ""),
                "image_path": r["image_path"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "content": content,
                "reasoning_content": reasoning,
                "raw_response": trace,
                "steps": steps,
                "n_steps": len(steps),
                "baseline_model": args.model,
            }
            for k in ("inst_category", "figure_id", "question_type", "reasoning_type"):
                if k in r: out_rec[k] = r[k]
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1
    # Stats
    all_recs = [json.loads(l) for l in open(out) if l.strip()]
    ns = [r["n_steps"] for r in all_recs]
    print(f"  written {written}  median_steps={int(np.median(ns)) if ns else 0}  "
          f"empty_step_rate={sum(1 for n in ns if n==0)/max(len(ns),1)*100:.1f}%")
    print(f"  -> {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
