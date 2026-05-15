"""D2 — Segment baseline raw_response into steps for perception pipeline.

Adapts d1_segment.py logic for baseline outputs (no <think> tag).
Treats the entire raw_response as reasoning content to segment.

Input:  baseline_*.jsonl (has 'raw_response')
Output: baseline_*_segmented.jsonl  (adds 'steps', 'n_steps')
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


def segment_steps(text: str, tokenizer, target_min: int = 80, target_max: int = 560) -> list[str]:
    if not text or not text.strip():
        return []
    # Strip final answer markers that some baselines emit at end
    text = re.sub(r"(?i)\b(final answer|the answer is)\s*:?.*$", "", text, flags=re.DOTALL).strip()
    if not text:
        return []
    # Split by double newline, drop empty
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if not chunks:
        chunks = [text]
    # Merge small adjacent segments
    merged = []
    buf = ""
    for c in chunks:
        candidate = (buf + "\n\n" + c).strip() if buf else c
        tlen = len(tokenizer.encode(candidate, add_special_tokens=False))
        if tlen < target_min:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = c
    if buf:
        merged.append(buf)
    # Split oversized segments at sentence boundary
    final = []
    for seg in merged:
        seglen = len(tokenizer.encode(seg, add_special_tokens=False))
        if seglen <= target_max:
            final.append(seg)
            continue
        # split at sentence boundary
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", seg)
        cur = ""
        for p in parts:
            cand = (cur + " " + p).strip() if cur else p
            if len(tokenizer.encode(cand, add_special_tokens=False)) > target_max and cur:
                final.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            final.append(cur)
    return [s for s in final if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="baseline_*.jsonl with raw_response")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_min", type=int, default=80)
    ap.add_argument("--target_max", type=int, default=560)
    ap.add_argument("--tokenizer", default=str(BASE / "models/qwen3.5-4b"),
                    help="tokenizer path for token counting (Qwen 4B by default)")
    args = ap.parse_args()

    if not Path(args.tokenizer).exists():
        # Fallback to chart_r1 tokenizer
        for cand in [BASE / "models/chart_r1_7b", BASE / "models/qwen3.6-27b"]:
            if cand.exists():
                args.tokenizer = str(cand)
                break
    print(f"[segment_baseline] tokenizer={args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    records = [json.loads(l) for l in open(args.input) if l.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            raw = r.get("raw_response", "") or ""
            if raw.startswith("__ERROR__") or raw.startswith("__IMG_ERROR__"):
                steps = []
            else:
                steps = segment_steps(raw, tok, args.target_min, args.target_max)
            out = dict(r)
            out["steps"] = steps
            out["n_steps"] = len(steps)
            # Preserve content/reasoning_content for outcome extraction
            if "content" not in out:
                out["content"] = ""
            if "reasoning_content" not in out:
                out["reasoning_content"] = raw  # whole raw_response as reasoning
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    # Stats
    all_recs = [json.loads(l) for l in open(out_path) if l.strip()]
    step_counts = [r["n_steps"] for r in all_recs]
    nonzero = [n for n in step_counts if n > 0]
    print(f"[segment_baseline] N={len(all_recs)}  median_steps={int(np.median(nonzero)) if nonzero else 0}  "
          f"empty_step_samples={sum(1 for n in step_counts if n==0)}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
