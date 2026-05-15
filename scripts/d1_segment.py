"""D1 segmentation — split <think>...</think> trace into reasoning steps.

Algorithm (guide §2.4):
  1. Extract content between <think>...</think>. If no tag, use reasoning_content
     (vLLM with --reasoning-parser qwen3 strips the tags and puts the thinking
     content there).
  2. Split by `\\n\\n`.
  3. Merge adjacent segments with token_count < target_min (default 80).
  4. Split segments with token_count > target_max (default 560) at sentence
     boundary `[.!?]\\s+[A-Z]`.

Input:  data/d1_pilot/traces.jsonl
Output: data/d1_pilot/segmented.jsonl + step distribution stats to stdout
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
IN_JSONL = BASE / "data/d1_pilot/traces.jsonl"
OUT_JSONL = BASE / "data/d1_pilot/segmented.jsonl"
DEFAULT_TOK_PATH = BASE / "models/qwen3.5-4b"


def extract_thinking(rec: dict) -> str:
    """Get the model's thinking text.

    With vLLM `--reasoning-parser qwen3`, the `<think>...</think>` block is
    pulled out and placed in `reasoning_content`. If absent (legacy), fall back
    to scraping from `content`.
    """
    rc = rec.get("reasoning_content") or ""
    if rc and not rc.startswith("ERROR"):
        return rc.strip()
    content = rec.get("content") or ""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def segment_steps(thinking: str, tokenizer, target_min: int = 80, target_max: int = 560) -> list[str]:
    if not thinking.strip():
        return []

    def tcount(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    raw = [s.strip() for s in thinking.split("\n\n") if s.strip()]
    if not raw:
        return []

    # Merge small segments forward into the running buffer.
    merged: list[str] = []
    buf = ""
    for seg in raw:
        cand = (buf + "\n\n" + seg).strip() if buf else seg
        if buf and tcount(cand) < target_min:
            buf = cand
        else:
            if buf:
                merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)

    # Split oversized segments at sentence boundary.
    final: list[str] = []
    sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    for seg in merged:
        if tcount(seg) <= target_max:
            final.append(seg)
            continue
        sentences = sent_split.split(seg)
        chunk = ""
        for s in sentences:
            cand = (chunk + " " + s).strip() if chunk else s
            if tcount(cand) <= target_max:
                chunk = cand
            else:
                if chunk:
                    final.append(chunk)
                chunk = s
        if chunk:
            final.append(chunk)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOK_PATH),
                    help="Tokenizer (HF path or hub id). Defaults to local Qwen3.5-VL-4B.")
    ap.add_argument("--target_min", type=int, default=80)
    ap.add_argument("--target_max", type=int, default=560)
    args = ap.parse_args()

    print(f"[segment] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    records = [json.loads(l) for l in open(IN_JSONL) if l.strip()]
    print(f"[segment] input traces: {len(records)}")

    out = []
    n_skipped = 0
    for rec in records:
        thinking = extract_thinking(rec)
        if not thinking or rec.get("error"):
            n_skipped += 1
            steps = []
            step_lens = []
        else:
            steps = segment_steps(thinking, tokenizer, args.target_min, args.target_max)
            step_lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in steps]
        rec["thinking"] = thinking
        rec["steps"] = steps
        rec["n_steps"] = len(steps)
        rec["step_token_lens"] = step_lens
        out.append(rec)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[segment] wrote {len(out)} -> {OUT_JSONL} (skipped {n_skipped} empty/error)")

    # Stats
    valid = [r for r in out if r["n_steps"] > 0]
    if not valid:
        print("[segment] no valid records to summarize")
        return
    all_n = [r["n_steps"] for r in valid]
    all_len = [l for r in valid for l in r["step_token_lens"]]
    print()
    print(f"steps/sample : median={np.median(all_n):.1f}  p25={np.percentile(all_n,25):.0f}  p75={np.percentile(all_n,75):.0f}  min={min(all_n)}  max={max(all_n)}")
    print(f"step tokens  : median={np.median(all_len):.0f}  p10={np.percentile(all_len,10):.0f}  p90={np.percentile(all_len,90):.0f}  min={min(all_len)}  max={max(all_len)}")
    print(f"valid={len(valid)}/{len(out)}  total_steps={len(all_len)}")


if __name__ == "__main__":
    main()
