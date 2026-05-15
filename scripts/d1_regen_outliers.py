"""Regenerate runaway-trace outliers with greedy + repetition_penalty.

Hypothesis: temp=0.6 default caused thinking-loop runaway on ReachQA reasoning
samples. Re-run those samples with guide-spec sampling (temp=0.0,
max_tokens=8192, repetition_penalty=1.1).

Output: data/d1_pilot/traces_greedy.jsonl (only outlier samples)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
SEG_IN = BASE / "data/d1_pilot/segmented.jsonl"
OUT_JSONL = BASE / "data/d1_pilot/traces_greedy.jsonl"

SYSTEM_PROMPT = (
    "You are a chart reasoning assistant. Look at the chart carefully and "
    "think step by step before giving your final answer."
)


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def gen_one(sample, client, model, sem, lock, out_path):
    async with sem:
        try:
            img_b64 = image_to_b64(sample["image_path"])
        except Exception as e:
            r = {**sample, "content": "", "reasoning_content": "", "error": f"IMG_ERR: {e}"}
            async with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return r

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"Question: {sample['question']}"},
            ]},
        ]
        # Guide §2.3 sampling: temp=0.0 greedy, max_tokens=8192, + user-requested rep_penalty=1.1
        extra = {
            "chat_template_kwargs": {"enable_thinking": True},
            "repetition_penalty": 1.1,
        }
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=8192,
                extra_body=extra,
            )
            content = resp.choices[0].message.content or ""
            raw = resp.choices[0].message.model_dump() if hasattr(resp.choices[0].message, "model_dump") else {}
            reasoning = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
            err = ""
        except Exception as e:
            content = ""
            reasoning = ""
            err = f"API_ERR: {type(e).__name__}: {str(e)[:200]}"

    r = {
        "id": sample["id"],
        "source": sample["source"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "qa_type": sample.get("qa_type"),
        "chart_type": sample.get("chart_type"),
        "content": content,
        "reasoning_content": reasoning,
        "error": err,
        "prev_n_steps": sample.get("prev_n_steps"),
    }
    async with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return r


async def run(ports, model, threshold):
    seg = [json.loads(l) for l in open(SEG_IN) if l.strip()]
    outliers = [r for r in seg if r["n_steps"] > threshold]
    print(f"[regen] outliers (n_steps > {threshold}): {len(outliers)}")
    print(f"[regen] sampling: temperature=0.0, max_tokens=8192, repetition_penalty=1.1")

    # Resume
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [r for r in outliers if r["id"] not in done]
    print(f"[regen] done={len(done)} todo={len(todo)}")
    if not todo:
        return 0

    # Strip seg-specific fields, keep prev_n_steps
    samples = [{
        "id": r["id"], "source": r["source"], "image_path": r["image_path"],
        "question": r["question"], "gold_answer": r["gold_answer"],
        "qa_type": r.get("qa_type"), "chart_type": r.get("chart_type"),
        "prev_n_steps": r["n_steps"],
    } for r in todo]

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy") for p in ports]
    sem = asyncio.Semaphore(len(clients) * 5)
    lock = asyncio.Lock()

    async def worker(i, s):
        return await gen_one(s, clients[i % len(clients)], model, sem, lock, OUT_JSONL)

    tasks = [worker(i, s) for i, s in enumerate(samples)]
    n_done = 0
    for f in asyncio.as_completed(tasks):
        r = await f
        n_done += 1
        if n_done % 5 == 0 or n_done == len(samples):
            print(f"  progress: {n_done}/{len(samples)}", flush=True)
    print(f"[regen] complete -> {OUT_JSONL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="8000")
    ap.add_argument("--model", default="Qwen3.5-VL-4B")
    ap.add_argument("--threshold", type=int, default=50, help="n_steps threshold for outlier")
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    return asyncio.run(run(ports, args.model, args.threshold))


if __name__ == "__main__":
    sys.exit(main() or 0)
