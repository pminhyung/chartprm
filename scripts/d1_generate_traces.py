"""D1 trace generation — Qwen3.5-VL-4B thinking-on, multi-host vLLM round-robin.

Pattern: eval_multi_server.py (AsyncOpenAI × N hosts + Semaphore(N×5) +
round-robin + asyncio.Lock for thread-safe JSONL append + resume by sample id).

Input:  data/d1_pilot/samples.jsonl (200 samples)
Output: data/d1_pilot/traces.jsonl  (append-mode, resume-safe)

Usage:
  python scripts/d1_generate_traces.py \
      --ports 8000,8001,8002,8003,8004,8005,8006,8007,8008,8009,8010,8011,8012,8013 \
      --model Qwen3.5-VL-4B
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

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL = BASE / "data/d1_pilot/samples.jsonl"
OUT_JSONL = BASE / "data/d1_pilot/traces.jsonl"

# D1 sampling spec (user-locked 2026-05-14, single-source-of-truth for all 200 samples).
# Rationale: thinking_general + repetition_penalty=1.1 to suppress thinking-loop runaway
# observed on ReachQA. NEVER mix sampling within one dataset — see
# memory/feedback_no_mixed_sampling.md.
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.1,
    "max_tokens": 8192,
}

SYSTEM_PROMPT = (
    "You are a chart reasoning assistant. Look at the chart carefully and "
    "think step by step before giving your final answer."
)


def load_samples() -> list[dict]:
    with open(IN_JSONL) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_done_ids(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def generate_one(
    sample: dict,
    client: AsyncOpenAI,
    model: str,
    sampling: dict,
    sem: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    out_path: Path,
) -> dict:
    async with sem:
        try:
            img_b64 = image_to_b64(sample["image_path"])
        except Exception as e:
            result = {**sample, "content": "", "reasoning_content": "", "error": f"IMG_ERR: {e}"}
            async with file_lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"Question: {sample['question']}"},
            ]},
        ]
        extra = {"chat_template_kwargs": {"enable_thinking": True}}
        for k in ("top_k", "min_p", "repetition_penalty"):
            if k in sampling:
                extra[k] = sampling[k]
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                presence_penalty=sampling["presence_penalty"],
                max_tokens=sampling["max_tokens"],
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

    result = {
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
    }
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def run(ports: list[int], model: str, concurrent_per_host: int = 5) -> int:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    done = load_done_ids(OUT_JSONL)
    todo = [s for s in samples if s["id"] not in done]
    print(f"[d1-traces] total={len(samples)} done={len(done)} todo={len(todo)}")
    if not todo:
        print("[d1-traces] all complete; nothing to do.")
        return 0

    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0) for p in ports]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    file_lock = asyncio.Lock()

    sampling = SAMPLING
    print(f"[d1-traces] ports={ports}")
    print(f"[d1-traces] sampling={sampling}")

    async def worker(i: int, s: dict):
        c = clients[i % len(clients)]
        return await generate_one(s, c, model, sampling, sem, file_lock, OUT_JSONL)

    tasks = [worker(i, s) for i, s in enumerate(todo)]
    done_count = 0
    err_count = 0
    for f in asyncio.as_completed(tasks):
        r = await f
        done_count += 1
        if r.get("error"):
            err_count += 1
        if done_count % 10 == 0 or done_count == len(todo):
            print(f"  progress: {done_count}/{len(todo)} (errors so far: {err_count})", flush=True)
    print(f"[d1-traces] complete. errors={err_count}/{len(todo)} written to {OUT_JSONL}")
    return 0 if err_count == 0 else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True,
                    help="Comma-separated vLLM ports (e.g. 8000,8001,...)")
    ap.add_argument("--model", default="Qwen3.5-VL-4B",
                    help="--served-model-name from vLLM launch")
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    return asyncio.run(run(ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
