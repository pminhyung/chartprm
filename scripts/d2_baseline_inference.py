"""D2 baseline inference — generate reasoning traces from external baselines.

Targets: zero-shot Qwen3.5-VL-4B (reused from existing pilot), Chart-R1 7B, ChartGemma 12B.
Same 100-sample d2_pilot subset.

Multi-host async, sample-level append, resume-by-id.

Input:  data/d1_pilot/segmented_v3.jsonl (use subset selected to match d2_pilot/samples.jsonl)
Output: data/d2_pilot/baseline_{name}.jsonl  with raw_response (str) per sample
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


PROMPT_TEMPLATE = """Looking at this chart, answer the following question step by step. Show your reasoning, then provide a clear final answer.

Question: {question}

Reasoning:"""


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def infer_one(client: AsyncOpenAI, model: str, rec: dict, sem: asyncio.Semaphore,
                    max_tokens: int = 4096, temperature: float = 0.0) -> dict:
    out = dict(rec)
    out["baseline_model"] = model
    try:
        img_b64 = image_to_b64(rec["image_path"])
    except Exception as e:
        out["raw_response"] = f"__IMG_ERROR__: {type(e).__name__}: {str(e)[:160]}"
        return out
    prompt = PROMPT_TEMPLATE.format(question=rec["question"])
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out["raw_response"] = (resp.choices[0].message.content or "")
        except Exception as e:
            out["raw_response"] = f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"
    return out


async def process_and_save(rec: dict, client_picker, model: str, sem: asyncio.Semaphore,
                           file_lock: asyncio.Lock, out_path: Path,
                           max_tokens: int, temperature: float):
    res = await infer_one(client_picker(), model, rec, sem, max_tokens=max_tokens,
                          temperature=temperature)
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


async def run(input_path: Path, output_path: Path, ports: list[int], model: str,
              concurrent_per_host: int, max_tokens: int, temperature: float,
              ids_filter: set[str] | None):
    records = [json.loads(l) for l in open(input_path) if l.strip()]
    if ids_filter is not None:
        records = [r for r in records if r["id"] in ids_filter]
    done = load_done(output_path)
    todo = [r for r in records if r["id"] not in done]
    print(f"[baseline_inference] model={model}  total={len(records)} done={len(done)} todo={len(todo)}")
    if not todo:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0)
               for p in ports]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    file_lock = asyncio.Lock()
    cnt = {"i": 0}
    def pick():
        c = clients[cnt["i"] % len(clients)]
        cnt["i"] += 1
        return c

    print(f"[baseline_inference] hosts={len(clients)} conc/host={concurrent_per_host} max_tokens={max_tokens} T={temperature}")
    tasks = [process_and_save(r, pick, model, sem, file_lock, output_path, max_tokens, temperature)
             for r in todo]
    n_done = 0
    for f in asyncio.as_completed(tasks):
        await f
        n_done += 1
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)}", flush=True)
    print(f"[baseline_inference] -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(BASE / "data/d1_pilot/segmented_v3.jsonl"),
                    help="source samples (must have id, image_path, question, gold_answer)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ports", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ids_from", default=None,
                    help="optional jsonl whose 'id' fields constrain inference set")
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    ids_filter = None
    if args.ids_from:
        ids_filter = set()
        for l in open(args.ids_from):
            if l.strip():
                try:
                    ids_filter.add(json.loads(l)["id"])
                except Exception:
                    pass
        print(f"[baseline_inference] ids_filter loaded: {len(ids_filter)}")
    asyncio.run(run(Path(args.input), Path(args.output), ports, args.model,
                    args.concurrent_per_host, args.max_tokens, args.temperature, ids_filter))


if __name__ == "__main__":
    sys.exit(main() or 0)
