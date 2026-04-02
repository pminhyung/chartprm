#!/usr/bin/env python
"""Difficulty filter: Qwen3.5-4B VLM zero-shot with image.

4 vLLM servers (GPUs 12-15), round-robin LB, 4 concurrent/server = 16 total.
Each QA gets 1 attempt. Keep QAs where model gets wrong (difficulty target).
Uses chart IMAGE for realistic difficulty measurement.

Usage:
  python scripts/run_difficulty_filter.py --preflight 50
  python scripts/run_difficulty_filter.py --bulk
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image
from openai import AsyncOpenAI
from tqdm import tqdm

BASE = "/ex_disk2/mhpark/poc/chartvr"
INPUT_PATH = os.path.join(BASE, "data/charts_v2/need_difficulty_filter.jsonl")
OUTPUT_PATH = os.path.join(BASE, "data/charts_v2/difficulty_filtered_new.jsonl")

ENDPOINTS = [
    "http://localhost:9201/v1",
    "http://localhost:9202/v1",
    "http://localhost:9203/v1",
    "http://localhost:9204/v1",
]
MODEL = "qwen35-4b"
CONCURRENCY_PER_SERVER = 4

SYSTEM_PROMPT = (
    "You are an expert chart analyst. Be concise in your reasoning. "
    "Focus on reading values and computing the answer. "
    "Do not describe the chart's visual appearance. "
    "Put your final answer inside <answer></answer> tags with ONLY the core value or keyword. "
    "Example: <answer>42</answer>"
)


def load_and_encode_image(image_path: str) -> str:
    """Load image, resize, base64 encode."""
    img = Image.open(image_path).convert("RGB")
    max_side = max(img.size)
    if max_side > 1024:
        scale = 1024 / max_side
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def relaxed_match(pred: str, gold: str) -> bool:
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) / max(abs(gf), 1e-10) <= 0.05 if gf != 0 else abs(pf) < 0.01
    except ValueError:
        return p.lower() == g.lower()


def extract_answer(content: str) -> str:
    matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


class RoundRobinLB:
    def __init__(self, endpoints, concurrency):
        self.endpoints = endpoints
        self._idx = 0
        self._sems = {ep: asyncio.Semaphore(concurrency) for ep in endpoints}
        self._clients = {ep: AsyncOpenAI(base_url=ep, api_key="dummy", timeout=120.0) for ep in endpoints}

    def next(self):
        ep = self.endpoints[self._idx % len(self.endpoints)]
        self._idx += 1
        return ep


async def attempt_one(lb: RoundRobinLB, qa: dict) -> dict:
    """One zero-shot attempt with image. Returns qa with difficulty_correct field."""
    ep = lb.next()
    async with lb._sems[ep]:
        client = lb._clients[ep]
        image_path = qa.get("image_path", "")

        if not image_path or not os.path.exists(image_path):
            return {**qa, "difficulty_correct": None, "difficulty_reason": "no_image"}

        try:
            img_b64 = load_and_encode_image(image_path)
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": f"Question: {qa['question']}"},
                    ]},
                ],
                max_tokens=4096,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            content = resp.choices[0].message.content or ""
            # Extract reasoning_content (thinking mode separates it)
            raw_dump = resp.choices[0].message.model_dump() if hasattr(resp.choices[0].message, 'model_dump') else {}
            reasoning = raw_dump.get("reasoning", "") or ""
            # Reconstruct full response for answer extraction
            if reasoning and '<think>' not in content:
                full_response = f"<think>{reasoning}</think>\n{content}"
            else:
                full_response = content
            pred = extract_answer(full_response)
            gold = str(qa.get("answer", ""))
            correct = relaxed_match(pred, gold)
            return {
                **qa,
                "difficulty_correct": correct,
                "difficulty_pred": pred,
                "difficulty_reason": "ok",
            }
        except Exception as e:
            return {**qa, "difficulty_correct": None, "difficulty_reason": f"error: {str(e)[:100]}"}


async def run_filter(qa_pairs: list, lb: RoundRobinLB, output_path: str, mode: str = "bulk"):
    """Run difficulty filter on all QAs."""
    # Resume
    done_questions = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_questions.add(rec.get("question", "")[:50])
                except:
                    pass

    pending = [qa for qa in qa_pairs if qa.get("question", "")[:50] not in done_questions]
    print(f"Total: {len(qa_pairs)}, Done: {len(done_questions)}, Pending: {len(pending)}", flush=True)

    if not pending:
        print("All done!", flush=True)
        return

    out_f = open(output_path, "a")
    stats = {"correct": 0, "wrong": 0, "error": 0, "kept": 0}

    tasks = [attempt_one(lb, qa) for qa in pending]
    pbar = tqdm(total=len(tasks), desc=f"Difficulty Filter ({mode})", unit="qa")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        pbar.update(1)

        if result.get("difficulty_correct") is None:
            stats["error"] += 1
        elif result["difficulty_correct"]:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1

        # Save all results
        out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        out_f.flush()

        # Keep = wrong answers (model can't solve → good for training)
        # Actually keep BOTH but mark. Filter later with 30-70% range across batches.
        total_done = stats["correct"] + stats["wrong"]
        if total_done > 0:
            acc = stats["correct"] / total_done
            pbar.set_postfix(acc=f"{acc:.3f}", ok=stats["correct"], wrong=stats["wrong"], err=stats["error"])

    pbar.close()
    out_f.close()

    total_done = stats["correct"] + stats["wrong"]
    acc = stats["correct"] / total_done if total_done > 0 else 0
    print(f"\nResults: acc={acc:.3f} ({stats['correct']}/{total_done}), errors={stats['error']}", flush=True)
    print(f"Filter: keep wrong answers (difficulty for 4B) for training", flush=True)


async def main_async(args):
    if not os.path.exists(INPUT_PATH):
        # Fallback to generated_qa if answer_verified_pass doesn't exist yet
        alt_path = os.path.join(BASE, "data/new_charts/generated_qa.jsonl")
        if os.path.exists(alt_path):
            print(f"Using {alt_path} (answer_verified_pass not ready yet)", flush=True)
            input_path = alt_path
        else:
            print(f"No input found at {INPUT_PATH}", flush=True)
            return
    else:
        input_path = INPUT_PATH

    with open(input_path) as f:
        qa_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(qa_pairs)} QAs", flush=True)

    lb = RoundRobinLB(ENDPOINTS, CONCURRENCY_PER_SERVER)

    if args.preflight:
        # Preflight: test with small sample
        import random
        random.seed(42)
        sample = random.sample(qa_pairs, min(args.preflight, len(qa_pairs)))
        preflight_out = os.path.join(BASE, "data/new_charts/difficulty_preflight.jsonl")
        print(f"\n=== PREFLIGHT ({len(sample)} samples) ===", flush=True)
        await run_filter(sample, lb, preflight_out, mode="preflight")
    else:
        # Bulk
        await run_filter(qa_pairs, lb, OUTPUT_PATH, mode="bulk")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=int, default=0, help="Run preflight with N samples")
    parser.add_argument("--bulk", action="store_true")
    args = parser.parse_args()

    if not args.preflight and not args.bulk:
        args.preflight = 50  # Default to preflight

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
