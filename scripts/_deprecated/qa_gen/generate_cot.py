#!/usr/bin/env python
"""CoT reasoning generation for v9 SFT data.

Three modes:
  rule              — Convert reasoning_steps list to prose CoT (instant, $0)
  rejection_sampling — Generate N candidates with Qwen3.5-4B, keep correct one (GPU, $0)
  api_reverse       — On-prem 397B generates reasoning given image+Q+A ($0)

Follows CLAUDE.md eval principles:
  - All available GPU servers used (TP=1 per GPU)
  - Async parallel (max 5 concurrent per server, round-robin)
  - Sample-level append + resume support
  - as_completed for max throughput (no batch straggler blocking)

Usage:
  # Mode 1: rule-based CoT (instant)
  python scripts/generate_cot.py --mode rule \
      --input data/charts_v9/rule_qa_block_a.jsonl data/charts_v9/rule_qa_block_b.jsonl \
      --output data/charts_v9/rule_qa_with_cot.jsonl

  # Mode 2: rejection sampling for ChartQA (multi-server)
  python scripts/generate_cot.py --mode rejection_sampling \
      --chartqa_dir data/chartqa/train/ \
      --output data/charts_v9/chartqa_with_cot.jsonl \
      --hosts "http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1" \
      --n_candidates 8

  # Mode 3: API reverse CoT for failures (on-prem 397B)
  python scripts/generate_cot.py --mode api_reverse \
      --input data/charts_v9/chartqa_cot_failures.jsonl \
      --output data/charts_v9/chartqa_cot_api.jsonl \
      --hosts "http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1"
"""
import argparse
import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.extraction import extract_answer, relaxed_accuracy
from chartvr.prompts import SYSTEM_PROMPT, QA_COT_REVERSE_PROMPT


# ═══════════════════════════════════════════
# Mode 1: Rule-based CoT
# ═══════════════════════════════════════════

def steps_to_cot(reasoning_steps, answer):
    """Convert reasoning_steps list to prose CoT text."""
    if isinstance(reasoning_steps, str):
        return reasoning_steps
    if not reasoning_steps:
        return f"Let me analyze the chart.\nThe answer is {answer}."
    lines = []
    for i, step in enumerate(reasoning_steps, 1):
        lines.append(f"Step {i}: {step}")
    lines.append(f"The answer is {answer}.")
    return "\n".join(lines)


def run_rule_mode(input_paths, output_path):
    """Convert reasoning_steps to reasoning field for all records."""
    done_slugs = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_slugs.add(f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:50]}")
                except Exception:
                    pass

    all_records = []
    for p in input_paths:
        if not os.path.exists(p):
            print(f"  Skipping {p} (not found)")
            continue
        with open(p) as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(all_records)} records, {len(done_slugs)} already done")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_f = open(output_path, "a")
    added = 0

    for rec in tqdm(all_records, desc="Rule CoT"):
        key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:50]}"
        if key in done_slugs:
            continue

        reasoning_steps = rec.get("reasoning_steps", rec.get("reasoning", []))
        answer = str(rec.get("answer", ""))
        reasoning = steps_to_cot(reasoning_steps, answer)

        rec["reasoning"] = reasoning
        out_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        added += 1
        done_slugs.add(key)

    out_f.close()
    print(f"Added reasoning to {added} records. Total: {len(done_slugs)}")


# ═══════════════════════════════════════════
# Shared: image encoding
# ═══════════════════════════════════════════

def encode_image_base64(image_path, max_side=1024):
    """Encode image as base64 data URL."""
    import io
    img = Image.open(image_path).convert("RGB")
    max_dim = max(img.size)
    if max_dim > max_side:
        scale = max_side / max_dim
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ═══════════════════════════════════════════
# Mode 2: Rejection Sampling
#   - N candidates fired in PARALLEL per sample
#   - as_completed for global throughput
#   - round-robin across servers
#   - sample-level append + resume
# ═══════════════════════════════════════════

async def rejection_sample_one(clients, sample, n_candidates, semaphore, counter):
    """Generate n candidates IN PARALLEL, return reasoning from first correct one.

    All n_candidates are fired concurrently (each going to a round-robin server).
    First correct one wins; remaining are discarded.
    """
    question = sample["question"]
    gold = str(sample["answer"])
    image_path = sample.get("image_path", "")

    if not image_path or not os.path.exists(image_path):
        return None

    try:
        img_b64 = await asyncio.to_thread(encode_image_base64, image_path)
    except Exception:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Question: {question}"},
        ]},
    ]

    async def _one_attempt(client):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model="default",
                    messages=messages,
                    # max_tokens omitted: vLLM auto = max_model_len - prompt_tokens
                    temperature=0.7,
                    extra_body={"enable_thinking": True},
                )
                content = resp.choices[0].message.content or ""
                extracted = extract_answer(content)

                if relaxed_accuracy(extracted, gold) >= 0.95:
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if think_match:
                        reasoning = think_match.group(1).strip()
                        word_count = len(reasoning.split())
                        if 30 <= word_count <= 1500:
                            return reasoning
            except Exception:
                pass
        return None

    # Fire all n_candidates in parallel, round-robin across servers
    tasks = []
    for i in range(n_candidates):
        c_idx = (counter + i) % len(clients)
        tasks.append(asyncio.create_task(_one_attempt(clients[c_idx])))

    # Return first successful result, cancel remaining
    try:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                return result
        return None
    finally:
        for t in tasks:
            t.cancel()


async def run_rejection_sampling(chartqa_dir, output_path, hosts, n_candidates=8, concurrency=5):
    """Run rejection sampling on ChartQA train data.

    Architecture (CLAUDE.md compliant):
      - Multiple vLLM servers (1 per GPU, TP=1)
      - Semaphore = concurrency × n_servers (e.g., 5 × 10 = 50 parallel requests)
      - as_completed: no batch blocking, continuous throughput
      - Append-mode JSONL: resume on crash
    """
    from openai import AsyncOpenAI

    # Load ChartQA data
    samples = []
    for fname in ["train_human.json", "train_augmented.json"]:
        fpath = os.path.join(chartqa_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        for item in data:
            img_name = item.get("imgname", "")
            img_path = os.path.join(chartqa_dir, "png", img_name)
            if not os.path.exists(img_path):
                continue
            samples.append({
                "question": item.get("query", ""),
                "answer": str(item.get("label", "")),
                "image_path": img_path,
                "source": "chartqa_train",
                "chart_slug": f"cqa_{img_name.replace('.png', '')}",
            })

    print(f"Loaded {len(samples)} ChartQA samples")

    # Resume: check both success and failure files
    done_slugs = set()
    fail_path = output_path.replace(".jsonl", "_failures.jsonl")
    for fpath in [output_path, fail_path]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    try:
                        done_slugs.add(json.loads(line).get("chart_slug", ""))
                    except Exception:
                        pass

    pending = [s for s in samples if s["chart_slug"] not in done_slugs]
    print(f"  Done: {len(done_slugs)}, Pending: {len(pending)}")

    if not pending:
        print("  All done!")
        return

    # Create async clients — one per host (each host = one GPU with vLLM TP=1)
    host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    clients = [AsyncOpenAI(base_url=h, api_key="EMPTY") for h in host_list]
    n_servers = len(clients)

    # Request-level semaphore: max concurrent HTTP requests across ALL servers
    total_concurrent = concurrency * n_servers
    semaphore = asyncio.Semaphore(total_concurrent)
    # Sample-level semaphore: limit active samples to prevent memory explosion
    # Each sample holds an image in memory (~0.5-1MB), so cap active samples
    max_active_samples = total_concurrent * 2
    sample_sem = asyncio.Semaphore(max_active_samples)
    print(f"  Servers: {n_servers}, Concurrent requests: {total_concurrent}, Max active samples: {max_active_samples}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_f = open(output_path, "a")
    fail_f = open(fail_path, "a")
    write_lock = asyncio.Lock()

    success = 0
    failed = 0
    pbar = tqdm(total=len(pending), desc="Rejection sampling")

    async def process_one(sample, idx):
        nonlocal success, failed
        async with sample_sem:
            reasoning = await rejection_sample_one(
                clients, sample, n_candidates, semaphore, counter=idx
            )

        async with write_lock:
            if reasoning:
                sample["reasoning"] = reasoning
                out_f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
                out_f.flush()
                success += 1
            else:
                fail_f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
                fail_f.flush()
                failed += 1
        pbar.update(1)

    # Launch ALL samples as tasks, let semaphores manage throughput + memory
    all_tasks = [process_one(s, i) for i, s in enumerate(pending)]
    try:
        for coro in asyncio.as_completed(all_tasks):
            await coro
    finally:
        out_f.close()
        fail_f.close()
        pbar.close()

    print(f"\nSuccess: {success}, Failed: {failed}")
    if failed > 0:
        print(f"Failures saved to {fail_path}")


# ═══════════════════════════════════════════
# Mode 3: API Reverse CoT
#   - Same architecture: as_completed + round-robin + append
# ═══════════════════════════════════════════

async def reverse_cot_one(client, sample, semaphore):
    """Generate reverse CoT: given Q+A, produce reasoning."""
    question = sample["question"]
    answer = str(sample["answer"])
    image_path = sample.get("image_path", "")
    csv_path = sample.get("csv_path", "")

    prompt = QA_COT_REVERSE_PROMPT.format(question=question, answer=answer)

    if csv_path and os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            csv_text = df.head(30).to_csv(index=False)
            prompt += f"\n\nData table:\n{csv_text}"
        except Exception:
            pass

    messages = [{"role": "user", "content": prompt}]

    if image_path and os.path.exists(image_path):
        try:
            img_b64 = await asyncio.to_thread(encode_image_base64, image_path)
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}]
        except Exception:
            pass

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model="default",
                messages=messages,
                # max_tokens omitted: vLLM auto = max_model_len - prompt_tokens
                temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = resp.choices[0].message.content or ""
            content = re.sub(r'<think>|</think>|<answer>.*?</answer>', '', content).strip()
            word_count = len(content.split())
            if 20 <= word_count <= 1500:
                return content
        except Exception:
            pass

    return None


async def run_api_reverse(input_paths, output_path, hosts, concurrency=5):
    """Run API reverse CoT. as_completed + round-robin + append."""
    from openai import AsyncOpenAI

    all_records = []
    for p in input_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

    # Resume
    done_slugs = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done_slugs.add(json.loads(line).get("chart_slug", ""))
                except Exception:
                    pass

    pending = [r for r in all_records if r.get("chart_slug", "") not in done_slugs]
    print(f"Loaded {len(all_records)}, pending: {len(pending)}")

    if not pending:
        print("All done!")
        return

    host_list = [h.strip() for h in hosts.split(",") if h.strip()]
    clients = [AsyncOpenAI(base_url=h, api_key="EMPTY") for h in host_list]
    n_servers = len(clients)
    total_concurrent = concurrency * n_servers
    semaphore = asyncio.Semaphore(total_concurrent)
    sample_sem = asyncio.Semaphore(total_concurrent * 2)
    print(f"  Servers: {n_servers}, Concurrent: {total_concurrent}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_f = open(output_path, "a")
    fail_path = output_path.replace(".jsonl", "_failures.jsonl")
    fail_f = open(fail_path, "a")
    write_lock = asyncio.Lock()
    success = 0
    failed = 0
    pbar = tqdm(total=len(pending), desc="API reverse CoT")

    async def process_one(sample, idx):
        nonlocal success, failed
        async with sample_sem:
            client = clients[idx % n_servers]
            reasoning = await reverse_cot_one(client, sample, semaphore)
        async with write_lock:
            if reasoning:
                sample["reasoning"] = reasoning
                out_f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
                out_f.flush()
                success += 1
            else:
                fail_f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
                fail_f.flush()
                failed += 1
        pbar.update(1)

    all_tasks = [process_one(s, i) for i, s in enumerate(pending)]
    try:
        for coro in asyncio.as_completed(all_tasks):
            await coro
    finally:
        pbar.close()
        out_f.close()
        fail_f.close()

    print(f"\nSuccess: {success}, Failed: {failed}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CoT reasoning generation v9")
    parser.add_argument("--mode", choices=["rule", "rejection_sampling", "api_reverse"], required=True)
    parser.add_argument("--input", nargs="*", help="Input JSONL files (for rule/api_reverse)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--chartqa_dir", help="ChartQA train directory (for rejection_sampling)")
    parser.add_argument("--hosts", help="Comma-separated vLLM host URLs")
    parser.add_argument("--n_candidates", type=int, default=8, help="Candidates per sample (rejection_sampling)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests per host")
    args = parser.parse_args()

    if args.mode == "rule":
        if not args.input:
            parser.error("--input required for rule mode")
        run_rule_mode(args.input, args.output)

    elif args.mode == "rejection_sampling":
        if not args.chartqa_dir:
            parser.error("--chartqa_dir required for rejection_sampling mode")
        if not args.hosts:
            parser.error("--hosts required for rejection_sampling mode")
        asyncio.run(run_rejection_sampling(
            args.chartqa_dir, args.output, args.hosts,
            args.n_candidates, args.concurrency
        ))

    elif args.mode == "api_reverse":
        if not args.input:
            parser.error("--input required for api_reverse mode")
        if not args.hosts:
            parser.error("--hosts required for api_reverse mode")
        asyncio.run(run_api_reverse(args.input, args.output, args.hosts, args.concurrency))


if __name__ == "__main__":
    main()
