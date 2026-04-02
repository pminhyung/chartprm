"""
Reasoning Reward Evaluation — OpenAI SDK + vLLM server.

Runs inference on Qwen3-VL-8B-Thinking with enable_thinking=True,
parses reasoning_content, and applies reward scoring pipeline.

Usage:
  # Start vLLM server first:
  CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
      --model models/qwen3vl-8b-thinking --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.85 --max-model-len 8192 \
      --trust-remote-code --port 9300 --enable-reasoning \
      --reasoning-parser deepseek_r1

  # Then run eval:
  python scripts/reasoning_reward_eval.py \
      --server_url http://localhost:9300/v1 \
      --model_id models/qwen3vl-8b-thinking \
      --benchmark chartqa_human \
      --max_samples 50 \
      --output results/v4/reasoning_reward_eval.json
"""
import asyncio
import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, asdict

from PIL import Image
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.rewards.reasoning_chain import (
    compute_process_reward as compute_process_reward_v5,
    split_reasoning,
    extract_chart_numbers,
)
from code.rewards.llm_verifier import (
    VerifierOutput, ValueReading, Computation, score_sentence,
    split_reasoning as split_v6, best_table_match,
    ScoredSentence,
)

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# System prompt
SYSTEM_PROMPT = (
    "You are an expert chart analyst. "
    "After your reasoning, you MUST put your final answer inside <answer> and </answer> tags. "
    "The <answer> tag should contain ONLY the core numeric value or keyword — "
    "no sentences, no units unless required, no explanation. "
    "Examples: <answer>42</answer> or <answer>Yes</answer> or <answer>2018</answer>"
)


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════

BENCHMARKS = {
    "chartqa_human": {
        "data": os.path.join(BASE, "data/chartqa/test/test_human.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_augmented": {
        "data": os.path.join(BASE, "data/chartqa/test/test_augmented.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
}


def load_benchmark(name: str):
    config = BENCHMARKS[name]
    with open(config["data"]) as f:
        data = json.load(f)
    km = config["key_mapping"]
    samples = []
    for item in data:
        q = item.get(km["question"])
        a = str(item.get(km["answer"]))
        img = item.get(km["image"])
        img_path = os.path.join(config["images"], img)
        if os.path.exists(img_path):
            # CSV lookup for reward scoring
            base_name = os.path.splitext(img)[0]
            csv_candidates = [
                os.path.join(BASE, "data/chartqa/train/tables", f"{base_name}.csv"),
                os.path.join(BASE, "data/chartqa/test/tables", f"{base_name}.csv"),
            ]
            csv_path = next((p for p in csv_candidates if os.path.exists(p)), "")
            samples.append({
                "question": q, "answer": a,
                "image_path": img_path, "csv_path": csv_path,
            })
    return samples


# ═══════════════════════════════════════════
# Answer Extraction & Matching
# ═══════════════════════════════════════════

def extract_answer(response: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'\\boxed\{(.*?)\}', response)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def relaxed_accuracy(pred: str, gold: str) -> float:
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


# ═══════════════════════════════════════════
# Inference with Reasoning
# ═══════════════════════════════════════════

async def run_inference_one(
    client: AsyncOpenAI,
    model_id: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run inference for one sample, capturing reasoning_content."""
    async with semaphore:
        try:
            # Prepare image
            img = Image.open(sample["image_path"]).convert("RGB")
            max_side = max(img.size)
            if max_side > 1024:
                scale = 1024 / max_side
                img = img.resize(
                    (int(img.size[0] * scale), int(img.size[1] * scale)),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": f"Question: {sample['question']}"},
                    ],
                },
            ]

            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )

            choice = resp.choices[0]
            content = choice.message.content or ""

            # Parse reasoning_content (Qwen3 thinking mode)
            # vLLM with --enable-reasoning puts reasoning in reasoning_content field
            reasoning_content = ""
            if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                reasoning_content = choice.message.reasoning_content
            elif '<think>' in content:
                # Fallback: extract from <think> tags
                m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                if m:
                    reasoning_content = m.group(1).strip()

            # Full response for backward compat (combine reasoning + answer)
            if reasoning_content and '<think>' not in content:
                full_response = f"<think>{reasoning_content}</think>\n{content}"
            else:
                full_response = content

            return {
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "image_path": sample["image_path"],
                "csv_path": sample["csv_path"],
                "raw_response": content,
                "reasoning_content": reasoning_content,
                "full_response": full_response,
                "error": None,
            }
        except Exception as e:
            return {
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "image_path": sample["image_path"],
                "csv_path": sample["csv_path"],
                "raw_response": "",
                "reasoning_content": "",
                "full_response": "",
                "error": str(e),
            }


async def run_inference_batch(
    client: AsyncOpenAI,
    model_id: str,
    samples: List[dict],
    max_concurrent: int = 4,
) -> List[dict]:
    """Run inference on all samples with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        run_inference_one(client, model_id, s, semaphore)
        for s in samples
    ]
    results = []
    total = len(tasks)
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Inference: {i+1}/{total}")
    return results


# ═══════════════════════════════════════════
# Reward Scoring
# ═══════════════════════════════════════════

def compute_rewards(result: dict, sigma: float = 0.10) -> dict:
    """Compute all reward components for one inference result."""
    response = result["full_response"]
    gold = result["gold_answer"]
    csv_path = result["csv_path"]
    question = result["question"]

    # 1. Answer extraction + accuracy
    pred = extract_answer(response)
    r_acc = relaxed_accuracy(pred, gold)

    # 2. Format reward
    r_fmt = 0.0
    has_reasoning = bool(result["reasoning_content"])
    if has_reasoning:
        r_fmt += 0.5
        sents = split_reasoning(response)
        if len(sents) >= 3:
            r_fmt += 0.5
    r_fmt = min(r_fmt, 1.0)

    # 3. Process reward (v5 rule-based)
    r_proc_v5 = 0.0
    sentence_analyses_v5 = []
    if csv_path and os.path.exists(csv_path):
        r_proc_v5, sentence_analyses_v5 = compute_process_reward_v5(response, csv_path, sigma)

    # 4. Composite reward
    w_acc, w_proc, w_fmt = 0.5, 0.3, 0.2
    total = w_acc * r_acc + w_proc * r_proc_v5 + w_fmt * r_fmt

    # 5. Sentence-level summary
    label_counts = {}
    for a in sentence_analyses_v5:
        label_counts[a.label] = label_counts.get(a.label, 0) + 1

    return {
        "predicted_answer": pred,
        "r_accuracy": r_acc,
        "r_process_v5": r_proc_v5,
        "r_format": r_fmt,
        "total_reward": total,
        "num_sentences": len(sentence_analyses_v5),
        "label_counts": label_counts,
        "has_csv": bool(csv_path and os.path.exists(csv_path)),
        "reasoning_length": len(result["reasoning_content"]),
    }


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

async def main_async(args):
    # Load data
    samples = load_benchmark(args.benchmark)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.benchmark}")
    print(f"  With CSV: {sum(1 for s in samples if s['csv_path'])}")

    # Setup client
    client = AsyncOpenAI(base_url=args.server_url, api_key="dummy")

    # Run inference
    print(f"\nRunning inference (model={args.model_id}, concurrency={args.concurrency})...")
    t0 = time.time()
    inference_results = await run_inference_batch(
        client, args.model_id, samples, max_concurrent=args.concurrency,
    )
    t_inference = time.time() - t0
    print(f"  Inference done in {t_inference:.1f}s ({len(inference_results)/t_inference:.1f} samples/s)")

    # Check errors
    errors = [r for r in inference_results if r["error"]]
    print(f"  Errors: {len(errors)}/{len(inference_results)}")
    for e in errors[:3]:
        print(f"    {e['question'][:50]}: {e['error'][:100]}")

    # Compute rewards
    print("\nComputing rewards...")
    all_results = []
    for r in inference_results:
        if r["error"]:
            all_results.append({
                **r,
                "rewards": {"predicted_answer": "", "r_accuracy": 0, "r_process_v5": 0,
                            "r_format": 0, "total_reward": 0, "error": True},
            })
            continue
        rewards = compute_rewards(r, sigma=args.sigma)
        all_results.append({**r, "rewards": rewards})

    # Summary statistics
    valid = [r for r in all_results if not r.get("error")]
    if valid:
        avg_acc = sum(r["rewards"]["r_accuracy"] for r in valid) / len(valid)
        avg_proc = sum(r["rewards"]["r_process_v5"] for r in valid) / len(valid)
        avg_fmt = sum(r["rewards"]["r_format"] for r in valid) / len(valid)
        avg_total = sum(r["rewards"]["total_reward"] for r in valid) / len(valid)
        has_reasoning = sum(1 for r in valid if r["rewards"]["reasoning_length"] > 0)
        has_csv = sum(1 for r in valid if r["rewards"]["has_csv"])

        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY ({args.benchmark})")
        print(f"{'='*60}")
        print(f"  Samples: {len(valid)} (errors: {len(errors)})")
        print(f"  With reasoning: {has_reasoning}/{len(valid)}")
        print(f"  With CSV: {has_csv}/{len(valid)}")
        print(f"  R_accuracy:  {avg_acc:.4f} (correct: {sum(1 for r in valid if r['rewards']['r_accuracy'] > 0)}/{len(valid)})")
        print(f"  R_process:   {avg_proc:.4f}")
        print(f"  R_format:    {avg_fmt:.4f}")
        print(f"  R_total:     {avg_total:.4f}")

        # Label distribution
        all_labels = {}
        for r in valid:
            for label, count in r["rewards"].get("label_counts", {}).items():
                all_labels[label] = all_labels.get(label, 0) + count
        if all_labels:
            print(f"\n  Sentence labels:")
            for label, count in sorted(all_labels.items(), key=lambda x: -x[1]):
                print(f"    {label}: {count}")

    # Save results
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Remove non-serializable fields
    for r in all_results:
        if "rewards" in r and "label_counts" in r["rewards"]:
            pass  # already serializable

    with open(output_path, "w") as f:
        json.dump({
            "model": args.model_id,
            "benchmark": args.benchmark,
            "n_samples": len(all_results),
            "n_errors": len(errors),
            "avg_accuracy": avg_acc if valid else 0,
            "avg_process_reward": avg_proc if valid else 0,
            "avg_total_reward": avg_total if valid else 0,
            "inference_time_s": t_inference,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reasoning Reward Evaluation")
    parser.add_argument("--server_url", default="http://localhost:9300/v1")
    parser.add_argument("--model_id", default="models/qwen3vl-8b-thinking")
    parser.add_argument("--benchmark", default="chartqa_human",
                        choices=list(BENCHMARKS.keys()))
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--output", default="results/v4/reasoning_reward_eval.json")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
