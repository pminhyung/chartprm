"""
Day 1 Verifier Validation: Test rule-based verifier on 50 samples.

1. Sample 50 from chartvr_train_final.jsonl (proportional by source)
2. Generate zero-shot reasoning with Qwen3.5-4B via vLLM
3. Run compute_process_reward_fast() on each
4. Analyze distributions and compare with gold reasoning_steps

Usage:
  # First start vLLM servers on GPUs 2-11 (TP=1, 10 servers):
  for i in 2 3 4 5 6 7 8 9 10 11; do
      port=$((8000+i))
      CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
          --model /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
          --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
          --max-model-len 8192 --port $port --trust-remote-code &
  done

  # Then run:
  python scripts/validate_rule_verifier.py --ports 8002,8003,...,8011
"""
import asyncio
import argparse
import base64
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image
from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.rewards.rule_verifier_fast import compute_process_reward_fast

DATA_PATH = "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/chartvr_train_final.jsonl"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def sample_proportional(data, n=50):
    """Sample n items proportionally by source."""
    from collections import Counter
    sources = Counter(d["source"] for d in data)
    total = sum(sources.values())
    samples = []
    for source, count in sources.items():
        k = max(1, round(n * count / total))
        pool = [d for d in data if d["source"] == source]
        samples.extend(random.sample(pool, min(k, len(pool))))
    random.shuffle(samples)
    return samples[:n]


async def generate_reasoning(samples, server_urls, model_id):
    """Generate zero-shot reasoning for samples using vLLM servers."""
    clients = [AsyncOpenAI(base_url=url, api_key="dummy") for url in server_urls]
    sem = asyncio.Semaphore(len(server_urls) * 4)

    async def infer_one(idx, sample):
        client = clients[idx % len(clients)]
        async with sem:
            img = Image.open(sample["image_path"]).convert("RGB")
            max_side = max(img.size)
            if max_side > 1024:
                scale = 1024 / max_side
                img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": sample["question"]},
                ]}
            ]
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                raw = msg.model_dump() if hasattr(msg, 'model_dump') else {}
                reasoning = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
                if reasoning:
                    full = f"<think>{reasoning}</think>\n{content}"
                elif "<think>" in content:
                    full = content
                else:
                    full = content
                return full
            except Exception as e:
                return f"ERROR: {e}"

    tasks = [infer_one(i, s) for i, s in enumerate(samples)]
    return await asyncio.gather(*tasks)


def analyze_results(samples, responses, scores):
    """Analyze rule verifier scores."""
    print(f"\n{'='*60}")
    print("Rule-Based Verifier Validation Results")
    print(f"{'='*60}")

    print(f"\nTotal samples: {len(samples)}")
    print(f"Score distribution:")
    print(f"  Mean:   {np.mean(scores):.3f}")
    print(f"  Std:    {np.std(scores):.3f}")
    print(f"  Min:    {np.min(scores):.3f}")
    print(f"  Max:    {np.max(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")

    # Bucket analysis
    buckets = {"0.0": 0, "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in scores:
        if s == 0.0:
            buckets["0.0"] += 1
        elif s < 0.2:
            buckets["0.0-0.2"] += 1
        elif s < 0.4:
            buckets["0.2-0.4"] += 1
        elif s < 0.6:
            buckets["0.4-0.6"] += 1
        elif s < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    print(f"\nScore buckets:")
    for k, v in buckets.items():
        print(f"  {k}: {v} ({v/len(scores)*100:.0f}%)")

    # Per-source analysis
    from collections import defaultdict
    source_scores = defaultdict(list)
    for s, score in zip(samples, scores):
        source_scores[s["source"]].append(score)
    print(f"\nPer-source scores:")
    for source, ss in sorted(source_scores.items()):
        print(f"  {source}: mean={np.mean(ss):.3f}, std={np.std(ss):.3f}, n={len(ss)}")

    # Check if scores are all clustered (bad for GRPO)
    if np.std(scores) < 0.05:
        print(f"\n⚠️ WARNING: Score std={np.std(scores):.3f} is very low. All scores clustered.")
        print("   This means rule verifier can't distinguish good vs bad reasoning.")
    elif np.std(scores) >= 0.1:
        print(f"\n✅ Score std={np.std(scores):.3f} shows good discrimination.")

    # Print 3 example cases
    print(f"\n--- Example Cases ---")
    indices = [0, len(samples)//2, -1]
    for idx in indices:
        s = samples[idx]
        print(f"\n[{idx}] Source: {s['source']}, Score: {scores[idx]:.3f}")
        print(f"  Q: {s['question'][:80]}...")
        print(f"  A: {s['answer']}")
        resp = responses[idx]
        if len(resp) > 200:
            resp = resp[:200] + "..."
        print(f"  Response: {resp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="8002,8003,8004,8005,8006,8007,8008,8009,8010,8011")
    parser.add_argument("--model_id", default="/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    with open(DATA_PATH) as f:
        data = [json.loads(line) for line in f]
    samples = sample_proportional(data, args.n_samples)
    print(f"Sampled {len(samples)} from {len(data)} total")

    # Generate reasoning
    ports = [int(p) for p in args.ports.split(",")]
    server_urls = [f"http://localhost:{p}/v1" for p in ports]
    print(f"Using {len(server_urls)} servers: {server_urls[:3]}...")

    responses = asyncio.run(generate_reasoning(samples, server_urls, args.model_id))

    # Run rule verifier
    scores = []
    for s, resp in zip(samples, responses):
        if resp.startswith("ERROR"):
            scores.append(0.0)
            continue
        score = compute_process_reward_fast(resp, s["csv_path"])
        scores.append(score)

    # Analyze
    analyze_results(samples, responses, scores)

    # Save results
    out_path = "results/v7_verifier_validation.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for s, resp, score in zip(samples, responses, scores):
            f.write(json.dumps({
                "question": s["question"],
                "answer": s["answer"],
                "source": s["source"],
                "csv_path": s["csv_path"],
                "response": resp[:2000],
                "r_proc_rule": score,
            }, ensure_ascii=False) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
