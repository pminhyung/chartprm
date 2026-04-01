"""
Within-group reward variance simulation.

For N prompts, generate 16 completions each (temperature=0.7),
compute conditional reward, measure within-group std.

Usage:
  # Start vLLM servers on GPUs 2-11 (TP=1) first, then:
  python scripts/simulate_reward_variance.py --ports 8002,...,8011 --n_prompts 20
"""
import asyncio
import argparse
import base64
import io
import json
import os
import random
import re
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


def cerm_accuracy(pred: str, gold: str) -> float:
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 / (1.0 + abs(pf - gf) / abs(gf))
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


def extract_answer_v2(response: str) -> str:
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        nums = re.findall(r'[-+]?\d*\.?\d+', after)
        if nums:
            return nums[-1]
    nums = re.findall(r'[-+]?\d*\.?\d+', response)
    return nums[-1] if nums else response.strip().split('\n')[-1]


async def generate_multiple(sample, server_urls, model_id, n_gen=16):
    """Generate n_gen completions for one prompt."""
    clients = [AsyncOpenAI(base_url=url, api_key="dummy") for url in server_urls]
    sem = asyncio.Semaphore(len(server_urls) * 4)

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

    async def gen_one(idx):
        client = clients[idx % len(clients)]
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                raw = msg.model_dump() if hasattr(msg, 'model_dump') else {}
                reasoning = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
                if reasoning:
                    return f"<think>{reasoning}</think>\n{content}"
                return content
            except Exception as e:
                return f"ERROR: {e}"

    tasks = [gen_one(i) for i in range(n_gen)]
    return await asyncio.gather(*tasks)


def compute_conditional_reward(response, gold, csv_path):
    """Conditional: correct->1.0, wrong->R_proc."""
    pred = extract_answer_v2(response)
    r_acc = cerm_accuracy(pred, str(gold))
    if r_acc >= 0.95:
        return 1.0
    return compute_process_reward_fast(response, csv_path)


async def main_async(args):
    random.seed(args.seed)

    with open(DATA_PATH) as f:
        data = [json.loads(line) for line in f]
    samples = random.sample(data, min(args.n_prompts, len(data)))

    ports = [int(p) for p in args.ports.split(",")]
    server_urls = [f"http://localhost:{p}/v1" for p in ports]
    print(f"Using {len(server_urls)} servers, {args.n_prompts} prompts x {args.n_gen} generations")

    all_stds = []
    all_outcome_stds = []

    for i, sample in enumerate(samples):
        responses = await generate_multiple(sample, server_urls, args.model_id, args.n_gen)

        # Conditional rewards
        rewards = [compute_conditional_reward(r, sample["answer"], sample["csv_path"]) for r in responses]
        std = np.std(rewards)
        all_stds.append(std)

        # Outcome-only rewards for comparison
        outcome_rewards = [cerm_accuracy(extract_answer_v2(r), str(sample["answer"])) for r in responses]
        outcome_std = np.std(outcome_rewards)
        all_outcome_stds.append(outcome_std)

        n_correct = sum(1 for r in outcome_rewards if r >= 0.95)
        print(f"  [{i+1}/{len(samples)}] correct={n_correct}/{args.n_gen}, "
              f"cond_std={std:.3f}, outcome_std={outcome_std:.3f}, "
              f"rewards={[f'{r:.2f}' for r in rewards[:5]]}...")

    print(f"\n{'='*60}")
    print("Within-Group Variance Results")
    print(f"{'='*60}")
    print(f"Conditional reward std:  mean={np.mean(all_stds):.3f}, median={np.median(all_stds):.3f}")
    print(f"Outcome-only std:        mean={np.mean(all_outcome_stds):.3f}, median={np.median(all_outcome_stds):.3f}")
    print(f"Groups with cond_std>0.1:  {sum(1 for s in all_stds if s > 0.1)}/{len(all_stds)}")
    print(f"Groups with cond_std=0:    {sum(1 for s in all_stds if s < 0.01)}/{len(all_stds)}")

    if np.mean(all_stds) > 0.1:
        print("\n✅ Conditional reward provides sufficient within-group variance for GRPO.")
    else:
        print("\n⚠️ Low variance. Consider using API verifier or adjusting reward design.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", default="8002,8003,8004,8005,8006,8007,8008,8009,8010,8011")
    parser.add_argument("--model_id", default="/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b")
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--n_gen", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
