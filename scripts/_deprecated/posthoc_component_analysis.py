#!/usr/bin/env python
"""Post-hoc per-component reward analysis.

For checkpoints that didn't log individual R_accuracy/R_process/R_format
during training, this script runs inference on a sample and measures
each component separately.

Usage:
  python scripts/posthoc_component_analysis.py \
      --checkpoint ckpt/cvr_v6b \
      --data data/charts_v2/grpo_train_final.json \
      --n_samples 100 \
      --output results/posthoc_cvr_v6b.jsonl
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys

from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm


BASE = "/ex_disk2/mhpark/poc/chartvr"
VERIFIER_URL = "http://10.1.211.148:8000/v1"
VERIFIER_MODEL = "Qwen3.5-397B-A17B-FP8"


def relaxed_match(pred, gold):
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) / max(abs(gf), 1e-10) <= 0.05 if gf != 0 else abs(pf) < 0.01
    except ValueError:
        return p.lower() == g.lower()


def extract_answer(content):
    matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def extract_reasoning(resp):
    m = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_and_encode_image(image_path):
    img = Image.open(image_path).convert("RGB")
    max_side = max(img.size)
    if max_side > 1024:
        scale = 1024 / max_side
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def run_analysis(args):
    # Load data
    with open(args.data) as f:
        data = json.load(f)

    import random
    random.seed(42)
    samples = random.sample(data, min(args.n_samples, len(data)))
    print(f"Analyzing {len(samples)} samples from {args.data}")

    # Setup vLLM client (model checkpoint must be served)
    model_client = AsyncOpenAI(
        base_url=f"http://localhost:{args.vllm_port}/v1",
        api_key="dummy",
        timeout=120.0,
    )

    # Setup verifier client
    from code.rewards.llm_verifier import VerifierClient, csv_to_text, compute_process_reward

    verifier = VerifierClient(base_url=VERIFIER_URL, model_id=VERIFIER_MODEL)

    SYSTEM_PROMPT = (
        "You are an expert chart analyst. "
        "After your reasoning, put your final answer inside "
        "<answer></answer> tags with ONLY the core value or keyword. "
        "Example: <answer>42</answer>"
    )

    sem = asyncio.Semaphore(4)
    results = []

    async def analyze_one(sample):
        async with sem:
            img_path = sample.get("image", "")
            if not os.path.exists(img_path):
                return None

            img_b64 = load_and_encode_image(img_path)
            question = sample["problem"]
            gold = sample["solution"]
            csv_path = sample.get("csv_path", "")

            # Run inference
            try:
                resp = await model_client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": f"Question: {question}"},
                        ]},
                    ],
                    max_tokens=2048,
                    temperature=0.6,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                content = resp.choices[0].message.content or ""
            except Exception as e:
                return {"error": str(e)[:100], "question": question}

            # Extract components
            pred = extract_answer(content)
            reasoning = extract_reasoning(content)

            # R_accuracy (CERM)
            r_acc = 1.0 if relaxed_match(pred, gold) else 0.0

            # R_process (verifier)
            r_proc = 0.0
            n_vclaims = 0
            n_cclaims = 0
            if csv_path and os.path.exists(csv_path):
                try:
                    csv_text = csv_to_text(csv_path)
                    extraction, _ = await verifier.verify_single(csv_text, question, reasoning)
                    r_proc, details = compute_process_reward(extraction, reasoning)
                    n_vclaims = len(extraction.value_claims)
                    n_cclaims = len(extraction.computation_claims)
                except Exception as e:
                    r_proc = 0.0

            # R_format
            r_fmt = 0.0
            if '<think>' in content and '</think>' in content:
                r_fmt += 0.25
            if '<answer>' in content:
                r_fmt += 0.25
            lines = [l for l in reasoning.split('\n') if l.strip()] if reasoning else []
            if len(lines) >= 3:
                r_fmt += 0.5

            return {
                "question": question[:100],
                "gold": gold,
                "pred": pred,
                "r_accuracy": r_acc,
                "r_process": r_proc,
                "r_format": min(r_fmt, 1.0),
                "r_total": 0.5 * r_acc + 0.3 * r_proc + 0.2 * min(r_fmt, 1.0),
                "n_value_claims": n_vclaims,
                "n_computation_claims": n_cclaims,
                "reasoning_length": len(reasoning),
            }

    tasks = [analyze_one(s) for s in samples]
    pbar = tqdm(total=len(tasks), desc="Post-hoc Analysis")

    with open(args.output, "w") as out_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            pbar.update(1)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                results.append(result)

    pbar.close()

    # Summary
    valid = [r for r in results if "error" not in r]
    if valid:
        import numpy as np
        print(f"\n=== Post-hoc Component Analysis ({len(valid)} samples) ===")
        print(f"R_accuracy mean: {np.mean([r['r_accuracy'] for r in valid]):.4f}")
        print(f"R_process  mean: {np.mean([r['r_process'] for r in valid]):.4f}")
        print(f"R_format   mean: {np.mean([r['r_format'] for r in valid]):.4f}")
        print(f"R_total    mean: {np.mean([r['r_total'] for r in valid]):.4f}")
        print(f"Avg value claims: {np.mean([r['n_value_claims'] for r in valid]):.1f}")
        print(f"Avg comp claims:  {np.mean([r['n_computation_claims'] for r in valid]):.1f}")
        print(f"Zero claims ratio: {np.mean([1 if r['n_value_claims']==0 and r['n_computation_claims']==0 else 0 for r in valid]):.3f}")
        print(f"\nSaved to: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir (must be served via vLLM)")
    parser.add_argument("--data", default=os.path.join(BASE, "data/charts_v2/grpo_train_final.json"))
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--output", default="results/posthoc_analysis.jsonl")
    parser.add_argument("--vllm_port", type=int, default=9100)
    parser.add_argument("--model_name", default="qwen35-4b", help="Model name as registered in vLLM")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    asyncio.run(run_analysis(args))


if __name__ == "__main__":
    main()
