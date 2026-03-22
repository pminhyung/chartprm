"""
ChartVCR Evaluation (v4).

Evaluates models on all benchmarks using vLLM for inference.

Models:
  1. Qwen3-VL-8B-Thinking zero-shot
  2. GRPO baseline (ckpt/baseline)
  3. GRPO CVR (ckpt/cvr)

Benchmarks:
  - ChartQA-Human test (relaxed accuracy)
  - ChartQA-Augmented test (relaxed accuracy)
  - CharXiv val reasoning (relaxed accuracy)
  - ChartQA-Pro test (relaxed accuracy)

Usage:
  python eval_all.py --model qwen3vl_zeroshot --benchmark chartqa_human --gpu-ids 0,1,2,3
  python eval_all.py --all --gpu-ids 0,1,2,3
"""
import json
import os
import re
import argparse
import base64
import io
from PIL import Image
from collections import defaultdict

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

MODELS = {
    "qwen3vl_zeroshot": os.path.join(BASE, "models/qwen3vl-8b-thinking"),
    "grpo_baseline": os.path.join(BASE, "ckpt/baseline"),
    "grpo_cvr": os.path.join(BASE, "ckpt/cvr"),
}

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
    "charxiv_reasoning": {
        "data": os.path.join(BASE, "data/charxiv/val_reasoning.json"),
        "images": os.path.join(BASE, "data/charxiv/images"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_pro": {
        "data": os.path.join(BASE, "data/chartqa_pro/test.json"),
        "images": os.path.join(BASE, "data/chartqa_pro/images"),
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
}

# System prompt forces answer format: <answer> key value only </answer>
SYSTEM_PROMPT = (
    "You are an expert chart analyst. "
    "After your reasoning, you MUST put your final answer inside <answer> and </answer> tags. "
    "The <answer> tag should contain ONLY the core numeric value or keyword — "
    "no sentences, no units unless required, no explanation. "
    "Examples: <answer>42</answer> or <answer>Yes</answer> or <answer>2018</answer>"
)

EVAL_PROMPT = "Look at this chart and answer the question.\n\nQuestion: {question}"


def relaxed_accuracy(pred: str, gold: str) -> float:
    """ChartQA standard relaxed accuracy."""
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        return 1.0 if abs(pf - gf) / abs(gf) <= 0.05 else 0.0
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


def extract_answer(response: str) -> str:
    """Extract final answer from Qwen3-VL-Thinking response.

    The model produces:
    <think>...reasoning...</think>
    Final answer (may be in \boxed{}, <answer>, or just plain text after </think>)
    """
    # Try \boxed{}
    m = re.search(r'\\boxed\{(.*?)\}', response)
    if m:
        return _normalize(m.group(1).strip())

    # Try <answer>...</answer>
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if m:
        return _normalize(m.group(1).strip())

    # After </think>
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                return _normalize(lines[-1])

    # Last line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return _normalize(lines[-1]) if lines else ""


def _normalize(answer: str) -> str:
    """Strip common prefixes/suffixes and extract the core answer value."""
    # Remove bold/quotes
    answer = answer.replace("**", "").strip()
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()

    # Remove trailing period
    if len(answer) < 80 and answer.endswith('.'):
        answer = answer[:-1].strip()

    # Strip common prefixes
    prefixes = [
        "Final answer:", "The answer is", "Answer:", "Therefore,",
        "So the answer is", "Thus,", "Hence,", "In conclusion,",
        "There are", "There is", "The value is", "The difference is",
        "It is", "The total is", "The average is",
    ]
    for p in prefixes:
        if answer.lower().startswith(p.lower()):
            answer = answer[len(p):].strip()
            break

    # "Yes, because..." / "No, the sum..." -> "Yes" / "No"
    if len(answer) > 20 and ',' in answer:
        first = answer.split(',')[0].strip()
        if first.lower() in ('yes', 'no'):
            return first

    # Check for Yes/No FIRST (before number extraction)
    ans_lower = answer.lower()
    if 'no' in ans_lower.split()[:3] or ans_lower.startswith('no'):
        return 'No'
    if 'yes' in ans_lower.split()[:3] or ans_lower.startswith('yes'):
        return 'Yes'

    # If answer is a full sentence, try to extract just the number/value
    if len(answer) > 15:
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?', answer)
        if nums:
            return nums[-1].replace(',', '')
        m = re.match(r'(\d+)\s+\w+', answer)
        if m:
            return m.group(1)

    return answer


def load_benchmark(name: str):
    """Load benchmark data."""
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
            samples.append({"question": q, "answer": a, "image_path": img_path})
    return samples


def run_inference(model_path: str, samples: list, gpu_ids: str = "0,1,2,3",
                  server_url: str = None, model_id: str = None,
                  enable_thinking: bool = True, max_concurrent: int = 8):
    """Run inference via vLLM OpenAI-compatible server.

    Args:
        model_path: Local model path (used as model_id if model_id not set)
        samples: List of dicts with question, answer, image_path
        server_url: vLLM server URL (e.g. http://localhost:8000/v1)
        model_id: Model name on the server
        enable_thinking: Enable reasoning mode
        max_concurrent: Max concurrent API requests
    """
    import asyncio
    from openai import AsyncOpenAI

    if server_url is None:
        server_url = os.environ.get("VLLM_SERVER_URL", "http://localhost:8000/v1")
    if model_id is None:
        model_id = os.environ.get("VLLM_MODEL_ID", model_path)

    client = AsyncOpenAI(base_url=server_url, api_key="dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    async def infer_one(sample, sem):
        async with sem:
            img = Image.open(sample["image_path"]).convert("RGB")
            max_side = max(img.size)
            if max_side > 1024:
                scale = 1024 / max_side
                img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": EVAL_PROMPT.format(question=sample["question"])},
                    ]
                }
            ]

            try:
                extra = {}
                if enable_thinking:
                    extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.0,
                    **extra,
                )
                content = resp.choices[0].message.content or ""
                # Get reasoning if available
                raw_dump = resp.choices[0].message.model_dump() if hasattr(resp.choices[0].message, 'model_dump') else {}
                reasoning = raw_dump.get("reasoning", "") or ""
                if reasoning:
                    full_response = f"<think>{reasoning}</think>\n{content}"
                else:
                    full_response = content
                return {"content": content, "reasoning": reasoning, "response": full_response, "error": None}
            except Exception as e:
                return {"content": "", "reasoning": "", "response": "", "error": str(e)}

    async def run_all():
        sem = asyncio.Semaphore(max_concurrent)
        tasks = [infer_one(s, sem) for s in samples]
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    try:
        outputs = loop.run_until_complete(run_all())
    finally:
        loop.close()

    results = []
    for i, output in enumerate(outputs):
        # Extract answer from CONTENT only (not reasoning)
        pred = extract_answer(output["content"]) if not output["error"] else ""
        gold = samples[i]["answer"]
        acc = relaxed_accuracy(pred, gold)
        results.append({
            "question": samples[i]["question"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "response": output["response"],
            "content": output["content"],
            "reasoning_content": output["reasoning"],
            "accuracy": acc,
            "error": output["error"],
        })

    return results


def evaluate_single(model_name: str, benchmark_name: str, gpu_ids: str,
                    server_url: str = None, model_id: str = None,
                    enable_thinking: bool = True):
    """Evaluate a single model on a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {benchmark_name}")
    print(f"{'='*60}")

    model_path = MODELS[model_name]
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None

    samples = load_benchmark(benchmark_name)
    print(f"  Loaded {len(samples)} samples")

    if not samples:
        print("  No samples found")
        return None

    results = run_inference(
        model_path, samples, gpu_ids=gpu_ids,
        server_url=server_url, model_id=model_id,
        enable_thinking=enable_thinking,
    )

    if not results:
        print("  No results")
        return None

    accuracy = sum(r["accuracy"] for r in results) / len(results)
    correct = sum(1 for r in results if r["accuracy"] > 0)
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(results)})")

    output_dir = os.path.join(BASE, "results/v4")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_{benchmark_name}.json")

    with open(output_path, "w") as f:
        json.dump({
            "model": model_name,
            "benchmark": benchmark_name,
            "accuracy": accuracy,
            "n_samples": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    parser.add_argument("--server-url", type=str, default=None,
                        help="vLLM OpenAI-compatible server URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Model ID on the server")
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    enable_thinking = not args.no_thinking

    if not args.server_url:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if args.all:
        all_results = {}
        for model_name in MODELS:
            all_results[model_name] = {}
            for bench_name in BENCHMARKS:
                acc = evaluate_single(model_name, bench_name, args.gpu_ids,
                                      server_url=args.server_url, model_id=args.model_id,
                                      enable_thinking=enable_thinking)
                if acc is not None:
                    all_results[model_name][bench_name] = acc
        # Print summary
        print(f"\n{'='*80}")
        print(f"{'Model':>25} | {'CQA-H':>8} | {'CQA-A':>8} | {'CharXiv':>8} | {'CQA-Pro':>8}")
        print("-" * 70)
        for m, res in all_results.items():
            row = f"{m:>25}"
            for b in BENCHMARKS:
                a = res.get(b)
                row += f" | {f'{a:.2%}':>8}" if a is not None else f" | {'N/A':>8}"
            print(row)
    elif args.model and args.benchmark:
        evaluate_single(args.model, args.benchmark, args.gpu_ids,
                        server_url=args.server_url, model_id=args.model_id,
                        enable_thinking=enable_thinking)
    else:
        print("Usage: --all or --model X --benchmark Y")
