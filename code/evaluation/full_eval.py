"""
Phase 6: Evaluate all models on all benchmarks.
Uses vLLM for fast inference, then computes metrics.

Usage:
    python code/evaluation/full_eval.py --model-name sft_only --benchmark chartqa_human
    python code/evaluation/full_eval.py --all  # Run all model×benchmark combinations
"""
import json
import os
import re
import argparse
import base64
import random
from collections import defaultdict

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

MODELS = {
    "qwen25vl_zeroshot": os.path.join(BASE, "models/qwen25vl-7b"),
    "qwen3vl_thinking": os.path.join(BASE, "models/qwen3vl-8b-thinking"),
    "sft_only": os.path.join(BASE, "checkpoints/sft"),
    "grpo_baseline": os.path.join(BASE, "checkpoints/grpo_baseline"),
    "grpo_cvr": os.path.join(BASE, "checkpoints/grpo_cvr"),
}

BENCHMARKS = {
    "chartqa_human": {
        "data": os.path.join(BASE, "data/chartqa/test/test_human.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "metric": "relaxed_accuracy",
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_augmented": {
        "data": os.path.join(BASE, "data/chartqa/test/test_augmented.json"),
        "images": os.path.join(BASE, "data/chartqa/test/png"),
        "metric": "relaxed_accuracy",
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "charxiv_reasoning": {
        "data": os.path.join(BASE, "data/charxiv/val_reasoning.json"),
        "images": os.path.join(BASE, "data/charxiv/images"),
        "metric": "relaxed_accuracy",
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
    "chartqa_pro": {
        "data": os.path.join(BASE, "data/chartqa_pro/test.json"),
        "images": os.path.join(BASE, "data/chartqa_pro/images"),
        "metric": "relaxed_accuracy",
        "key_mapping": {"question": "query", "answer": "label", "image": "imgname"},
    },
}

EVAL_PROMPT = """Look at this chart and answer the question.
Write your step-by-step reasoning inside <thinking>...</thinking> tags.
Then give your final answer inside <answer>...</answer> tags.

Question: {question}"""


def relaxed_accuracy(prediction: str, gold: str, tolerance: float = 0.05) -> float:
    """Standard relaxed accuracy metric for chart QA."""
    pred = prediction.strip().lower()
    gold_str = gold.strip().lower()

    # Try numeric comparison
    def to_float(s):
        try:
            s = s.replace(",", "").replace("%", "").replace("$", "").strip()
            return float(s)
        except ValueError:
            return None

    pred_f = to_float(pred)
    gold_f = to_float(gold_str)

    if pred_f is not None and gold_f is not None:
        if gold_f == 0:
            return 1.0 if abs(pred_f) < 0.01 else 0.0
        return 1.0 if abs(pred_f - gold_f) / abs(gold_f) <= tolerance else 0.0

    # String comparison
    return 1.0 if pred == gold_str else 0.0


def extract_answer(response: str) -> str:
    """Extract answer from model response with robust normalization."""
    # Try <answer>...</answer> format
    match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if match:
        return _normalize_answer(match.group(1).strip())

    # Try \boxed{...}
    match = re.search(r'\\boxed\{(.*?)\}', response)
    if match:
        return _normalize_answer(match.group(1).strip())

    # Remove <thinking>...</thinking> block if present
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()
    if cleaned:
        # Try to find the last substantive line
        lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
        if lines:
            return _normalize_answer(lines[-1])

    # Absolute last resort: last non-empty line of full response
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return _normalize_answer(lines[-1]) if lines else ""


def _normalize_answer(answer: str) -> str:
    """Strip common prefixes/suffixes from extracted answers."""
    # Remove common prefixes
    prefixes = [
        "Final answer:", "Final Answer:", "The answer is",
        "The answer is:", "Answer:", "answer:",
        "Therefore, the answer is", "So the answer is",
        "So, the answer is", "Thus, the answer is",
    ]
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break

    # Remove trailing period if answer is short
    if len(answer) < 50 and answer.endswith('.'):
        answer = answer[:-1].strip()

    # Remove surrounding quotes
    if (answer.startswith('"') and answer.endswith('"')) or \
       (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1].strip()

    # Remove bold markdown
    answer = answer.replace("**", "").strip()

    # If still a full sentence with the actual value, try to extract just the value
    # e.g., "No, the sum value of Madagascar is not more than Fiji" -> "No"
    if len(answer) > 30 and ',' in answer:
        first_part = answer.split(',')[0].strip()
        if first_part.lower() in ('yes', 'no'):
            answer = first_part

    return answer


def load_benchmark(benchmark_name: str):
    """Load benchmark data."""
    config = BENCHMARKS[benchmark_name]
    with open(config["data"]) as f:
        data = json.load(f)

    km = config["key_mapping"]
    samples = []
    for item in data:
        question = item.get(km["question"])
        answer = str(item.get(km["answer"]))
        image = item.get(km["image"])
        image_path = os.path.join(config["images"], image)

        if os.path.exists(image_path):
            samples.append({
                "question": question,
                "answer": answer,
                "image_path": image_path,
            })

    return samples


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def run_inference(model_path: str, samples: list, gpu_ids: str = "0,1,2,3",
                  max_samples: int = None) -> list:
    """Run inference via OpenAI SDK against a vLLM-served model endpoint."""
    from openai import OpenAI
    from PIL import Image as PILImage
    import io

    if max_samples:
        samples = samples[:max_samples]

    # model_path can be a local path or a model name served by vLLM
    model_name = os.path.basename(model_path)
    client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    results = []
    for i, sample in enumerate(samples):
        # Resize large images
        img = PILImage.open(sample["image_path"]).convert("RGB")
        max_side = max(img.size)
        if max_side > 1024:
            scale = 1024 / max_side
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), PILImage.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": EVAL_PROMPT.format(question=sample["question"])},
            ]
        }]

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=1024,
            )
            response = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] Error: {e}")
            response = ""

        pred = extract_answer(response)
        gold = sample["answer"]
        acc = relaxed_accuracy(pred, gold)

        results.append({
            "question": sample["question"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "response": response,
            "accuracy": acc,
        })

        if (i + 1) % 50 == 0:
            running_acc = sum(r["accuracy"] for r in results) / len(results)
            print(f"  [{i+1}/{len(samples)}] running accuracy: {running_acc:.2%}")

    return results


def evaluate_single(model_name: str, benchmark_name: str, gpu_ids: str = "0,1,2,3"):
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

    results = run_inference(model_path, samples, gpu_ids=gpu_ids)

    if not results:
        print("  No results generated")
        return None

    accuracy = sum(r["accuracy"] for r in results) / len(results)
    print(f"  Accuracy: {accuracy:.2%} ({sum(r['accuracy'] > 0 for r in results)}/{len(results)})")

    # Save results
    output_dir = os.path.join(BASE, "results/main_table")
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


def evaluate_all(gpu_ids: str = "0,1,2,3"):
    """Evaluate all models on all benchmarks."""
    all_results = {}

    for model_name in MODELS:
        all_results[model_name] = {}
        for bench_name in BENCHMARKS:
            acc = evaluate_single(model_name, bench_name, gpu_ids=gpu_ids)
            if acc is not None:
                all_results[model_name][bench_name] = acc

    # Save complete results
    output_path = os.path.join(BASE, "results/main_table/complete_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("COMPLETE RESULTS")
    print(f"{'='*80}")
    header = f"{'Model':<25}" + "".join(f"{b:<20}" for b in BENCHMARKS)
    print(header)
    print("-" * len(header))
    for model_name, results in all_results.items():
        row = f"{model_name:<25}"
        for bench_name in BENCHMARKS:
            acc = results.get(bench_name)
            row += f"{f'{acc:.2%}':<20}" if acc is not None else f"{'N/A':<20}"
        print(row)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if args.all:
        evaluate_all(gpu_ids=args.gpu_ids)
    elif args.model_name and args.benchmark:
        evaluate_single(args.model_name, args.benchmark, gpu_ids=args.gpu_ids)
    else:
        print("Usage: --all or --model-name X --benchmark Y")
