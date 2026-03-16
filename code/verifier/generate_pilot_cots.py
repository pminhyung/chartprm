"""
Phase 1.1: Generate 100 CoT traces from Qwen3-VL-8B-Thinking on ChartQA train samples.
Uses vLLM for efficient batch inference.
Qwen3-VL-8B-Thinking has native thinking mode — produces <think>...</think> reasoning.
Filters for computation-requiring questions to test verifier on meaningful CoTs.
"""
import json
import os
import random
import re
import base64
from vllm import LLM, SamplingParams

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

COMPUTE_KEYWORDS = [
    "ratio", "difference", "sum", "average", "total",
    "how many", "percentage", "change", "increase", "decrease",
    "more than", "less than", "compare", "between", "subtract",
    "add", "multiply", "divide", "calculate", "what is the value"
]

PROMPT_TEMPLATE = """Look at this chart and answer the question.

Question: {question}

Provide your final answer in \\boxed{{}}."""


def generate_cots(n=100, gpu_ids="0,1"):
    """Generate CoT traces using Qwen3-VL-8B-Thinking via vLLM."""
    model_path = os.path.join(BASE, "models/qwen3vl-8b-thinking")
    output_path = os.path.join(BASE, "results/pilot/cot_traces.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load GRPO data and filter for computation questions
    with open(os.path.join(BASE, "data/grpo_train.json")) as f:
        all_data = json.load(f)

    compute_samples = [
        s for s in all_data
        if any(kw in s["problem"].lower() for kw in COMPUTE_KEYWORDS)
    ]

    random.seed(42)
    selected = random.sample(compute_samples, min(n, len(compute_samples)))
    print(f"Selected {len(selected)} computation-requiring samples from {len(compute_samples)} candidates")

    # Initialize vLLM
    num_gpus = len(gpu_ids.split(","))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=2048,
        top_p=0.95,
        top_k=20,
    )

    # Prepare prompts with images
    prompts = []
    for sample in selected:
        image_path = sample["image"]
        question = sample["problem"]

        # Read and encode image
        with open(image_path, "rb") as img_f:
            image_b64 = base64.b64encode(img_f.read()).decode("utf-8")

        prompt_text = PROMPT_TEMPLATE.format(question=question)

        # Qwen3-VL chat format: model produces <think> reasoning natively
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        prompts.append({
            "messages": messages,
            "sample": sample
        })

    # Run batch inference using chat API
    chat_inputs = [p["messages"] for p in prompts]
    outputs = llm.chat(chat_inputs, sampling_params)

    # Collect results
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        sample = prompts[i]["sample"]

        # Extract reasoning_content (text inside <think>...</think>)
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning_content = think_match.group(1).strip() if think_match else ""

        # Extract final answer from \boxed{}
        boxed_match = re.search(r'\\boxed\{(.*?)\}', response)
        extracted_answer = boxed_match.group(1).strip() if boxed_match else ""

        # If no boxed answer, try last line after </think>
        if not extracted_answer and '</think>' in response:
            after_think = response.split('</think>')[-1].strip()
            if after_think:
                extracted_answer = after_think.split('\n')[-1].strip()

        results.append({
            "question": sample["problem"],
            "gold_answer": sample["solution"],
            "image": sample["image"],
            "csv_path": sample["csv_path"],
            "model_response": response,
            "reasoning_content": reasoning_content,
            "extracted_answer": extracted_answer,
            "source": sample.get("source", "chartqa"),
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(results)} CoT traces → {output_path}")

    # Statistics
    has_think = sum(1 for r in results if r["reasoning_content"])
    has_answer = sum(1 for r in results if r["extracted_answer"])
    avg_reasoning_len = sum(len(r["reasoning_content"]) for r in results) / max(len(results), 1)
    print(f"  With <think> reasoning: {has_think}/{len(results)}")
    print(f"  With extracted answer: {has_answer}/{len(results)}")
    print(f"  Avg reasoning length: {avg_reasoning_len:.0f} chars")

    # Print sample
    if results:
        r = results[0]
        print(f"\nSample CoT:")
        print(f"  Q: {r['question'][:100]}")
        print(f"  Gold A: {r['gold_answer']}")
        print(f"  Model A: {r['extracted_answer']}")
        print(f"  Reasoning (first 300 chars): {r['reasoning_content'][:300]}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--gpu-ids", type=str, default="0,1")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    generate_cots(n=args.n, gpu_ids=args.gpu_ids)
