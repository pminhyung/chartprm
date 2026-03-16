"""
Prepare SFT data for Qwen2.5-VL-7B using ChartQA.
Converts to LLaMA-Factory sharegpt format with <thinking>...</thinking><answer>...</answer>.

Since we don't have BigCharts CoT data, we create simple CoT format:
- For numeric answers: brief reasoning + answer
- For text answers: direct answer with format

This SFT step mainly teaches the model the output format for GRPO training.
"""
import json
import os
import random

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")


def create_sft_sample(question, answer, image_path):
    """Create a single SFT sample in sharegpt format."""
    # Create simple CoT response
    response = f"<thinking>\nLet me analyze the chart to answer this question.\n\nLooking at the chart, I need to find: {question}\n\nBased on my analysis of the chart, the answer is {answer}.\n</thinking>\n<answer>{answer}</answer>"

    return {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question}"
            },
            {
                "from": "gpt",
                "value": response
            }
        ],
        "images": [image_path]
    }


def prepare_sft_data():
    """Create SFT training data from ChartQA."""
    samples = []

    for split_file in ["train_human.json", "train_augmented.json"]:
        path = os.path.join(BASE, f"data/chartqa/train/{split_file}")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            data = json.load(f)

        for item in data:
            question = item.get("query") or item.get("question")
            answer = str(item.get("label") if item.get("label") is not None else item.get("answer"))
            image = item.get("imgname") or item.get("image")

            if not question or answer is None or not image:
                continue

            image_path = os.path.join(BASE, f"data/chartqa/train/png/{image}")
            if not os.path.exists(image_path):
                continue

            samples.append(create_sft_sample(question, answer, image_path))

    random.seed(42)
    random.shuffle(samples)

    # Save in LLaMA-Factory format
    output_path = os.path.join(BASE, "data/sft_train.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"SFT data: {len(samples)} samples → {output_path}")

    # Also create a small val set from test
    val_samples = []
    test_path = os.path.join(BASE, "data/chartqa/test/test_human.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_data = json.load(f)

        for item in test_data[:100]:
            question = item.get("query") or item.get("question")
            answer = str(item.get("label") if item.get("label") is not None else item.get("answer"))
            image = item.get("imgname") or item.get("image")
            image_path = os.path.join(BASE, f"data/chartqa/test/png/{image}")

            if os.path.exists(image_path):
                val_samples.append(create_sft_sample(question, answer, image_path))

    val_path = os.path.join(BASE, "data/sft_val.json")
    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print(f"SFT val: {len(val_samples)} samples → {val_path}")

    # Print sample
    s = samples[0]
    print(f"\nSample:")
    print(f"  User: {s['conversations'][0]['value'][:100]}")
    print(f"  Assistant: {s['conversations'][1]['value'][:200]}")

    return samples


if __name__ == "__main__":
    prepare_sft_data()
