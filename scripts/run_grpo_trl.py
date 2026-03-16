"""
GRPO Training using TRL's GRPOTrainer (>=0.29 with native VLM support).
"""
import os
import sys
import json
import re
import math
import torch
from datasets import Dataset

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")
sys.path.insert(0, BASE)

from transformers import AutoProcessor
from trl import GRPOTrainer, GRPOConfig


# ==================== REWARD FUNCTIONS ====================

def _extract_answer(content):
    match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def cerm_reward(completions, solution, **kwargs):
    """Continuous error magnitude reward."""
    rewards = []
    for completion, sol in zip(completions, solution):
        content = completion[0]["content"] if completion else ""
        answer = _extract_answer(content)
        if not answer:
            rewards.append(0.0)
            continue
        try:
            pred_f = float(answer.replace(",", "").rstrip("%"))
            gold_f = float(sol.replace(",", "").rstrip("%"))
            if gold_f == 0:
                r = 1.0 if pred_f == 0 else 0.0
            else:
                r = 1.0 / (1.0 + abs(pred_f - gold_f) / abs(gold_f))
        except ValueError:
            r = 1.0 if answer.strip().lower() == sol.strip().lower() else 0.0
        rewards.append(r)
    return rewards

def format_reward(completions, **kwargs):
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    return [1.0 if re.fullmatch(pattern, (c[0]["content"] if c else "").strip(), re.DOTALL) else 0.0
            for c in completions]


# ==================== DATASET ====================

def load_grpo_dataset():
    with open(os.path.join(BASE, "data/grpo_train.json")) as f:
        data = json.load(f)

    records = []
    for item in data:
        if not os.path.exists(item["image"]):
            continue
        records.append({
            "prompt": [{"role": "user", "content": [
                {"type": "image", "image": f"file://{item['image']}"},
                {"type": "text", "text": item["problem"]},
            ]}],
            "solution": item["solution"],
            "csv_path": item.get("csv_path", ""),
        })
    return Dataset.from_list(records)


# ==================== MAIN ====================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", choices=["baseline", "cvr"], default="baseline")
    parser.add_argument("--output-dir", default=os.path.join(BASE, "checkpoints/grpo_baseline"))
    args = parser.parse_args()

    model_path = os.path.join(BASE, "checkpoints/sft")

    if args.reward_mode == "baseline":
        reward_funcs = [cerm_reward, format_reward]
    else:
        from code.rewards.chart_vcr_reward import chart_vcr_reward
        reward_funcs = [chart_vcr_reward]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_generations=4,
        max_completion_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=200,
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        # No vLLM - use native HF generate() with DeepSpeed ZeRO-3
        deepspeed=os.path.join(BASE, "code/bigcharts-r1/src/open-r1-multimodal/local_scripts/zero3.json"),
    )

    dataset = load_grpo_dataset()
    print(f"Dataset: {len(dataset)} samples, Reward: {args.reward_mode}")

    # Monkey-patch AutoModelForCausalLM to use VLM class for Qwen2.5-VL
    from transformers import AutoModelForImageTextToText
    import trl.trainer.grpo_trainer as _grpo_mod
    _grpo_mod.AutoModelForCausalLM = AutoModelForImageTextToText

    # Pass model_path as string — let TRL + DeepSpeed handle distributed loading
    training_args.model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
