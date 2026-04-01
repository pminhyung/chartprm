"""
ChartVCR v7: DAPO + LoRA + Conditional Reward GRPO Training.

Usage:
  # Row A: DAPO + Outcome-only
  accelerate launch --num_processes 8 train_grpo_dapo.py \
      --reward_type outcome_only --output_dir ckpt/row_a

  # Row B: DAPO + Conditional CVR
  accelerate launch --num_processes 8 train_grpo_dapo.py \
      --reward_type conditional_cvr --verifier_type rule --output_dir ckpt/row_b
"""
import argparse
import json
import logging
import os
import re
import sys
from typing import List

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor

# ═══════════════════════════════════════════
# System Prompt (v7)
# ═══════════════════════════════════════════

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


# ═══════════════════════════════════════════
# Answer Extraction v2
# ═══════════════════════════════════════════

def _normalize_tf(s: str) -> str:
    mapping = {'true': 'yes', 'false': 'no'}
    return mapping.get(s.strip().lower(), s)


def extract_answer_v2(response: str) -> str:
    """Extract answer with TF/YN normalization, unit stripping, truncation fallback."""
    # Priority 1: <answer> tag
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if m:
        ans = m.group(1).strip()
        ans = re.sub(r'(million|billion|thousand|percent|%|dollars|\$|USD)', '', ans, flags=re.I).strip()
        ans = re.sub(r'^(approximately|about|around|roughly|~)\s*', '', ans, flags=re.I).strip()
        if len(ans) > 50:
            nums = re.findall(r'[-+]?\d*\.?\d+', ans)
            if nums:
                ans = nums[0]
        return _normalize_tf(ans)

    # Priority 2: after </think> (handles both <think>...</think> and ...</think> formats)
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            # Look for <answer> within post-think content
            m2 = re.search(r'<answer>(.*?)</answer>', after, re.DOTALL)
            if m2:
                return _normalize_tf(m2.group(1).strip())
            nums = re.findall(r'[-+]?\d*\.?\d+', after)
            if nums:
                return nums[-1]
            lines = [l.strip() for l in after.split('\n') if l.strip()]
            if lines:
                return _normalize_tf(lines[-1])

    # Priority 3: last number
    nums = re.findall(r'[-+]?\d*\.?\d+', response)
    if nums:
        return nums[-1]
    return _normalize_tf(response.strip().split('\n')[-1])


# ═══════════════════════════════════════════
# CERM Accuracy (Continuous Error Magnitude)
# ═══════════════════════════════════════════

def cerm_accuracy(pred: str, gold: str) -> float:
    """BigCharts-R1 style continuous accuracy. Returns float [0, 1]."""
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return 1.0 if abs(pf) < 0.01 else 0.0
        relative_change = abs(pf - gf) / abs(gf)
        return 1.0 / (1.0 + relative_change)
    except ValueError:
        return 1.0 if p.lower() == g.lower() else 0.0


# ═══════════════════════════════════════════
# Metrics Logger
# ═══════════════════════════════════════════

_metrics_logger = logging.getLogger("chartvr_metrics")
_metrics_handler = logging.FileHandler("logs/chartvr_v7_metrics.jsonl", mode="a")
_metrics_handler.setFormatter(logging.Formatter("%(message)s"))
_metrics_logger.addHandler(_metrics_handler)
_metrics_logger.setLevel(logging.INFO)
_batch_counter = {"n": 0}


# ═══════════════════════════════════════════
# Reward Functions
# ═══════════════════════════════════════════

def reward_outcome_only_dapo(completions, answer, **kwargs) -> List[float]:
    """Row A: DAPO + CERM outcome-only."""
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    results = []
    r_acc_list = []
    for comp, gold in zip(completions, answer):
        resp = comp[0]["content"] if comp else ""
        pred = extract_answer_v2(resp)
        r_acc = cerm_accuracy(pred, str(gold))
        results.append(r_acc)
        r_acc_list.append(r_acc)

    _batch_counter["n"] += 1
    _metrics_logger.info(json.dumps({
        "batch": _batch_counter["n"],
        "reward_type": "outcome_only",
        "reward_mean": float(np.mean(results)),
        "reward_std": float(np.std(results)),
        "r_acc_mean": float(np.mean(r_acc_list)),
        "frac_correct": sum(1 for r in r_acc_list if r >= 0.95) / len(r_acc_list),
        "n_completions": len(results),
    }))
    return results


def reward_conditional_cvr(completions, answer, csv_path="", question="",
                           verifier_type="rule", **kwargs) -> List[float]:
    """Row B: Process-Augmented Outcome Reward.

    Correct -> 1.0
    Wrong   -> min(1.0, R_acc + 0.3 * R_proc)

    Preflight analysis showed pure conditional (wrong->R_proc) has LESS variance
    than outcome (0.186 vs 0.322). Additive hybrid fixes this (est. std=0.351).
    """
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    if isinstance(csv_path, str):
        csv_path = [csv_path] * len(completions)
    if isinstance(question, str):
        question = [question] * len(completions)

    from code.rewards.rule_verifier_fast import compute_process_reward_fast

    results = []
    r_acc_list, r_proc_list = [], []
    n_correct, n_augmented = 0, 0
    think_lengths = []

    for comp, gold, csv, q in zip(completions, answer, csv_path, question):
        resp = comp[0]["content"] if comp else ""
        pred = extract_answer_v2(resp)
        r_acc = cerm_accuracy(pred, str(gold))

        # Track think length (handles both <think>...</think> and ...</think>)
        think_match = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
        if think_match:
            think_lengths.append(len(think_match.group(1).split()))
        elif '</think>' in resp:
            think_lengths.append(len(resp[:resp.index('</think>')].split()))
        else:
            think_lengths.append(0)

        # Conditional: correct -> 1.0, wrong -> R_proc
        if r_acc >= 0.95:
            results.append(1.0)
            r_acc_list.append(r_acc)
            r_proc_list.append(1.0)
            n_correct += 1
            continue

        # Wrong answer -> process reward determines score
        if verifier_type == "rule" and csv:
            r_proc = compute_process_reward_fast(resp, csv)
        else:
            r_proc = 0.5  # fallback

        r_total = min(1.0, r_acc + 0.3 * r_proc)
        results.append(r_total)
        r_acc_list.append(r_acc)
        r_proc_list.append(r_proc)
        n_augmented += 1

    _batch_counter["n"] += 1
    _metrics_logger.info(json.dumps({
        "batch": _batch_counter["n"],
        "reward_type": "conditional_cvr",
        "reward_mean": float(np.mean(results)),
        "reward_std": float(np.std(results)),
        "r_acc_mean": float(np.mean(r_acc_list)),
        "r_proc_mean": float(np.mean(r_proc_list)),
        "frac_correct": n_correct / len(results) if results else 0,
        "frac_augmented": n_augmented / len(results) if results else 0,
        "avg_think_tokens": float(np.mean(think_lengths)) if think_lengths else 0,
        "n_completions": len(results),
    }))
    return results


# ═══════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════

def load_chartvr_dataset(data_path: str, max_image_side: int = 1024) -> Dataset:
    """Load chartvr_train_final.jsonl into HF Dataset for TRL GRPOTrainer.

    Uses Dataset.from_list() and flat text prompts to avoid pyarrow mixed-type errors.
    Images are passed via 'images' column (TRL injects them via processing_class).
    """
    with open(data_path) as f:
        raw = [json.loads(line) for line in f]

    flat_samples = []
    for i, item in enumerate(raw):
        q = item["question"]
        # Flat text prompt (no multimodal content dicts — avoids pyarrow issues)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {q}"},
        ]

        img = Image.open(item["image_path"]).convert("RGB")
        max_side = max(img.size)
        if max_side > max_image_side:
            scale = max_image_side / max_side
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)

        flat_samples.append({
            "prompt": messages,
            "images": [img],
            "answer": str(item["answer"]),
            "csv_path": item["csv_path"],
            "question": q,
        })
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1}/{len(raw)} images")

    print(f"  All {len(flat_samples)} images loaded from {data_path}")
    return Dataset.from_list(flat_samples)


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b")
    parser.add_argument("--data", default="/ex_disk2/mhpark/poc/chartvr/data/charts_v2/chartvr_train_final.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_type", choices=["outcome_only", "conditional_cvr"], default="outcome_only")
    parser.add_argument("--verifier_type", choices=["rule", "api"], default="rule")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    # Ensure logs dir exists
    os.makedirs("logs", exist_ok=True)

    # ── Dataset ──
    dataset = load_chartvr_dataset(args.data)

    # ── Reward function ──
    if args.reward_type == "outcome_only":
        reward_fn = reward_outcome_only_dapo
    else:
        verifier_type = args.verifier_type
        def reward_fn(completions, answer, csv_path="", question="", **kw):
            return reward_conditional_cvr(
                completions, answer, csv_path, question,
                verifier_type=verifier_type, **kw,
            )
    print(f"Reward type: {args.reward_type} (verifier: {args.verifier_type})")

    # ── DAPO GRPOConfig ──
    from trl import GRPOConfig, GRPOTrainer

    vllm_port = int(os.environ.get("VLLM_PORT", "9100"))
    num_train_gpus = int(os.environ.get("NUM_TRAIN_GPUS", "8"))

    reward_funcs = [reward_fn]

    # Soft overlong punishment — only if max_completion_length > 2048
    if args.max_completion_length > 2048:
        from trl.rewards import get_soft_overlong_punishment
        sop = get_soft_overlong_punishment(
            max_completion_len=args.max_completion_length,
            soft_punish_cache=1024,
        )
        reward_funcs.append(sop)

    config = GRPOConfig(
        output_dir=args.output_dir,
        # DAPO core
        loss_type="dapo",
        mask_truncated_completions=True,
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        # vLLM server mode
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=vllm_port,
        vllm_server_timeout=600.0,
        # Generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        generation_batch_size=max(args.num_generations, num_train_gpus * args.per_device_batch_size),
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_liger_kernel=True,
        max_grad_norm=1.0,
        # Logging
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    # ── LoRA Config ──
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # ── Model ──
    from transformers import AutoModelForImageTextToText
    import trl.trainer.grpo_trainer as _grpo_mod
    _grpo_mod.AutoModelForCausalLM = AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # Enable gradient checkpointing input requirement for PEFT compatibility
    if args.use_lora and hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Proxy missing attrs to tokenizer
    _orig_getattr = type(processor).__getattr__ if hasattr(type(processor), '__getattr__') else None
    def _proc_getattr(self, name):
        if name == 'tokenizer':
            raise AttributeError(name)
        if _orig_getattr:
            try:
                return _orig_getattr(self, name)
            except AttributeError:
                pass
        try:
            tok = object.__getattribute__(self, 'tokenizer')
            return getattr(tok, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    type(processor).__getattr__ = _proc_getattr

    # ── Trainer ──
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
