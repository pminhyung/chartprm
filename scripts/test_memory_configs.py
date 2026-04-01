"""
Test different training configurations for memory usage.
Runs 2 steps with each config and reports peak GPU memory.

Usage:
  CUDA_VISIBLE_DEVICES=4 python scripts/test_memory_configs.py --config <config_name>

Configs to test:
  1. baseline: grad_ckpt=False, liger=False (current v7 setting)
  2. grad_ckpt: grad_ckpt=True(reentrant), liger=False
  3. liger: grad_ckpt=False, liger=True
  4. both: grad_ckpt=True(reentrant) + liger=True

For each, test with:
  - num_gen=4, max_comp=1024 (current)
  - num_gen=8, max_comp=2048
  - num_gen=16, max_comp=4096
"""
import argparse
import json
import os
import sys
import torch
import numpy as np
from datasets import Dataset
from PIL import Image

os.environ.setdefault("VLLM_PORT", "9100")

def get_model_and_processor(model_path):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import trl.trainer.grpo_trainer as _grpo_mod
    _grpo_mod.AutoModelForCausalLM = AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(model_path)

    # Proxy missing attrs
    _orig_getattr = type(processor).__getattr__ if hasattr(type(processor), '__getattr__') else None
    def _proc_getattr(self, name):
        if name == 'tokenizer':
            raise AttributeError(name)
        if _orig_getattr:
            try: return _orig_getattr(self, name)
            except AttributeError: pass
        try:
            tok = object.__getattribute__(self, 'tokenizer')
            return getattr(tok, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    type(processor).__getattr__ = _proc_getattr

    return model, processor


def make_tiny_dataset(data_path, n=8):
    SYSTEM = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    with open(data_path) as f:
        raw = [json.loads(line) for line in f][:n]

    samples = []
    for item in raw:
        img = Image.open(item["image_path"]).convert("RGB")
        mx = max(img.size)
        if mx > 1024:
            s = 1024 / mx
            img = img.resize((int(img.size[0]*s), int(img.size[1]*s)), Image.LANCZOS)
        samples.append({
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": f"Question: {item['question']}"}],
            "images": [img],
            "answer": str(item["answer"]),
        })
    return Dataset.from_list(samples)


def dummy_reward(completions, answer, **kw):
    return [0.5] * len(completions)


def test_config(config_name, num_gen, max_comp, num_gpus):
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    model_path = "/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b"
    data_path = "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/chartvr_train_final.jsonl"

    # Config variations
    grad_ckpt = config_name in ("grad_ckpt", "both")
    liger = config_name in ("liger", "both")

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  grad_checkpointing={grad_ckpt}, liger={liger}")
    print(f"  num_gen={num_gen}, max_comp={max_comp}, gpus={num_gpus}")
    print(f"{'='*60}")

    dataset = make_tiny_dataset(data_path, n=num_gen * 2)

    config = GRPOConfig(
        output_dir=f"/tmp/test_config_{config_name}",
        loss_type="dapo",
        mask_truncated_completions=True,
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=int(os.environ.get("VLLM_PORT", "9100")),
        vllm_server_timeout=600.0,
        num_generations=num_gen,
        max_completion_length=max_comp,
        generation_batch_size=max(num_gen, num_gpus),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": True} if grad_ckpt else {},
        use_liger_kernel=liger,
        max_grad_norm=1.0,
        max_steps=2,
        logging_steps=1,
        save_steps=999,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model, processor = get_model_and_processor(model_path)

    try:
        trainer = GRPOTrainer(
            model=model, args=config, train_dataset=dataset,
            reward_funcs=dummy_reward, processing_class=processor,
            peft_config=peft_config,
        )
        trainer.train()

        # Report peak memory
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n✅ SUCCESS: peak GPU memory = {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
        return True, peak_mb
    except Exception as e:
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\n❌ FAILED: {type(e).__name__}: {str(e)[:200]}")
        print(f"   Peak memory before failure: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")
        return False, peak_mb
    finally:
        del model, trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="both", choices=["baseline", "grad_ckpt", "liger", "both"])
    parser.add_argument("--num_gen", type=int, default=4)
    parser.add_argument("--max_comp", type=int, default=1024)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    success, peak = test_config(args.config, args.num_gen, args.max_comp, args.num_gpus)
