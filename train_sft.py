"""
ChartVCR v8: SFT Training with LoRA.

Usage:
  # 4B SFT (Row 2)
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 accelerate launch \
      --num_processes 8 --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
      train_sft.py \
      --model_path models/qwen3.5-4b \
      --data data/sft_30k.jsonl \
      --output_dir ckpt/sft_4b \
      --use_lora --lora_rank 64 --lora_alpha 128

  # 9B SFT
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 accelerate launch \
      --num_processes 8 --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
      train_sft.py \
      --model_path models/qwen3.5-9b \
      --data data/sft_30k.jsonl \
      --output_dir ckpt/sft_9b \
      --use_lora --lora_rank 32 --lora_alpha 64
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from chartvr.prompts import EVAL_SYSTEM_PROMPT


def load_sft_dataset(data_path: str, max_image_side: int = 1024) -> Dataset:
    """Load SFT data with LAZY image decoding.

    Previously this eagerly ran PIL.Image.open + resize on every sample up
    front and stored the PIL objects in the Dataset. With 43K samples × 8
    parallel worker processes (DeepSpeed) that was a >2-hour startup bottleneck.

    Now we:
      1. Scan the JSONL once, dropping rows whose image file is missing.
      2. Store only image paths + pre-rendered messages (no PIL objects).
      3. Attach a Dataset.set_transform() hook so PIL.open/resize happens
         per-batch during DataLoader iteration (parallel workers).
    """
    with open(data_path) as f:
        raw = [json.loads(line) for line in f]

    samples = []
    skipped = 0
    for i, item in enumerate(raw):
        q = item["question"]
        ans = str(item["answer"])
        img_path = item.get("image_path", "")

        if not img_path or not os.path.exists(img_path):
            skipped += 1
            continue

        # Prefer pre-assembled assistant_text from teacher distillation
        # (full `<think>...</think><answer>...</answer>` string, used as-is).
        # Fallback: reconstruct from `reasoning` / `reasoning_steps` (legacy v7-v9.1 data).
        assistant_text = item.get("assistant_text", "")
        if assistant_text:
            response = assistant_text
        else:
            reasoning = item.get("reasoning", item.get("reasoning_steps", ""))
            if isinstance(reasoning, list):
                reasoning = "\n".join(reasoning)

            if not reasoning:
                raise ValueError(
                    f"Sample missing reasoning — template fallback disabled. "
                    f"Q: {q[:80]} | source={item.get('source', '?')} | img={img_path}"
                )
            response = f"<think>\n{reasoning}\n</think>\n<answer>{ans}</answer>"

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Look at this chart and answer the question.\n\nQuestion: {q}"},
            {"role": "assistant", "content": response},
        ]

        samples.append({
            "messages": messages,
            "image_path": img_path,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{len(raw)} (skipped {skipped})")

    print(f"  SFT dataset: {len(samples)} rows scanned, {skipped} skipped from {data_path}")
    ds = Dataset.from_list(samples)

    def _lazy_load(batch):
        imgs = []
        for p in batch["image_path"]:
            try:
                img = Image.open(p).convert("RGB")
                m = max(img.size)
                if m > max_image_side:
                    s = max_image_side / m
                    img = img.resize(
                        (int(img.size[0] * s), int(img.size[1] * s)),
                        Image.LANCZOS,
                    )
            except Exception:
                # Should never happen — path was validated at scan time —
                # but fall back to a 64x64 white PIL so training can continue.
                img = Image.new("RGB", (64, 64), color="white")
            imgs.append([img])
        batch["images"] = imgs
        return batch

    ds.set_transform(_lazy_load)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    dataset = load_sft_dataset(args.data)

    # Model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    if args.use_lora and hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    # Proxy missing attrs to tokenizer (same as train_grpo_dapo.py)
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

    # LoRA
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

    # SFT Config
    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_liger_kernel=True,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_length=args.max_length,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,  # keep image_path for lazy PIL transform
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
