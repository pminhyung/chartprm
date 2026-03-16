"""
Phase 2: Direct SFT training using trl.SFTTrainer
Teaches Qwen2.5-VL-7B the <thinking>/<answer> format on ChartQA data.
No LLaMA-Factory dependency.
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
)
from transformers import Trainer as _Trainer


class Trainer(_Trainer):
    """Override to fix num_items_in_batch and accelerate fp32 conversion issues with Qwen2.5-VL."""
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels", None)

        with self.compute_loss_context_manager():
            outputs = model(**inputs, labels=labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        del inputs
        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
from PIL import Image

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")


def load_sft_data(max_samples=None):
    """Load SFT data."""
    with open(os.path.join(BASE, "data/sft_train.json")) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    # Convert to messages format for SFTTrainer
    records = []
    for item in data:
        convs = item["conversations"]
        image_path = item["images"][0]

        if not os.path.exists(image_path):
            continue

        records.append({
            "question": convs[0]["value"].replace("<image>\n", ""),
            "answer": convs[1]["value"],
            "image_path": image_path,
        })

    return Dataset.from_list(records)


def main():
    model_path = os.path.join(BASE, "models/qwen25vl-7b")
    output_dir = os.path.join(BASE, "checkpoints/sft")

    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("Loading data...")
    dataset = load_sft_data()
    print(f"Dataset: {len(dataset)} samples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        average_tokens_across_devices=False,
        deepspeed=os.path.join(BASE, "code/bigcharts-r1/src/open-r1-multimodal/local_scripts/zero3.json"),
    )

    def collate_fn(examples):
        # Process one at a time to avoid image count mismatches
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        extra_keys = {}

        for example in examples:
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{example['image_path']}"},
                    {"type": "text", "text": example["question"]},
                ]},
                {"role": "assistant", "content": example["answer"]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            try:
                images = [Image.open(example["image_path"]).convert("RGB")]
            except Exception:
                images = [Image.new("RGB", (224, 224), (128, 128, 128))]

            single = processor(
                text=[text],
                images=images if images else None,
                return_tensors="pt",
                padding=False,
            )

            all_input_ids.append(single["input_ids"].squeeze(0))
            all_attention_masks.append(single["attention_mask"].squeeze(0))

            labels = single["input_ids"].squeeze(0).clone()
            all_labels.append(labels)

            # Collect extra keys (pixel_values, image_grid_thw, etc.)
            for k, v in single.items():
                if k not in ("input_ids", "attention_mask"):
                    if k not in extra_keys:
                        extra_keys[k] = []
                    # Keep original dimensions — don't squeeze
                    extra_keys[k].append(v)

        # Pad sequences
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = processor.tokenizer.pad_token_id or 0

        batch = {
            "input_ids": torch.stack([
                torch.nn.functional.pad(ids, (0, max_len - ids.shape[0]), value=pad_id)
                for ids in all_input_ids
            ]),
            "attention_mask": torch.stack([
                torch.nn.functional.pad(mask, (0, max_len - mask.shape[0]), value=0)
                for mask in all_attention_masks
            ]),
            "labels": torch.stack([
                torch.nn.functional.pad(lbl, (0, max_len - lbl.shape[0]), value=-100)
                for lbl in all_labels
            ]),
        }

        # Concatenate extra keys (pixel_values, image_grid_thw)
        for k, vs in extra_keys.items():
            try:
                batch[k] = torch.cat(vs, dim=0)
            except Exception:
                try:
                    batch[k] = torch.stack(vs)
                except Exception:
                    batch[k] = vs[0]

        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
