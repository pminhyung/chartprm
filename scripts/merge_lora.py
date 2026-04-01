"""Merge LoRA adapter into base model for inference."""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model path")
    parser.add_argument("--lora", required=True, help="LoRA checkpoint path")
    parser.add_argument("--output", required=True, help="Output merged model path")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter: {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora)

    print("Merging and unloading...")
    model = model.merge_and_unload()

    print(f"Saving to: {args.output}")
    model.save_pretrained(args.output)

    processor = AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
