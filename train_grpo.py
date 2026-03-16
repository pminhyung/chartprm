"""
ChartVCR GRPO Training (v4 — DEFINITIVE).

Base model: Qwen3-VL-8B-Thinking (reasoning-native VLM).
No SFT needed. This model already produces <think>...</think> reasoning traces.

Usage:
  # Baseline (outcome-only reward):
  accelerate launch --config_file accelerate_config.yaml \
      train_grpo.py --reward_type outcome_only --output_dir ckpt/baseline

  # Ours (ChartVCR reward):
  accelerate launch --config_file accelerate_config.yaml \
      train_grpo.py --reward_type chartvr --output_dir ckpt/cvr
"""
import argparse
import json
import re
import math
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════

def load_grpo_dataset(data_dir=None):
    """Load ChartQA train as GRPO dataset. Uses disk cache if available."""
    cache_path = os.path.join(BASE, "data/grpo_dataset_v4")
    if os.path.exists(cache_path):
        from datasets import load_from_disk
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(cache_path)

    if data_dir is None:
        data_dir = os.path.join(BASE, "data/chartqa")

    samples = []
    for split_file in ["train_human.json", "train_augmented.json"]:
        path = os.path.join(data_dir, "train", split_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        for item in raw:
            q = item.get("query") or item.get("question", "")
            a = str(item.get("label") if item.get("label") is not None else item.get("answer", ""))
            img = item.get("imgname") or item.get("image", "")
            base_name = os.path.splitext(img)[0]
            csv_path = os.path.join(data_dir, "train", "tables", f"{base_name}.csv")
            img_path = os.path.join(data_dir, "train", "png", img)

            if os.path.exists(csv_path) and os.path.exists(img_path):
                # Store flat strings — reconstruct messages in dataset map
                samples.append({
                    "question": q,
                    "image_path": img_path,
                    "answer": a,
                    "csv_path": csv_path,
                })

    SYSTEM_TEXT = (
        "You are an expert chart analyst. "
        "After your reasoning, put your final answer inside "
        "<answer></answer> tags with ONLY the core value or keyword. "
        "Example: <answer>42</answer>"
    )

    # Build prompt as flat strings to avoid pyarrow mixed-type issues.
    # TRL will use process_class to tokenize + image inject via "images" column.
    def build_text_prompt(q):
        return [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": f"Question: {q}"},
        ]

    from PIL import Image as PILImage

    flat_samples = []
    for i, s in enumerate(samples):
        img = PILImage.open(s["image_path"]).convert("RGB")
        # Resize large images for memory efficiency
        max_side = max(img.size)
        if max_side > 1024:
            scale = 1024 / max_side
            img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), PILImage.LANCZOS)
        flat_samples.append({
            "prompt": build_text_prompt(s["question"]),
            "images": [img],
            "answer": s["answer"],
            "csv_path": s["csv_path"],
        })
        if (i+1) % 5000 == 0:
            print(f"  Loaded {i+1}/{len(samples)} images")

    print(f"  All {len(flat_samples)} images loaded")
    return Dataset.from_list(flat_samples)

# ════════════════════════════════════════════════════════════
# Reward Functions
# ════════════════════════════════════════════════════════════

NUM_RE = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
CALC_RE = rf'({NUM_RE})\s*([+\-*/×÷])\s*({NUM_RE})\s*[=≈]\s*({NUM_RE})'


def relaxed_match(pred: str, gold: str) -> bool:
    """ChartQA relaxed accuracy: 5% tolerance for numeric, exact for text."""
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        if gf == 0:
            return abs(pf) < 0.01
        return abs(pf - gf) / abs(gf) <= 0.05
    except ValueError:
        return p.lower() == g.lower()


def extract_answer(response: str) -> str:
    """Extract final answer from model response."""
    # Try \boxed{}
    m = re.search(r'\\boxed\{(.*?)\}', response)
    if m:
        return m.group(1).strip()
    # Try <answer>...</answer>
    m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # After </think>
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        lines = [l.strip() for l in after.split('\n') if l.strip()]
        if lines:
            return lines[-1]
    # Last line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    return lines[-1] if lines else ""


def get_table_values(csv_path: str) -> set:
    """Extract all numeric values from a chart's data table CSV."""
    try:
        df = pd.read_csv(csv_path)
        vals = set()
        for col in df.columns:
            for v in df[col]:
                try:
                    vals.add(float(v))
                except (ValueError, TypeError):
                    pass
        return vals
    except Exception:
        return set()


def gaussian_score(model_val: float, table_val: float, sigma: float = 0.10) -> float:
    """Continuous accuracy score using Gaussian decay."""
    if table_val == 0:
        return 1.0 if abs(model_val) < 0.01 else 0.0
    return math.exp(-0.5 * (abs(model_val - table_val) / (abs(table_val) * sigma)) ** 2)


def compute_process_reward(response: str, csv_path: str, sigma: float = 0.10) -> float:
    """
    ChartVCR process reward.
    Sentence-level verification + causal error attribution on <think> trace.
    """
    table_vals = get_table_values(csv_path)
    if not table_vals:
        return 0.0

    think_m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if not think_m:
        return 0.0
    think = think_m.group(1).strip()

    sentences = [s.strip() for s in think.split('\n') if s.strip() and len(s.strip()) > 5]
    if not sentences:
        return 0.0

    tainted = set()
    rewards = []

    for sent in sentences:
        nums = []
        for m in re.findall(NUM_RE, sent):
            try:
                nums.append(float(m.replace(',', '')))
            except ValueError:
                pass
        if not nums:
            continue

        # input_quality: how accurate are the numbers vs table
        iq_list = []
        uses_tainted = False
        for n in nums:
            if any(abs(n - t) / max(abs(t), 1e-10) < 0.05 for t in tainted):
                uses_tainted = True
                iq_list.append(0.3)  # Tainted input gets low quality
            else:
                best = max((gaussian_score(n, tv, sigma) for tv in table_vals), default=0.0)
                iq_list.append(best)
        input_quality = min(iq_list) if iq_list else 1.0

        # logic_score: check arithmetic correctness
        cm = re.search(CALC_RE, sent)
        if cm:
            try:
                a = float(cm.group(1).replace(',', ''))
                op = cm.group(2).replace('×', '*').replace('÷', '/')
                b = float(cm.group(3).replace(',', ''))
                stated = float(cm.group(4).replace(',', ''))
                if op in '+-*/' and (op != '/' or b != 0):
                    expected = eval(f"{a}{op}{b}")
                    logic_score = 1.0 if abs(stated - expected) / max(abs(expected), 1e-10) < 0.05 else 0.0
                else:
                    logic_score = 1.0
            except Exception:
                logic_score = 1.0
        else:
            logic_score = 1.0

        r = logic_score * input_quality
        rewards.append(r)

        # Taint numbers from incorrect sentences (source errors)
        if r < 0.5 and not uses_tainted:
            for n in nums:
                tainted.add(n)

    return sum(rewards) / len(rewards) if rewards else 0.0


def reward_outcome_only(completions, answer, **kwargs):
    """Baseline: R_accuracy + R_format."""
    # answer may be a list (one per completion) or a single string
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    results = []
    for comp, gold in zip(completions, answer):
        resp = comp[0]["content"] if comp else ""
        ans = extract_answer(resp)
        r_acc = 1.0 if relaxed_match(ans, gold) else 0.0
        r_fmt = 1.0 if '<think>' in resp and '</think>' in resp else 0.0
        results.append(r_acc + r_fmt)
    return results


def reward_chartvr(completions, answer, csv_path="", sigma=0.10,
                   w_acc=0.5, w_proc=0.3, w_fmt=0.2, **kwargs):
    """ChartVCR: R_accuracy + R_process(causal) + R_format."""
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    if isinstance(csv_path, str):
        csv_path = [csv_path] * len(completions)
    results = []
    for comp, gold, csv in zip(completions, answer, csv_path):
        resp = comp[0]["content"] if comp else ""
        ans = extract_answer(resp)
        r_acc = 1.0 if relaxed_match(ans, gold) else 0.0
        r_proc = compute_process_reward(resp, csv, sigma) if csv else 0.0
        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.5
            lines = [l for l in resp.split('<think>')[1].split('</think>')[0].split('\n') if l.strip()]
            if len(lines) >= 3:
                r_fmt += 0.5
        results.append(w_acc * r_acc + w_proc * r_proc + w_fmt * min(r_fmt, 1.0))
    return results


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", choices=["outcome_only", "chartvr"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=os.path.join(BASE, "models/qwen3vl-8b-thinking"))
    parser.add_argument("--data_dir", default=os.path.join(BASE, "data/chartqa"))
    parser.add_argument("--sigma", type=float, default=0.10)
    # Ablation controls
    parser.add_argument("--w_process", type=float, default=0.3)
    parser.add_argument("--disable_causal", action="store_true")
    args = parser.parse_args()

    # Load dataset
    dataset = load_grpo_dataset(args.data_dir)
    print(f"GRPO dataset: {len(dataset)} samples")

    # Select reward function
    if args.reward_type == "outcome_only":
        reward_fn = reward_outcome_only
    else:
        # For ablation: adjust weights via closure
        def reward_fn(completions, answer, csv_path="", **kw):
            return reward_chartvr(
                completions, answer, csv_path,
                sigma=args.sigma,
                w_proc=args.w_process,
                w_acc=0.5 + (0.3 - args.w_process),  # redistribute removed weight to acc
                **kw,
            )

    print(f"Reward type: {args.reward_type}")

    # GRPO config — server mode (vLLM on GPUs 12-15, training on GPUs 0-7)
    config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=9100,
        vllm_server_timeout=300.0,
        num_generations=2,
        max_completion_length=1024,
        generation_batch_size=10,
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        warmup_steps=50,
        # Precision
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        # Logging & saving
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    # Model — use AutoModelForImageTextToText for VLMs (Qwen3-VL)
    from transformers import AutoModelForImageTextToText
    import trl.trainer.grpo_trainer as _grpo_mod
    _grpo_mod.AutoModelForCausalLM = AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Patch: TRL expects tokenizer attributes on the processor
    # Proxy missing attrs to tokenizer — must use object.__getattribute__ to avoid recursion
    _orig_getattr = type(processor).__getattr__ if hasattr(type(processor), '__getattr__') else None
    def _proc_getattr(self, name):
        if name == 'tokenizer':
            raise AttributeError(name)  # Let normal lookup handle 'tokenizer'
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

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
