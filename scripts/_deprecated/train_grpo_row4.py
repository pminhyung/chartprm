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
import logging
import re
import math
import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

# Per-component metric logger (for ablation analysis)
_metrics_logger = logging.getLogger("chartvr_metrics")
_metrics_logger.setLevel(logging.INFO)
_metrics_batch_counter = {"n": 0}

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════

def load_grpo_dataset(data_dir=None):
    """Load ChartQA train as GRPO dataset. Uses disk cache if available."""
    cache_path = os.path.join(BASE, "data/grpo_dataset_v6")
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
                    if isinstance(v, str):
                        cleaned = v.replace('%', '').replace('$', '').replace(',', '').strip()
                        try:
                            vals.add(float(cleaned))
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
        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp: r_fmt += 0.5
        if '<answer>' in resp: r_fmt += 0.5
        results.append(r_acc + r_fmt)
    return results


def reward_chartvr(completions, answer, csv_path="", question="",
                   sigma=0.10, w_acc=0.5, w_proc=0.3, w_fmt=0.2,
                   verifier=None, **kwargs):
    """ChartVCR: R_accuracy + R_process(structured LLM) + R_format.
    Batch async verifier calls (6 concurrent) for speed."""
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    if isinstance(csv_path, str):
        csv_path = [csv_path] * len(completions)
    if isinstance(question, str):
        question = [question] * len(completions)

    from code.rewards.llm_verifier import csv_to_text, compute_process_reward as compute_proc_reward, extract_reasoning
    import asyncio

    # Prepare all data
    resps = [comp[0]["content"] if comp else "" for comp in completions]

    # Batch verifier calls (async, 6 concurrent)
    extractions = [None] * len(resps)
    if verifier:
        async def batch_verify():
            sem = asyncio.Semaphore(4)
            async def verify_one(idx):
                csv, q, resp = csv_path[idx], question[idx], resps[idx]
                if not csv:
                    return idx, None, ""
                async with sem:
                    csv_text = csv_to_text(csv)
                    reasoning = extract_reasoning(resp)
                    ext, raw = await verifier.verify_single(csv_text, q, reasoning)
                    return idx, ext, reasoning
            tasks = [verify_one(i) for i in range(len(resps))]
            return await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        batch_results = loop.run_until_complete(batch_verify())
        loop.close()
        reasoning_texts = [""] * len(resps)
        for idx, ext, reasoning in batch_results:
            extractions[idx] = ext
            reasoning_texts[idx] = reasoning
    else:
        reasoning_texts = [""] * len(resps)

    # Score each completion
    results = []
    r_acc_list, r_proc_list, r_fmt_list = [], [], []
    total_vclaims, total_cclaims, claims_zero_count = 0, 0, 0
    for i, (resp, gold, csv, q) in enumerate(zip(resps, answer, csv_path, question)):
        ans = extract_answer(resp)
        r_acc = 1.0 if relaxed_match(ans, gold) else 0.0

        if extractions[i] is not None:
            r_proc, _ = compute_proc_reward(extractions[i], reasoning_texts[i])
        elif csv:
            r_proc = compute_process_reward(resp, csv, sigma)
        else:
            r_proc = 0.0

        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.25
        if '<answer>' in resp:
            r_fmt += 0.25
        lines = [l for l in resp.split('<think>')[1].split('</think>')[0].split('\n') if l.strip()] if '<think>' in resp and '</think>' in resp else []
        if len(lines) >= 3:
            r_fmt += 0.5
        r_acc_list.append(r_acc)
        r_proc_list.append(r_proc)
        r_fmt_list.append(min(r_fmt, 1.0))
        n_vclaims = len(extractions[i].value_claims) if extractions[i] else 0
        n_cclaims = len(extractions[i].computation_claims) if extractions[i] else 0
        total_vclaims += n_vclaims
        total_cclaims += n_cclaims
        if n_vclaims == 0 and n_cclaims == 0:
            claims_zero_count += 1
        results.append(w_acc * r_acc + w_proc * r_proc + w_fmt * min(r_fmt, 1.0))

    # Log per-component metrics
    _metrics_batch_counter["n"] += 1
    import numpy as np
    _metrics_logger.info(json.dumps({
        "batch": _metrics_batch_counter["n"],
        "r_accuracy_mean": float(np.mean(r_acc_list)),
        "r_process_mean": float(np.mean(r_proc_list)),
        "r_format_mean": float(np.mean(r_fmt_list)),
        "r_total_mean": float(np.mean(results)),
        "num_value_claims": total_vclaims,
        "num_computation_claims": total_cclaims,
        "claims_zero_ratio": claims_zero_count / len(results) if results else 0,
        "n_completions": len(results),
    }))
    return results


def reward_chartvr_v2(completions, answer, csv_path="", question="",
                      w_acc=0.5, w_proc=0.3, w_fmt=0.2,
                      verifier=None, **kwargs):
    """ChartVCR V2: final-chain scoring + length penalty.
    Eliminates verbosity incentive from V1."""
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    if isinstance(csv_path, str):
        csv_path = [csv_path] * len(completions)
    if isinstance(question, str):
        question = [question] * len(completions)

    from code.rewards.llm_verifier import csv_to_text, compute_process_reward_v2 as compute_proc_v2, extract_reasoning
    import asyncio

    resps = [comp[0]["content"] if comp else "" for comp in completions]

    # Batch verifier calls
    extractions = [None] * len(resps)
    if verifier:
        async def batch_verify():
            sem = asyncio.Semaphore(4)
            async def verify_one(idx):
                csv, q, resp = csv_path[idx], question[idx], resps[idx]
                if not csv:
                    return idx, None, ""
                async with sem:
                    csv_text = csv_to_text(csv)
                    reasoning = extract_reasoning(resp)
                    ext, raw = await verifier.verify_single(csv_text, q, reasoning)
                    return idx, ext, reasoning
            tasks = [verify_one(i) for i in range(len(resps))]
            return await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        batch_results = loop.run_until_complete(batch_verify())
        loop.close()
        reasoning_texts = [""] * len(resps)
        for idx, ext, reasoning in batch_results:
            extractions[idx] = ext
            reasoning_texts[idx] = reasoning
    else:
        reasoning_texts = [""] * len(resps)

    results = []
    r_acc_list, r_proc_list, r_fmt_list, len_penalty_list = [], [], [], []
    total_vclaims, total_cclaims, claims_zero_count = 0, 0, 0
    for i, (resp, gold, csv, q) in enumerate(zip(resps, answer, csv_path, question)):
        ans = extract_answer(resp)
        r_acc = 1.0 if relaxed_match(ans, gold) else 0.0

        if extractions[i] is not None:
            r_proc, _ = compute_proc_v2(extractions[i], reasoning_texts[i])
        else:
            r_proc = 0.0

        # R_format: no line-count bonus (anti-verbosity)
        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.5
        if '<answer>' in resp:
            r_fmt += 0.5

        # Length penalty on thinking tokens
        think_match = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
        if think_match:
            think_tokens = len(think_match.group(1).split())
            length_penalty = max(0, min(1, (think_tokens - 300) / 300)) * 0.2
        else:
            length_penalty = 0

        r_acc_list.append(r_acc)
        r_proc_list.append(r_proc)
        r_fmt_list.append(r_fmt)
        len_penalty_list.append(length_penalty)
        n_vclaims = len(extractions[i].value_claims) if extractions[i] else 0
        n_cclaims = len(extractions[i].computation_claims) if extractions[i] else 0
        total_vclaims += n_vclaims
        total_cclaims += n_cclaims
        if n_vclaims == 0 and n_cclaims == 0:
            claims_zero_count += 1
        total = w_acc * r_acc + w_proc * r_proc + w_fmt * r_fmt - length_penalty
        results.append(max(0.0, total))

    # Log per-component metrics
    _metrics_batch_counter["n"] += 1
    import numpy as np
    _metrics_logger.info(json.dumps({
        "batch": _metrics_batch_counter["n"],
        "reward_version": "v2",
        "r_accuracy_mean": float(np.mean(r_acc_list)),
        "r_process_mean": float(np.mean(r_proc_list)),
        "r_format_mean": float(np.mean(r_fmt_list)),
        "length_penalty_mean": float(np.mean(len_penalty_list)),
        "r_total_mean": float(np.mean(results)),
        "num_value_claims": total_vclaims,
        "num_computation_claims": total_cclaims,
        "claims_zero_ratio": claims_zero_count / len(results) if results else 0,
        "n_completions": len(results),
    }))
    return results


def reward_geval(completions, answer, csv_path="", question="",
                 w_acc=0.5, w_proc=0.3, w_fmt=0.2,
                 verifier=None, **kwargs):
    """G-Eval ablation (A2): holistic LLM scoring instead of structured extraction."""
    if isinstance(answer, str):
        answer = [answer] * len(completions)
    if isinstance(csv_path, str):
        csv_path = [csv_path] * len(completions)
    if isinstance(question, str):
        question = [question] * len(completions)
    results = []
    for comp, gold, csv, q in zip(completions, answer, csv_path, question):
        resp = comp[0]["content"] if comp else ""
        ans = extract_answer(resp)
        r_acc = 1.0 if relaxed_match(ans, gold) else 0.0

        if verifier and csv:
            from code.rewards.llm_verifier import compute_geval_reward
            r_proc = compute_geval_reward(resp, csv, q, verifier)
        else:
            r_proc = 0.0

        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.25
        if '<answer>' in resp:
            r_fmt += 0.25
        lines = [l for l in resp.split('<think>')[1].split('</think>')[0].split('\n') if l.strip()] if '<think>' in resp and '</think>' in resp else []
        if len(lines) >= 3:
            r_fmt += 0.5
        results.append(w_acc * r_acc + w_proc * r_proc + w_fmt * min(r_fmt, 1.0))
    return results


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", choices=["outcome_only", "chartvr", "chartvr_v2", "geval"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=os.path.join(BASE, "models/qwen3.5-4b"))
    parser.add_argument("--data_dir", default=os.path.join(BASE, "data/chartqa"))
    parser.add_argument("--data_json", default=None, help="Direct JSON file path (skips ChartQA loading)")
    parser.add_argument("--sigma", type=float, default=0.10)
    # Ablation controls
    parser.add_argument("--w_process", type=float, default=0.3)
    parser.add_argument("--disable_causal", action="store_true")
    # Verifier server
    parser.add_argument("--verifier_url", default="http://10.1.211.148:8000/v1")
    parser.add_argument("--verifier_model", default="Qwen3.5-397B-A17B-FP8")
    args = parser.parse_args()

    # Setup per-component metrics logger
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_log_path = os.path.join(args.output_dir, "component_metrics.jsonl")
    _mh = logging.FileHandler(metrics_log_path)
    _mh.setFormatter(logging.Formatter("%(message)s"))
    _metrics_logger.addHandler(_mh)
    _metrics_logger.info(json.dumps({"event": "init", "reward_type": args.reward_type}))

    # Load dataset
    if args.data_json:
        # Load our custom dataset directly
        from datasets import Dataset
        from PIL import Image as PILImage
        with open(args.data_json) as f:
            raw_data = json.load(f)
        SYSTEM_TEXT = (
            "You are an expert chart analyst. "
            "After your reasoning, put your final answer inside "
            "<answer></answer> tags with ONLY the core value or keyword. "
            "Example: <answer>42</answer>"
        )
        flat_samples = []
        for i, s in enumerate(raw_data):
            img_path = s.get("image", "")
            if not os.path.exists(img_path):
                continue
            img = PILImage.open(img_path).convert("RGB")
            max_side = max(img.size)
            if max_side > 1024:
                scale = 1024 / max_side
                img = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)), PILImage.LANCZOS)
            flat_samples.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_TEXT},
                    {"role": "user", "content": f"Question: {s['problem']}"},
                ],
                "images": [img],
                "answer": s["solution"],
                "csv_path": s.get("csv_path", ""),
                "question": s["problem"],
            })
            if (i+1) % 1000 == 0:
                print(f"  Loaded {i+1}/{len(raw_data)} images")
        print(f"  All {len(flat_samples)} images loaded")
        dataset = Dataset.from_list(flat_samples)
    else:
        dataset = load_grpo_dataset(args.data_dir)
    print(f"GRPO dataset: {len(dataset)} samples")

    # Initialize verifier if needed
    verifier = None
    if args.reward_type in ("chartvr", "chartvr_v2", "geval"):
        from code.rewards.llm_verifier import VerifierClient
        verifier = VerifierClient(
            base_url=args.verifier_url,
            model_id=args.verifier_model,
        )
        print(f"Verifier: {args.verifier_model} @ {args.verifier_url}")

    # Select reward function
    if args.reward_type == "outcome_only":
        reward_fn = reward_outcome_only
    elif args.reward_type == "geval":
        def reward_fn(completions, answer, csv_path="", question="", **kw):
            return reward_geval(
                completions, answer, csv_path, question,
                verifier=verifier, **kw,
            )
    elif args.reward_type == "chartvr_v2":
        def reward_fn(completions, answer, csv_path="", question="", **kw):
            return reward_chartvr_v2(
                completions, answer, csv_path, question,
                w_proc=args.w_process,
                w_acc=0.5 + (0.3 - args.w_process),
                verifier=verifier,
                **kw,
            )
    else:
        # chartvr: structured extraction + deterministic scoring
        def reward_fn(completions, answer, csv_path="", question="", **kw):
            return reward_chartvr(
                completions, answer, csv_path, question,
                sigma=args.sigma,
                w_proc=args.w_process,
                w_acc=0.5 + (0.3 - args.w_process),
                verifier=verifier,
                **kw,
            )

    print(f"Reward type: {args.reward_type}")

    vllm_port = int(os.environ.get("VLLM_PORT", "9100"))
    num_train_gpus = int(os.environ.get("NUM_TRAIN_GPUS", "8"))

    config = GRPOConfig(
        output_dir=args.output_dir,
        # vLLM server mode (server on separate GPUs)
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="0.0.0.0",
        vllm_server_port=vllm_port,
        vllm_server_timeout=300.0,
        # Generation
        num_generations=4,
        max_completion_length=1024,
        generation_batch_size=num_train_gpus * 4,  # Must equal global_batch = num_gpus * per_device_batch
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        warmup_steps=50,
        # Memory optimization
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        use_liger_kernel=False,
        # Logging & saving
        logging_steps=5,
        save_steps=200,
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
