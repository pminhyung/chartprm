# ChartVCR: Research Execution Plan for Claude Code Agent
# Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO
# ═══════════════════════════════════════════════════════════

## META-INSTRUCTIONS FOR AGENT

You are executing a complete ML research project. This document is your single source of truth.
Follow it TOP-TO-BOTTOM. Do not skip steps. Do not reinterpret instructions.
Every decision has been made. Your job is to IMPLEMENT, not to redesign.

When you encounter a CHECKPOINT, verify the stated condition before proceeding.
When you encounter a DECISION POINT, follow the stated fallback.
When you encounter a BUG, fix it and document the fix in `logs/bugfixes.md`.

All work happens in `/home/user/chartvr/`. All results go to `/home/user/chartvr/results/`.
The final paper lives at `/home/user/chartvr/paper/main.tex`.

---

## TABLE OF CONTENTS

```
PHASE 0: Environment Setup & Data Preparation        (Day 1-2)
PHASE 1: Pilot Study — Verifier Accuracy Validation   (Day 3-4)
PHASE 2: SFT Warm-up (BigCharts-R1 Reproduction)      (Day 5-7)
PHASE 3: GRPO Baseline Reproduction                    (Day 8-10)
PHASE 4: ChartVCR Implementation                       (Day 11-13)
PHASE 5: ChartVCR-GRPO Training                        (Day 14-16)
PHASE 6: Evaluation on All Benchmarks                  (Day 17-18)
PHASE 7: Ablation Experiments                          (Day 19-22)
PHASE 8: Analysis & Visualization                      (Day 23-24)
PHASE 9: Paper Writing                                 (Day 25-28)
PHASE 10: Camera-Ready Polish                          (Day 29-30)
```

---
---

## PHASE 0: ENVIRONMENT SETUP & DATA PREPARATION

### 0.1 Directory Structure

```bash
mkdir -p /home/user/chartvr/{
  code/{rewards,verifier,evaluation,data_processing,analysis},
  data/{chartqa,plotqa,dvqa,figureqa,bigcharts,charxiv,chartqa_pro,chartmuseum},
  checkpoints/{sft,grpo_baseline,grpo_cvr},
  results/{main_table,ablations,analysis,pilot},
  logs,
  paper/{figures,tables,sections}
}
```

### 0.2 Environment Setup

```bash
# Base environment
conda create -n chartvr python=3.10 -y
conda activate chartvr

# Clone BigCharts-R1 (our foundation codebase)
git clone https://github.com/ServiceNow/BigCharts-R1.git code/bigcharts-r1
cd code/bigcharts-r1
bash setup.sh

# Additional dependencies
pip install vllm>=0.6.0        # For verifier serving
pip install pandas numpy scipy  # For reward computation
pip install matplotlib seaborn  # For analysis plots
pip install texttable           # For LaTeX table generation

# LLaMA-Factory for SFT
git clone https://github.com/hiyouga/LLaMA-Factory.git code/llama-factory
cd code/llama-factory
pip install -e ".[torch,metrics]"
```

### 0.3 Model Downloads

```bash
# Policy model (finetune target)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen25vl-7b

# Verifier model (DIFFERENT family from policy - critical: no circular evaluation)
huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir models/phi4-mini
# Phi-4-mini: 3.8B params, different architecture family from Qwen
# If Phi-4-mini unavailable, fallback: google/gemma-2-2b-it

# Backup verifier (if Phi-4-mini pilot study fails)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen25-3b
# NOTE: Using Qwen2.5-3B as verifier for Qwen2.5-VL-7B policy is acceptable
# because verifier is TEXT-ONLY (no vision encoder) and different param size
```

### 0.4 Dataset Downloads

```bash
# ═══════════════════════════════════════════════════
# CRITICAL: Verify data table availability for each dataset
# Only datasets WITH data tables are used for GRPO training
# Datasets WITHOUT data tables are EVALUATION-ONLY
# ═══════════════════════════════════════════════════

# --- TRAINING DATASETS (have data tables) ---

# ChartQA (primary RL data + evaluation)
# Source: https://huggingface.co/datasets/ahmed-masry/ChartQA
# Structure: train/test with png/, tables/, annotations/, *.json
huggingface-cli download ahmed-masry/ChartQA --local-dir data/chartqa
# VERIFY after download:
#   ls data/chartqa/train/tables/ | head -5    → must show CSV files
#   ls data/chartqa/train/png/ | head -5       → must show PNG files
#   cat data/chartqa/train/train_human.json | python -c "import json,sys; d=json.load(sys.stdin); print(len(d))"
#   → Expected: ~7398 for human, ~20901 for augmented

# PlotQA (supplementary RL data)
# Source: https://github.com/NiteshMethworx/PlotQA
# Has ground-truth tables (synthetic)
# Download v2 test split: 1K random sample for RL
# AGENT NOTE: PlotQA is very large (>8GB). Download only what we need.
# We will create a 1K subset in PHASE 0.6

# DVQA (supplementary RL data)
# Source: https://github.com/kushalkafle/DVQA_dataset
# Synthetic bar charts with exact tables
# Download test split: 1K random sample for RL

# BigCharts SFT data
# Source: https://huggingface.co/datasets/ServiceNow/BigCharts
# This is the SFT dataset from BigCharts-R1
huggingface-cli download ServiceNow/BigCharts --local-dir data/bigcharts

# --- EVALUATION-ONLY DATASETS (no data tables) ---

# CharXiv
git clone https://github.com/princeton-nlp/CharXiv.git data/charxiv_repo
# Eval script is in data/charxiv_repo/
# Images and questions from HuggingFace
huggingface-cli download princeton-nlp/CharXiv --local-dir data/charxiv

# ChartQA-Pro
huggingface-cli download ahmed-masry/ChartQAPro --local-dir data/chartqa_pro

# ChartMuseum (if available on HF; otherwise arxiv supplementary)
# Check: huggingface-cli download ChartMuseum/ChartMuseum --local-dir data/chartmuseum
# If not available, skip — use remaining 5 benchmarks
```

### 0.5 Data Table Verification (MANDATORY)

```python
# File: code/data_processing/verify_tables.py
"""
Run this BEFORE any training. Confirms data tables exist and are parseable.
CHECKPOINT: All assertions must pass.
"""
import os, json, pandas as pd, glob

def verify_chartqa():
    tables_dir = "data/chartqa/train/tables/"
    png_dir = "data/chartqa/train/png/"
    
    csvs = glob.glob(f"{tables_dir}/*.csv")
    pngs = glob.glob(f"{png_dir}/*.png")
    
    assert len(csvs) > 4000, f"Expected >4000 CSVs, got {len(csvs)}"
    assert len(pngs) > 4000, f"Expected >4000 PNGs, got {len(pngs)}"
    
    # Verify CSV-PNG alignment
    csv_names = {os.path.splitext(os.path.basename(f))[0] for f in csvs}
    png_names = {os.path.splitext(os.path.basename(f))[0] for f in pngs}
    overlap = csv_names & png_names
    assert len(overlap) > 3500, f"CSV-PNG overlap only {len(overlap)}"
    
    # Verify CSV is parseable
    sample_csv = csvs[0]
    df = pd.read_csv(sample_csv)
    assert len(df) > 0, f"Empty CSV: {sample_csv}"
    assert len(df.columns) >= 2, f"Too few columns: {sample_csv}"
    
    # Count numeric values per table
    numeric_counts = []
    for csv_path in csvs[:100]:
        df = pd.read_csv(csv_path)
        n_numeric = sum(
            pd.to_numeric(df[col], errors='coerce').notna().sum() 
            for col in df.columns
        )
        numeric_counts.append(n_numeric)
    
    avg_numeric = sum(numeric_counts) / len(numeric_counts)
    print(f"ChartQA: {len(csvs)} tables, {len(pngs)} images, "
          f"avg {avg_numeric:.1f} numeric values per table")
    
    return True

def verify_chartqa_qa_format():
    """Verify QA JSON format matches expected structure."""
    with open("data/chartqa/train/train_human.json") as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) > 2000
    
    sample = data[0]
    assert "question" in sample or "query" in sample
    assert "answer" in sample or "label" in sample
    assert "imgname" in sample or "image" in sample
    
    print(f"ChartQA Human train: {len(data)} QA pairs")
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Sample: {json.dumps(sample, indent=2)[:500]}")
    
    return sample.keys()  # Return keys for format adaptation

if __name__ == "__main__":
    verify_chartqa()
    keys = verify_chartqa_qa_format()
    print("\n✅ All data verifications passed")
    print(f"QA format keys: {list(keys)}")
```

```bash
# RUN:
python code/data_processing/verify_tables.py
# CHECKPOINT: Must print "All data verifications passed"
# If it fails, check download completeness and retry
```

### 0.6 Prepare GRPO Training Data

```python
# File: code/data_processing/prepare_rl_data.py
"""
Convert all training datasets to unified GRPO format.
Each sample: {"problem": str, "solution": str, "image": str, "csv_path": str}

CRITICAL: csv_path must point to a real, parseable CSV file.
"""
import json, os, glob, random, pandas as pd

def load_chartqa_train():
    """Load ChartQA train split with CSV paths."""
    samples = []
    
    for split_file in ["train_human.json", "train_augmented.json"]:
        path = f"data/chartqa/train/{split_file}"
        with open(path) as f:
            data = json.load(f)
        
        for item in data:
            # Adapt to actual key names (verified in 0.5)
            question = item.get("question") or item.get("query")
            answer = str(item.get("answer") or item.get("label"))
            image = item.get("imgname") or item.get("image")
            
            # Derive CSV path from image name
            img_base = os.path.splitext(image)[0]
            csv_path = f"data/chartqa/train/tables/{img_base}.csv"
            image_path = f"data/chartqa/train/png/{image}"
            
            # Only include if CSV exists
            if os.path.exists(csv_path) and os.path.exists(image_path):
                samples.append({
                    "problem": question,
                    "solution": answer,
                    "image": image_path,
                    "csv_path": csv_path
                })
    
    print(f"ChartQA train: {len(samples)} samples with valid CSV")
    return samples

def prepare_grpo_data():
    """Create unified GRPO training data."""
    all_samples = []
    
    # ChartQA (primary)
    chartqa = load_chartqa_train()
    all_samples.extend(chartqa)
    
    # PlotQA subset (1K) — implement similar loader if PlotQA downloaded
    # DVQA subset (1K) — implement similar loader if DVQA downloaded
    # FigureQA subset (1K) — implement similar loader if FigureQA downloaded
    
    # For now, if supplementary datasets not available,
    # ChartQA alone (~4800 samples) is sufficient.
    # BigCharts-R1 used ChartQA + 1K each from PlotQA/DVQA/FigureQA = ~7.8K
    # Chart-RL showed 10 complex > 6000 simple, so ChartQA alone is defensible.
    
    random.shuffle(all_samples)
    
    output_path = "data/grpo_train.json"
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"Total GRPO data: {len(all_samples)} → {output_path}")
    return all_samples

if __name__ == "__main__":
    prepare_grpo_data()
```

```bash
python code/data_processing/prepare_rl_data.py
# CHECKPOINT: data/grpo_train.json exists with >4000 samples
# Each sample has csv_path pointing to existing file
```

---
---

## PHASE 1: PILOT STUDY — VERIFIER ACCURACY VALIDATION

### Purpose
Before building the entire ChartVCR framework, validate that:
1. The verifier LLM can classify sentence-level correctness (correct/incorrect/not_verifiable)
2. The causal attribution (source vs propagated error) is separable
3. The number extraction regex works on real CoT outputs

### 1.1 Generate Pilot CoT Traces

```python
# File: code/verifier/generate_pilot_cots.py
"""
Generate 100 CoT traces from Qwen2.5-VL-7B on ChartQA train samples.
These will be manually labeled + verifier-labeled for comparison.
"""
import json, random, torch
from transformers import AutoProcessor, AutoModelForCausalLM

def generate_cots(n=100):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "models/qwen25vl-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("models/qwen25vl-7b")
    
    # Load ChartQA samples (only computation-requiring questions)
    with open("data/grpo_train.json") as f:
        all_data = json.load(f)
    
    # Filter for questions likely requiring computation
    COMPUTE_KEYWORDS = [
        "ratio", "difference", "sum", "average", "total",
        "how many", "percentage", "change", "increase", "decrease",
        "more than", "less than", "compare", "between"
    ]
    
    compute_samples = [
        s for s in all_data
        if any(kw in s["problem"].lower() for kw in COMPUTE_KEYWORDS)
    ]
    
    selected = random.sample(compute_samples, min(n, len(compute_samples)))
    
    PROMPT_TEMPLATE = """Look at this chart and answer the question step by step.
Show your reasoning clearly, with each step on a new line.

Question: {question}

Think step by step, then give your final answer."""
    
    results = []
    for sample in selected:
        # Generate CoT
        prompt = PROMPT_TEMPLATE.format(question=sample["problem"])
        # [Agent: implement actual VLM inference with image loading]
        # Use the model's chat template with <think> tags if applicable
        
        # Placeholder structure:
        response = generate_with_image(model, processor, sample["image"], prompt)
        
        results.append({
            "question": sample["problem"],
            "gold_answer": sample["solution"],
            "image": sample["image"],
            "csv_path": sample["csv_path"],
            "model_response": response
        })
    
    with open("results/pilot/cot_traces.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} CoT traces")
    return results
```

### 1.2 Extract Sentences and Numbers

```python
# File: code/rewards/sentence_parser.py
"""
Core utility: parse CoT into sentences and extract numbers.
This is used by BOTH the pilot study AND the actual reward function.
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedSentence:
    text: str
    index: int
    numbers: List[float]           # All numbers found in this sentence
    has_computation: bool          # Contains arithmetic pattern (a op b = c)
    computation_result: Optional[Tuple[float, float, str, float]]  
    # (operand1, operand2, operator, stated_result)

def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    # Match: 45.3, 1,234, 45.3%, $45.3M, 45.3 billion, etc.
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return numbers

def extract_computation(text: str) -> Optional[Tuple[float, float, str, float]]:
    """
    Extract explicit arithmetic: "45.3 - 38.1 = 7.2"
    Returns (a, b, op, result) or None.
    """
    pattern = r'([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([+\-*/×÷])\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*[=≈]\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, text)
    if match:
        try:
            a = float(match.group(1).replace(",", ""))
            op = match.group(2)
            if op == '×': op = '*'
            if op == '÷': op = '/'
            b = float(match.group(3).replace(",", ""))
            result = float(match.group(4).replace(",", ""))
            return (a, b, op, result)
        except ValueError:
            return None
    return None

def parse_cot_to_sentences(response: str) -> List[ParsedSentence]:
    """
    Split CoT into sentences and annotate each.
    
    Splitting strategy:
    - Primary: split on "\n\n" (BigCharts-R1 step delimiter)
    - Secondary: if <think> tags present, extract content first
    - Tertiary: split on "\n" if "\n\n" produces too few segments
    """
    # Extract think content if present
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        content = think_match.group(1).strip()
    else:
        # Try to get content before final answer
        content = response.strip()
    
    # Split into sentences
    sentences = content.split("\n\n")
    if len(sentences) < 3:
        sentences = content.split("\n")
    
    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    parsed = []
    for i, sent in enumerate(sentences):
        numbers = extract_numbers(sent)
        computation = extract_computation(sent)
        parsed.append(ParsedSentence(
            text=sent,
            index=i,
            numbers=numbers,
            has_computation=computation is not None,
            computation_result=computation
        ))
    
    return parsed
```

### 1.3 Test Verifier Accuracy

```python
# File: code/verifier/pilot_test.py
"""
Test verifier LLM on pilot CoT sentences.

Protocol:
1. For each sentence with numbers >= 2, ask verifier to classify
2. Compare verifier labels with human labels (manual)
3. Report accuracy

HUMAN LABELING INSTRUCTIONS (for researcher):
Open results/pilot/sentences_for_labeling.jsonl
For each sentence, assign:
- "correct": the math/value extraction in this sentence is right
- "incorrect": the math/value extraction is wrong
- "not_verifiable": no math/extraction to verify (pure text reasoning)
Save as results/pilot/human_labels.jsonl
"""
import json
from vllm import LLM, SamplingParams

EXTRACTION_VERIFY_PROMPT = """Given this data table from a chart:
{csv_content}

A model analyzing this chart wrote:
"{sentence}"

Does this sentence correctly read or extract a value from the data table?
If there is no value extraction in this sentence, respond "not_verifiable".
If a value is extracted and matches the table (within 10% tolerance), respond "correct".
If a value is extracted but does not match the table, respond "incorrect".

Respond with ONLY one word: correct, incorrect, or not_verifiable"""

COMPUTATION_VERIFY_PROMPT = """A model wrote this reasoning step:
"{sentence}"

Does this sentence contain a mathematical calculation? 
If yes, is the calculation mathematically correct?

Respond with ONLY one word:
- "correct" if the calculation is present and mathematically correct
- "incorrect" if the calculation is present but wrong
- "not_verifiable" if there is no calculation to verify"""

def run_verifier_pilot():
    # Load pilot CoT traces
    with open("results/pilot/cot_traces.json") as f:
        traces = json.load(f)
    
    # Load verifier
    verifier = LLM(
        model="models/phi4-mini",  
        # If Phi-4-mini not available, use "models/qwen25-3b"
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    sampling_params = SamplingParams(
        temperature=0, max_tokens=10, stop=["\n"]
    )
    
    all_sentences = []
    
    for trace in traces:
        sentences = parse_cot_to_sentences(trace["model_response"])
        csv_content = open(trace["csv_path"]).read()[:1000]  # Truncate large CSVs
        
        for sent in sentences:
            if len(sent.numbers) == 0:
                label_type = "skip"
                prompt = None
            elif len(sent.numbers) >= 2 and sent.has_computation:
                label_type = "computation"
                prompt = COMPUTATION_VERIFY_PROMPT.format(sentence=sent.text)
            elif len(sent.numbers) >= 1:
                label_type = "extraction"
                prompt = EXTRACTION_VERIFY_PROMPT.format(
                    csv_content=csv_content, sentence=sent.text
                )
            else:
                label_type = "skip"
                prompt = None
            
            all_sentences.append({
                "trace_idx": traces.index(trace),
                "sentence_idx": sent.index,
                "sentence": sent.text,
                "numbers": sent.numbers,
                "label_type": label_type,
                "prompt": prompt,
                "csv_path": trace["csv_path"]
            })
    
    # Run verifier on all prompts
    prompts = [s["prompt"] for s in all_sentences if s["prompt"]]
    outputs = verifier.generate(prompts, sampling_params)
    
    # Map results back
    prompt_idx = 0
    for sent_data in all_sentences:
        if sent_data["prompt"]:
            raw_output = outputs[prompt_idx].outputs[0].text.strip().lower()
            # Normalize
            if "correct" in raw_output and "incorrect" not in raw_output:
                sent_data["verifier_label"] = "correct"
            elif "incorrect" in raw_output:
                sent_data["verifier_label"] = "incorrect"
            else:
                sent_data["verifier_label"] = "not_verifiable"
            prompt_idx += 1
        else:
            sent_data["verifier_label"] = "skip"
    
    # Save for human comparison
    with open("results/pilot/verifier_labels.json", "w") as f:
        json.dump(all_sentences, f, indent=2)
    
    # Save sentences needing human labels
    to_label = [s for s in all_sentences if s["verifier_label"] != "skip"]
    with open("results/pilot/sentences_for_labeling.jsonl", "w") as f:
        for s in to_label:
            f.write(json.dumps({
                "id": f"{s['trace_idx']}_{s['sentence_idx']}",
                "sentence": s["sentence"],
                "numbers": s["numbers"],
                "label_type": s["label_type"],
                "verifier_label": s["verifier_label"]
            }) + "\n")
    
    print(f"Total sentences: {len(all_sentences)}")
    print(f"  - Skip (no numbers): {sum(1 for s in all_sentences if s['verifier_label']=='skip')}")
    print(f"  - Correct: {sum(1 for s in all_sentences if s['verifier_label']=='correct')}")
    print(f"  - Incorrect: {sum(1 for s in all_sentences if s['verifier_label']=='incorrect')}")
    print(f"  - Not verifiable: {sum(1 for s in all_sentences if s['verifier_label']=='not_verifiable')}")
    print(f"\nHuman labeling needed: {len(to_label)} sentences")
    print(f"Saved to: results/pilot/sentences_for_labeling.jsonl")

if __name__ == "__main__":
    from code.rewards.sentence_parser import parse_cot_to_sentences
    run_verifier_pilot()
```

### 1.4 Pilot Evaluation (After Human Labels)

```python
# File: code/verifier/pilot_evaluate.py
"""
Compare verifier labels with human labels.

CHECKPOINT CONDITIONS:
- Overall accuracy >= 80%: PROCEED with LLM verifier
- Accuracy 70-80%: Try improved prompt, retest
- Accuracy < 70%: FALLBACK to rule-based only (no LLM verifier)

DECISION POINT: If fallback triggered, skip PHASE 4.3 (verifier serving)
and use rule-based rewards only in PHASE 4.
"""
import json
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_pilot():
    with open("results/pilot/verifier_labels.json") as f:
        verifier_data = json.load(f)
    
    # Load human labels
    human_labels = {}
    with open("results/pilot/human_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            human_labels[item["id"]] = item["human_label"]
    
    # Compare
    y_true, y_pred = [], []
    for s in verifier_data:
        sid = f"{s['trace_idx']}_{s['sentence_idx']}"
        if sid in human_labels and s["verifier_label"] != "skip":
            y_true.append(human_labels[sid])
            y_pred.append(s["verifier_label"])
    
    # 3-class accuracy
    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Binary: error detection (correct vs incorrect, excluding not_verifiable)
    binary_true = [1 if t == "incorrect" else 0 for t, p in zip(y_true, y_pred) if t != "not_verifiable"]
    binary_pred = [1 if p == "incorrect" else 0 for t, p in zip(y_true, y_pred) if t != "not_verifiable"]
    binary_acc = sum(1 for t, p in zip(binary_true, binary_pred) if t == p) / max(len(binary_true), 1)
    
    print(f"\n{'='*50}")
    print(f"PILOT STUDY RESULTS")
    print(f"{'='*50}")
    print(f"Total compared: {len(y_true)}")
    print(f"3-class accuracy: {overall_acc:.1%}")
    print(f"Binary error detection accuracy: {binary_acc:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # DECISION
    if overall_acc >= 0.80:
        print("\n✅ DECISION: PROCEED with LLM verifier")
        decision = "llm_verifier"
    elif overall_acc >= 0.70:
        print("\n⚠️ DECISION: Try improved prompt, then retest")
        decision = "retry_prompt"
    else:
        print("\n❌ DECISION: FALLBACK to rule-based only")
        decision = "rule_based_only"
    
    # Save decision
    with open("results/pilot/decision.json", "w") as f:
        json.dump({
            "overall_accuracy": overall_acc,
            "binary_accuracy": binary_acc,
            "decision": decision,
            "n_samples": len(y_true)
        }, f, indent=2)
    
    return decision
```

### 1.5 Number Flow Tracking (Rule-Based Causal Attribution)

```python
# File: code/rewards/causal_attribution.py
"""
Rule-based causal error attribution.
NO LLM calls. Pure Python logic.

This is used in Phase 2 of the reward computation:
Phase 1 (LLM): classify each sentence as correct/incorrect/not_verifiable
Phase 2 (this code): refine incorrect → source_error or propagated_error
"""
import math
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class AttributedSentence:
    text: str
    index: int
    numbers: List[float]
    raw_label: str                 # From Phase 1: correct/incorrect/not_verifiable
    causal_label: str              # Refined: correct/source_error/propagated_error/not_verifiable
    logic_score: float             # 1.0 if logic correct, 0.0 if not
    input_quality: float           # 0.0~1.0 based on upstream number accuracy
    sentence_reward: float         # = logic_score × input_quality

def numbers_match(a: float, b: float, tol: float = 0.05) -> bool:
    """Check if two numbers are approximately equal."""
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / abs(b) < tol

def value_accuracy_score(model_value: float, table_value: float, sigma: float = 0.10) -> float:
    """
    Continuous accuracy score using Gaussian decay.
    sigma=0.10 means 10% relative error gives score ~0.61
    
    This replaces binary 5% tolerance with smooth continuous reward.
    """
    if table_value == 0:
        return 1.0 if abs(model_value) < 0.01 else 0.0
    
    relative_error = abs(model_value - table_value) / abs(table_value)
    return math.exp(-0.5 * (relative_error / sigma) ** 2)

def find_closest_table_value(number: float, table_values: Set[float]) -> Tuple[float, float]:
    """Find the closest value in the table and return (closest_val, accuracy_score)."""
    if not table_values:
        return (0.0, 0.0)
    
    best_score = 0.0
    best_val = 0.0
    for tv in table_values:
        score = value_accuracy_score(number, tv)
        if score > best_score:
            best_score = score
            best_val = tv
    
    return (best_val, best_score)

def compute_causal_rewards(
    sentences: List[dict],  # Each: {text, index, numbers, raw_label, computation_correct}
    table_values: Set[float],
    sigma: float = 0.10
) -> List[AttributedSentence]:
    """
    Main causal reward computation.
    
    Algorithm:
    1. Track which numbers are "tainted" (produced by incorrect sentences)
    2. For each incorrect sentence:
       - If all its input numbers are fresh (from chart): source_error
       - If any input number is tainted: propagated_error
    3. Compute:
       - logic_score: for source_error → 0.0; for propagated → check computation
       - input_quality: Gaussian score of input numbers vs table
       - sentence_reward: logic_score × input_quality
    """
    
    tainted_numbers: Set[float] = set()     # Numbers produced by incorrect sentences
    number_quality: Dict[float, float] = {} # number → quality score
    
    results = []
    
    for sent in sentences:
        numbers = sent["numbers"]
        raw_label = sent["raw_label"]
        
        if raw_label == "not_verifiable":
            results.append(AttributedSentence(
                text=sent["text"], index=sent["index"], numbers=numbers,
                raw_label="not_verifiable", causal_label="not_verifiable",
                logic_score=1.0, input_quality=1.0, sentence_reward=-1.0  # excluded
            ))
            continue
        
        # Compute input_quality: how good are the numbers this sentence uses?
        if numbers:
            input_scores = []
            for n in numbers:
                n_rounded = round(n, 2)
                
                # Is this number tainted?
                is_tainted = any(
                    numbers_match(n, tn, 0.05) for tn in tainted_numbers
                )
                
                if is_tainted:
                    # Use the tainted number's quality from when it was produced
                    matching_quality = min(
                        (number_quality.get(round(tn, 2), 0.0) 
                         for tn in tainted_numbers 
                         if numbers_match(n, tn, 0.05)),
                        default=0.0
                    )
                    input_scores.append(matching_quality)
                else:
                    # Fresh from chart: compare to table
                    _, score = find_closest_table_value(n, table_values)
                    input_scores.append(score)
                    number_quality[n_rounded] = score
            
            input_quality = min(input_scores)  # Conservative: weakest link
        else:
            input_quality = 1.0
        
        if raw_label == "correct":
            # Logic correct, check input quality
            logic_score = 1.0
            causal = "correct"
        else:
            # raw_label == "incorrect"
            # Determine: source or propagated?
            uses_tainted = any(
                any(numbers_match(n, tn, 0.05) for tn in tainted_numbers)
                for n in numbers
            )
            
            if uses_tainted:
                causal = "propagated_error"
                # Logic might still be correct (e.g., arithmetic is right)
                if sent.get("computation_correct") is True:
                    logic_score = 1.0
                elif sent.get("computation_correct") is False:
                    logic_score = 0.0
                else:
                    logic_score = 0.5  # Unknown, give partial credit
            else:
                causal = "source_error"
                logic_score = 0.0
            
            # Mark all numbers this sentence produces as tainted
            for n in numbers:
                tainted_numbers.add(round(n, 2))
        
        sentence_reward = logic_score * input_quality
        
        results.append(AttributedSentence(
            text=sent["text"], index=sent["index"], numbers=numbers,
            raw_label=raw_label, causal_label=causal,
            logic_score=logic_score, input_quality=input_quality,
            sentence_reward=sentence_reward
        ))
    
    return results
```

---
---

## PHASE 2: SFT WARM-UP (BIGCHARTS-R1 REPRODUCTION)

### 2.1 SFT Training

```bash
# Use BigCharts-R1's SFT pipeline with LLaMA-Factory
# Config file: code/bigcharts-r1/configs/sft_config.yaml

cd code/llama-factory

# Create SFT config
cat > configs/chartvr_sft.yaml << 'EOF'
model_name_or_path: ../../models/qwen25vl-7b
stage: sft
do_train: true
dataset: bigcharts_sft
template: qwen2_vl
output_dir: ../../checkpoints/sft
overwrite_output_dir: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 2.0e-5
num_train_epochs: 1
warmup_ratio: 0.1
lr_scheduler_type: cosine
logging_steps: 10
save_steps: 500
bf16: true
gradient_checkpointing: true
max_length: 4096
EOF

# Launch training (8×A100-40GB)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 src/train.py configs/chartvr_sft.yaml
```

**Expected duration:** 24-30 hours on 8×A100-40GB
**Expected output:** checkpoints/sft/

### 2.2 SFT Checkpoint Validation

```python
# File: code/evaluation/quick_eval.py
"""
Quick evaluation on ChartQA test to validate SFT checkpoint.
CHECKPOINT: ChartQA-Human accuracy should be 78-84%.
If below 75%, SFT training has a problem.
"""
# [Agent: implement using BigCharts-R1's eval script]
# python code/bigcharts-r1/src/eval/eval_bigcharts_r1.py \
#   --images-path data/chartqa/test/png \
#   --data-path data/chartqa/test/test_human.json \
#   --target-path results/sft_eval.json \
#   --model-path checkpoints/sft \
#   --accuracy-mode relaxed_accuracy \
#   --batch-size 4
```

---
---

## PHASE 3: GRPO BASELINE REPRODUCTION

### 3.1 Baseline GRPO (Outcome-Only Reward)

```bash
# BigCharts-R1 GRPO with original reward: R_accuracy + R_format
# This is our PRIMARY BASELINE for comparison

cd code/bigcharts-r1

# Modify rl_data.yaml to point to our data
cat > data_config/rl_data.yaml << 'EOF'
- json_path: ../../data/grpo_train.json
  image_root: ../../
EOF

# Modify run script for our hardware (A100-40GB vs H100-80GB)
# Key changes: reduce per_device_batch_size, enable gradient_checkpointing

bash src/open-r1-multimodal/run_grpo_bigcharts_job.sh
# Internal params: lr=1e-6, batch=8, G=8 candidates, epoch=1
```

**Expected duration:** 24 hours on 8×A100-40GB

### 3.2 Baseline Evaluation

```bash
# Evaluate on ChartQA test
python src/eval/eval_bigcharts_r1.py \
  --model-path ../../checkpoints/grpo_baseline \
  --images-path ../../data/chartqa/test/png \
  --data-path ../../data/chartqa/test/test_human.json \
  --target-path ../../results/main_table/baseline_chartqa_human.json \
  --accuracy-mode relaxed_accuracy

# CHECKPOINT: ChartQA-Human should be 82-86% (BigCharts-R1-7B reported ~84%)
# If more than 4% below BigCharts-R1 reported numbers, debug training
```

---
---

## PHASE 4: ChartVCR IMPLEMENTATION

### 4.1 Core Reward Function

```python
# File: code/rewards/chart_vcr_reward.py
"""
ChartVCR: Chart-Verifiable Causal Reward

THE MAIN CONTRIBUTION. This file implements the complete reward function.
"""
import json, re, math, pandas as pd
from typing import Optional, List, Set
from code.rewards.sentence_parser import parse_cot_to_sentences, extract_numbers
from code.rewards.causal_attribution import (
    compute_causal_rewards, value_accuracy_score, find_closest_table_value
)

class ChartVCRReward:
    """
    Complete ChartVCR reward computation.
    
    Usage:
        reward_fn = ChartVCRReward(verifier=verifier_model, sigma=0.10)
        score = reward_fn(response, gold_answer, csv_path)
    """
    
    def __init__(self, verifier=None, sigma=0.10, 
                 w_accuracy=0.5, w_process=0.3, w_format=0.2):
        """
        Args:
            verifier: vLLM model for sentence verification (None = rule-based only)
            sigma: Gaussian width for value tolerance (0.10 = 10% decay)
            w_accuracy: weight for final answer accuracy
            w_process: weight for process (sentence-level) reward
            w_format: weight for format compliance
        """
        self.verifier = verifier
        self.sigma = sigma
        self.w_accuracy = w_accuracy
        self.w_process = w_process
        self.w_format = w_format
    
    def _load_table_values(self, csv_path: str) -> Set[float]:
        """Extract all numeric values from a chart's data table."""
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return set()
        
        values = set()
        for col in df.columns:
            for val in df[col]:
                try:
                    values.add(float(val))
                except (ValueError, TypeError):
                    pass
        return values
    
    def _compute_accuracy_reward(self, response: str, gold: str) -> float:
        """Standard relaxed accuracy (same as BigCharts-R1)."""
        # Extract answer from response
        answer = self._extract_answer(response)
        if answer is None:
            return 0.0
        
        # Try numeric comparison first
        try:
            pred_num = float(re.sub(r'[,%$]', '', answer))
            gold_num = float(re.sub(r'[,%$]', '', gold))
            if gold_num == 0:
                return 1.0 if abs(pred_num) < 0.01 else 0.0
            return 1.0 if abs(pred_num - gold_num) / abs(gold_num) <= 0.05 else 0.0
        except ValueError:
            pass
        
        # String comparison (case-insensitive, strip)
        return 1.0 if answer.strip().lower() == gold.strip().lower() else 0.0
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract final answer from response."""
        # Try boxed format
        boxed = re.search(r'\\boxed\{(.*?)\}', response)
        if boxed:
            return boxed.group(1)
        
        # Try answer tags
        answer_tag = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_tag:
            return answer_tag.group(1).strip()
        
        # Last line heuristic
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else None
    
    def _compute_format_reward(self, response: str) -> float:
        """Format compliance reward."""
        score = 0.0
        if "<think>" in response and "</think>" in response:
            score += 0.5
        if "\\boxed{" in response or "<answer>" in response:
            score += 0.3
        
        # Bonus for multi-step reasoning (anti-shortcut)
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            steps = think_match.group(1).strip().split('\n\n')
            steps = [s for s in steps if s.strip()]
            if len(steps) >= 3:
                score += 0.2
        
        return min(score, 1.0)
    
    def _compute_process_reward(self, response: str, csv_path: str) -> float:
        """
        Process reward with causal attribution.
        This is the CORE NOVELTY.
        """
        table_values = self._load_table_values(csv_path)
        if not table_values:
            return 0.0  # Can't verify without table
        
        # Parse CoT into sentences
        parsed = parse_cot_to_sentences(response)
        if not parsed:
            return 0.0
        
        # Phase 1: Verify each sentence
        sentence_data = []
        for sent in parsed:
            if len(sent.numbers) == 0:
                raw_label = "not_verifiable"
                comp_correct = None
            elif sent.has_computation and sent.computation_result:
                # Verify computation: a op b = c?
                a, b, op, stated = sent.computation_result
                try:
                    if op == '+': expected = a + b
                    elif op == '-': expected = a - b
                    elif op == '*': expected = a * b
                    elif op == '/': expected = a / b if b != 0 else float('inf')
                    else: expected = None
                    
                    if expected is not None:
                        comp_correct = abs(stated - expected) / max(abs(expected), 1e-10) < 0.05
                    else:
                        comp_correct = None
                except:
                    comp_correct = None
                
                # Check if extracted values are in table
                in_table = any(
                    find_closest_table_value(n, table_values)[1] > 0.5
                    for n in sent.numbers[:2]  # Check operands
                )
                
                if comp_correct is True and in_table:
                    raw_label = "correct"
                elif comp_correct is False:
                    raw_label = "incorrect"
                else:
                    raw_label = "not_verifiable"
            elif len(sent.numbers) >= 1:
                # Value extraction: check against table
                best_score = max(
                    find_closest_table_value(n, table_values)[1]
                    for n in sent.numbers
                )
                if best_score > 0.7:
                    raw_label = "correct"
                elif best_score < 0.3:
                    raw_label = "incorrect"
                else:
                    raw_label = "not_verifiable"  # Ambiguous
            else:
                raw_label = "not_verifiable"
            
            sentence_data.append({
                "text": sent.text,
                "index": sent.index,
                "numbers": sent.numbers,
                "raw_label": raw_label,
                "computation_correct": comp_correct if sent.has_computation else None
            })
        
        # If LLM verifier available, use it for ambiguous cases
        if self.verifier is not None:
            sentence_data = self._refine_with_llm_verifier(
                sentence_data, csv_path, table_values
            )
        
        # Phase 2: Causal attribution (rule-based, no LLM)
        attributed = compute_causal_rewards(
            sentence_data, table_values, sigma=self.sigma
        )
        
        # Phase 3: Aggregate sentence rewards
        valid_rewards = [
            s.sentence_reward for s in attributed 
            if s.sentence_reward >= 0  # -1.0 means excluded
        ]
        
        if not valid_rewards:
            return 0.0
        
        return sum(valid_rewards) / len(valid_rewards)
    
    def _refine_with_llm_verifier(self, sentences, csv_path, table_values):
        """
        Use LLM verifier for sentences where rule-based is uncertain.
        Only called if self.verifier is not None.
        """
        # [Agent: implement batch LLM verification for uncertain sentences]
        # This is the LLM-based enhancement from PHASE 1
        # Use the same prompts tested in pilot study
        return sentences  # Placeholder: return unchanged if not implemented
    
    def __call__(self, response: str, gold_answer: str, csv_path: str) -> float:
        """
        Compute final ChartVCR reward.
        
        R = w_accuracy × R_acc + w_process × R_proc + w_format × R_fmt
        """
        r_acc = self._compute_accuracy_reward(response, gold_answer)
        r_proc = self._compute_process_reward(response, csv_path)
        r_fmt = self._compute_format_reward(response)
        
        total = (self.w_accuracy * r_acc + 
                 self.w_process * r_proc + 
                 self.w_format * r_fmt)
        
        return total
```

### 4.2 Integration with VLM-R1 GRPO

```python
# File: code/rewards/grpo_integration.py
"""
Modify BigCharts-R1's GRPO reward interface to use ChartVCR.

The key integration point is the reward function signature.
VLM-R1's GRPO expects: reward_fn(response: str, gold: str) -> float
We need:              reward_fn(response: str, gold: str, csv_path: str) -> float

MODIFICATION REQUIRED in BigCharts-R1 codebase:
1. src/open-r1-multimodal/grpo_trainer.py
   - data_collator must pass csv_path to reward function
2. Data format must include csv_path field (done in PHASE 0.6)
"""

# AGENT: These are the exact code modifications needed.
# Apply them as patches to the BigCharts-R1 codebase.

PATCH_1 = """
# In src/open-r1-multimodal/reward_functions.py
# ADD the following function:

from code.rewards.chart_vcr_reward import ChartVCRReward

_cvr_instance = None

def get_cvr_reward():
    global _cvr_instance
    if _cvr_instance is None:
        _cvr_instance = ChartVCRReward(
            verifier=None,  # Start with rule-based; add LLM verifier later
            sigma=0.10,
            w_accuracy=0.5,
            w_process=0.3,
            w_format=0.2
        )
    return _cvr_instance

def chart_vcr_reward(response: str, gold_answer: str, csv_path: str) -> float:
    return get_cvr_reward()(response, gold_answer, csv_path)
"""

PATCH_2 = """
# In src/open-r1-multimodal/grpo_trainer.py
# FIND the line where reward is computed (approximately):
#     reward = accuracy_reward(response, gold) + format_reward(response)
# REPLACE with:
#     if hasattr(sample, 'csv_path') and sample.csv_path:
#         reward = chart_vcr_reward(response, gold, sample.csv_path)
#     else:
#         reward = accuracy_reward(response, gold) + format_reward(response)
"""

PATCH_3 = """
# In the data loading section of grpo_trainer.py
# ENSURE that csv_path is loaded from the JSON data and passed through
# the data pipeline to the reward computation step.
# The exact location depends on how VLM-R1 structures its data loader.

# AGENT: Search for where 'problem' and 'solution' are read from the JSON,
# and add 'csv_path' to the same extraction logic.
"""
```

### 4.3 Verifier Serving (If Pilot Passed)

```bash
# Only if results/pilot/decision.json says "llm_verifier"
# Start vLLM server for the verifier model

python -m vllm.entrypoints.openai.api_server \
    --model models/phi4-mini \
    --port 8100 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --dtype bfloat16

# This runs on 1 GPU, using ~12GB (30% of one A100-40GB)
# The remaining 7 GPUs + 70% of GPU0 are for GRPO training
```

---
---

## PHASE 5: ChartVCR-GRPO TRAINING

### 5.1 Training Launch

```bash
# Same as Phase 3, but with ChartVCR reward instead of outcome-only
# Starting from the SAME SFT checkpoint (fair comparison)

cd code/bigcharts-r1

# The GRPO script should now use chart_vcr_reward
# Verify the patch from Phase 4.2 is applied

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash src/open-r1-multimodal/run_grpo_cvr_job.sh

# Parameters (same as baseline):
# lr=1e-6, batch=8, G=8, epoch=1
# Only difference: reward function
```

**Expected duration:** ~28-30h (20% overhead from reward computation)

### 5.2 Training Monitoring

```python
# File: code/analysis/monitor_training.py
"""
Monitor GRPO training curves.
Log every 50 steps:
- Average reward per batch
- R_accuracy component average
- R_process component average  
- R_format component average
- Average CoT length (number of sentences)
- Percentage of source_error vs propagated_error in CoT

CRITICAL METRICS TO WATCH:
- If average reward decreases for >200 steps: training is diverging
- If CoT length decreasing steadily: shortcut collapse happening
- If R_process is always 0: reward function is broken
"""
# [Agent: implement as a callback in the GRPO trainer]
```

---
---

## PHASE 6: EVALUATION ON ALL BENCHMARKS

### 6.1 Evaluation Script

```python
# File: code/evaluation/full_eval.py
"""
Evaluate all models on all benchmarks.
Generates results/main_table/complete_results.json
"""
import json, os, subprocess

MODELS = {
    "qwen25vl_zeroshot": "models/qwen25vl-7b",
    "sft_only": "checkpoints/sft",
    "grpo_baseline": "checkpoints/grpo_baseline",
    "grpo_cvr": "checkpoints/grpo_cvr",
}

BENCHMARKS = {
    "chartqa_human": {
        "images": "data/chartqa/test/png",
        "data": "data/chartqa/test/test_human.json",
        "metric": "relaxed_accuracy",
        "has_tables": True
    },
    "chartqa_augmented": {
        "images": "data/chartqa/test/png",
        "data": "data/chartqa/test/test_augmented.json",
        "metric": "relaxed_accuracy",
        "has_tables": True
    },
    "charxiv_reasoning": {
        "images": "data/charxiv/images",
        "data": "data/charxiv/val_reasoning.json",
        "metric": "gpt4o_judge",  # Uses CharXiv official eval
        "has_tables": False
    },
    "chartqa_pro": {
        "images": "data/chartqa_pro/images",
        "data": "data/chartqa_pro/test.json",
        "metric": "official_script",
        "has_tables": False
    },
    "plotqa_sub": {
        "images": "data/plotqa/test/png",
        "data": "data/plotqa/plotqa_sub_1k.json",
        "metric": "relaxed_accuracy",
        "has_tables": True
    }
}

def run_evaluation(model_name, model_path, benchmark_name, benchmark_config):
    """Run single model-benchmark evaluation."""
    output_path = f"results/main_table/{model_name}_{benchmark_name}.json"
    
    if benchmark_config["metric"] == "gpt4o_judge":
        # CharXiv evaluation requires GPT-4o API
        # Use CharXiv's official evaluation script
        # [Agent: implement using data/charxiv_repo/evaluation/]
        pass
    else:
        # Standard evaluation
        cmd = [
            "python", "code/bigcharts-r1/src/eval/eval_bigcharts_r1.py",
            "--model-path", model_path,
            "--images-path", benchmark_config["images"],
            "--data-path", benchmark_config["data"],
            "--target-path", output_path,
            "--accuracy-mode", benchmark_config["metric"],
            "--batch-size", "4"
        ]
        subprocess.run(cmd, check=True)
    
    return output_path

def run_all_evaluations():
    """Run all model × benchmark evaluations."""
    results = {}
    
    for model_name, model_path in MODELS.items():
        results[model_name] = {}
        for bench_name, bench_config in BENCHMARKS.items():
            print(f"\nEvaluating {model_name} on {bench_name}...")
            output = run_evaluation(model_name, model_path, bench_name, bench_config)
            
            # Parse result
            with open(output) as f:
                result_data = json.load(f)
            accuracy = result_data.get("accuracy", result_data.get("score", 0))
            results[model_name][bench_name] = accuracy
            print(f"  → {accuracy:.2%}")
    
    # Save complete results
    with open("results/main_table/complete_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_all_evaluations()
```

### 6.2 BoN@8 Evaluation (ChartQA Only)

```python
# File: code/evaluation/bon_eval.py
"""
Best-of-N evaluation with CVR as verifier.
ONLY on benchmarks with data tables (ChartQA, PlotQA).

Generate 8 responses, score each with ChartVCR (without R_accuracy),
select highest scoring response.
"""
def bon_evaluate(model, benchmark, N=8):
    """
    For each sample:
    1. Generate N responses
    2. Score each with R_process + R_format (NO R_accuracy - we don't know gold)
    3. Select highest scoring
    4. Compute relaxed accuracy of selected vs gold
    """
    # [Agent: implement this]
    # Key: the R_process here uses csv_path for verification
    # but does NOT use the gold answer
    pass
```

---
---

## PHASE 7: ABLATION EXPERIMENTS

### 7.1 Ablation Matrix

```
REQUIRED ABLATIONS (must complete):

A1: Reward component analysis
    A1a: R_accuracy only (= BigCharts-R1 baseline, already done in Phase 3)
    A1b: R_accuracy + R_process (no R_format)     → isolates R_process contribution
    A1c: R_accuracy + R_format (no R_process)      → isolates R_format contribution

A2: Causal attribution analysis
    A2a: Full ChartVCR (with causal attribution)   → already done in Phase 5
    A2b: Equal penalty (incorrect = 0.0, no source/propagated distinction)
    → Difference A2a - A2b = pure contribution of causal attribution

A3: Sigma ablation
    A3a: σ = 0.05 (strict)
    A3b: σ = 0.10 (default)
    A3c: σ = 0.15 (lenient)
    A3d: σ = 0.20 (very lenient)
    → Only on ChartQA-Human (quick eval)

IF TIME PERMITS:

A4: Weight sensitivity
    A4a: w_acc=0.7, w_proc=0.2, w_fmt=0.1
    A4b: w_acc=0.3, w_proc=0.5, w_fmt=0.2
    
A5: CoT length analysis (no training, analysis only)
    → Compare average CoT length: baseline vs CVR vs equal-penalty
    → Plot CoT length vs accuracy scatter for each model
```

### 7.2 Ablation Execution

```bash
# Each ablation = 1 GRPO training run (~28h)
# Required: A1b, A1c, A2b, A3a, A3c = 5 runs
# At 28h each: ~140h = 17.5h on 8×A100 = ~6 days

# PRIORITY ORDER (do first 3 regardless):
# 1. A2b (equal penalty) — most critical for narrative
# 2. A1b (R_acc + R_proc only) — proves R_process value
# 3. A3a and A3c (sigma sweep) — can evaluate on ChartQA only for speed

# A2b: Equal penalty GRPO
# Modify: in causal_attribution.py, set all propagated_error → logic_score=0.0
# This makes propagated_error identical to source_error in reward

# A1b: No format reward
# Modify: w_format=0.0, redistribute to w_accuracy=0.55, w_process=0.45

# A3: Sigma sweep
# Modify: sigma parameter in ChartVCRReward constructor
```

---
---

## PHASE 8: ANALYSIS & VISUALIZATION

### 8.1 Main Results Table

```python
# File: code/analysis/generate_main_table.py
"""Generate LaTeX table for main results."""

def generate_latex_table():
    with open("results/main_table/complete_results.json") as f:
        results = json.load(f)
    
    # LaTeX table
    header = r"""
\begin{table*}[t]
\centering
\caption{Main results on chart reasoning benchmarks. 
Relaxed accuracy (\%) for ChartQA and PlotQA; GPT-4o judged accuracy for CharXiv; 
official evaluation for ChartQA-Pro. 
Best results in \textbf{bold}, second best \underline{underlined}.}
\label{tab:main}
\begin{tabular}{lccccc|c}
\toprule
\textbf{Model} & \textbf{CQA-H} & \textbf{CQA-A} & \textbf{CharXiv-R} & \textbf{CQA-Pro} & \textbf{PlotQA} & \textbf{Avg} \\
\midrule
"""
    # [Agent: fill in from results dict, compute averages, bold/underline]
    
    return header  # + rows + footer
```

### 8.2 Key Analysis Figures

```python
# File: code/analysis/generate_figures.py
"""
Required figures for the paper:

Fig 1: Framework overview (draw manually or tikz)
Fig 2: Reward decomposition example (one CoT trace with color-coded sentences)
Fig 3: Causal attribution vs equal penalty comparison (bar chart by question type)
Fig 4: CoT length distribution: baseline vs CVR vs equal-penalty
Fig 5: Sigma sensitivity curve (accuracy vs σ on ChartQA-H)
Fig 6: Error type breakdown: source_error vs propagated_error frequency
"""

import matplotlib.pyplot as plt
import numpy as np

def fig3_causal_vs_equal():
    """Bar chart: accuracy by question type for causal vs equal penalty."""
    # [Agent: load from ablation results A2a vs A2b]
    # Separate bars for: computation-heavy, lookup, comparison, trend
    # ChartQA train has these categories derivable from keyword classification
    pass

def fig4_cot_length_distribution():
    """
    Histogram of CoT lengths (number of sentences).
    Three distributions overlaid: baseline, CVR, equal-penalty.
    
    KEY CLAIM: CVR maintains longer CoT than equal-penalty (anti-shortcut).
    """
    # [Agent: extract from generated responses in evaluation]
    pass

def fig5_sigma_sensitivity():
    """
    Line plot: ChartQA-Human accuracy vs σ value.
    X-axis: σ ∈ {0.05, 0.10, 0.15, 0.20}
    Y-axis: accuracy (%)
    """
    # [Agent: load from ablation A3 results]
    pass
```

---
---

## PHASE 9: PAPER WRITING

### 9.1 Paper Structure

```
paper/
├── main.tex           # Main file
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── analysis.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── figures/
│   ├── framework.pdf
│   ├── reward_example.pdf
│   ├── causal_vs_equal.pdf
│   ├── cot_length.pdf
│   ├── sigma_sensitivity.pdf
│   └── error_breakdown.pdf
├── tables/
│   ├── main_results.tex
│   ├── ablation_components.tex
│   ├── ablation_causal.tex
│   └── ablation_sigma.tex
└── references.bib
```

### 9.2 Section-by-Section Writing Plan

```
═══════════════════════════════════════════════════════
TITLE: Chart-Verifiable Causal Rewards for 
       Reasoning-Enhanced Chart Question Answering
═══════════════════════════════════════════════════════

ABSTRACT (150 words):
- Problem: Chart RL uses binary outcome rewards → no credit assignment for reasoning
- Insight: Chart data tables enable step-level verification; error propagation 
  must be distinguished from error origination
- Method: ChartVCR — causal reward separating source errors (penalized) from 
  propagated errors (logic correctness preserved), using Gaussian tolerance 
  for visual estimation
- Results: +X% on CharXiv, +Y% on ChartQA-Pro over BigCharts-R1
- Significance: Longer, more accurate reasoning chains without shortcut collapse

INTRODUCTION (1.5 pages):
- Chart QA is increasingly important (real-world data analysis)
- GRPO-based RL has become dominant (BigCharts-R1, Chart-RL, Chart-R1)
- All use binary outcome reward: answer correct → 1, wrong → 0
- Problem 1: No credit for partially correct reasoning
- Problem 2: Conflates error source with error propagation
- Problem 3: Over-penalizes long reasoning → shortcut CoT collapse
- Our insight: Chart data tables provide ground-truth oracle for step verification
- Our solution: ChartVCR with causal error attribution
- Contributions (3 bullet points)

RELATED WORK (1 page):
- Chart understanding models (TinyChart, ChartGemma, ChartInstruct)
- RL for chart reasoning (BigCharts-R1, Chart-RL, Chart-RVR, Chart-R1)
- Process reward models (PRM800K, VisualPRM, Math-Shepherd, ThinkPRM)
- Key gap: No PRM/process reward for charts; no causal error attribution in any domain

METHOD (2.5 pages):
§3.1 Problem Formulation
  - GRPO background (1 paragraph, equations)
  - Current reward: R = R_acc + R_fmt (binary)
  
§3.2 ChartVCR Framework Overview
  - Figure 1: complete pipeline diagram
  - Three phases: Independent Verification → Causal Attribution → Reward Aggregation

§3.3 Phase 1: Sentence-Level Verification
  - Sentence parsing (regex-based number extraction)
  - Computation verification (arithmetic check via Python execution)
  - Value extraction verification (Gaussian tolerance against data table)
  - Equation: value_accuracy_score with σ parameter
  
§3.4 Phase 2: Causal Error Attribution  
  - Number dependency graph construction
  - Source error vs propagated error classification
  - Formal definition with equations
  
§3.5 Phase 3: Causal Reward Aggregation
  - R_sentence = logic_score × input_quality
  - R_process = average of valid sentence rewards
  - R_total = w_acc × R_acc + w_proc × R_process + w_fmt × R_format
  - Discussion: why this avoids shortcut collapse

EXPERIMENTS (2.5 pages):
§4.1 Setup
  - Base model, training data, hyperparameters
  - Benchmarks table
  - Baselines list

§4.2 Main Results
  - Table 1: main comparison
  - Analysis paragraph per benchmark

§4.3 Ablation Studies
  - Table 2: component ablation (R_acc only, +R_proc, +R_fmt, full)
  - Table 3: causal vs equal penalty
  - Table 4: sigma sensitivity
  - Figure 3: causal vs equal by question type

§4.4 Analysis
  - Figure 4: CoT length comparison (key finding: CVR preserves long CoT)
  - Figure 5: error type breakdown
  - Case study: one example showing source vs propagated attribution

CONCLUSION (0.5 pages):
  - Summary of contributions
  - Limitation: requires data tables (not available for all charts)
  - Limitation: rule-based number tracking has edge cases
  - Future: extend to other verifiable reasoning domains (math, code, science)
  - Future: learn sigma per chart type
```

### 9.3 References (Must-Cite Papers)

```bibtex
% CRITICAL CITATIONS (reviewer will check):

% Chart benchmarks
@inproceedings{masry2022chartqa, ...}          % ChartQA
@inproceedings{wang2024charxiv, ...}           % CharXiv, NeurIPS 2024
@article{masry2025chartqapro, ...}             % ChartQA-Pro, ACL 2025
@inproceedings{zhu2025multichartqa, ...}       % MultiChartQA, NAACL 2025

% Chart RL methods
@article{masry2025bigcharts, ...}              % BigCharts-R1, COLM 2025
@article{zhang2026chartrl, ...}                % Chart-RL, arXiv 2603.06958
@article{sinha2025chartvr, ...}                % Chart-RVR, arXiv 2510.10973
@article{zhao2025chartr1, ...}                 % Chart-R1, arXiv 2507.15509

% Chart models
@inproceedings{zhang2024tinychart, ...}        % TinyChart, EMNLP 2024
@article{masry2025chartgemma, ...}             % ChartGemma
@article{yang2025ecd, ...}                     % ECD, ICCV 2025

% Process reward models
@inproceedings{lightman2024lets, ...}          % Let's Verify Step by Step, ICLR 2024
@article{wang2025visualprm, ...}               % VisualPRM, arXiv 2503.10291
@inproceedings{hosseini2025rewarding, ...}     % Rewarding Progress, ICLR 2025 Spotlight
@article{setlur2025thinkprm, ...}              % ThinkPRM, arXiv 2504.16828

% GRPO / RL fundamentals
@article{shao2024deepseekmath, ...}            % DeepSeek-Math (GRPO)
@article{guo2025deepseekr1, ...}               % DeepSeek-R1
@article{shen2025vlmr1, ...}                   % VLM-R1

% Visual RL
@inproceedings{deng2025reasonrft, ...}         % Reason-RFT, NeurIPS 2025
@article{yang2025visionaryr1, ...}             % Visionary-R1 (shortcut problem)
```

---
---

## PHASE 10: CAMERA-READY POLISH

### 10.1 Final Checklist

```
□ All numbers in paper match results JSON files
□ All figures are generated from actual data (no manual numbers)
□ Ablation tables include standard deviations (3 seeds if budget allows, else note "single run")
□ Related work cites ALL chart RL papers (BigCharts-R1, Chart-RL, Chart-RVR, Chart-R1)
□ Limitation section mentions: data table requirement, rule-based heuristics, sigma sensitivity
□ Code repository prepared for release (anonymized for review)
□ Supplementary material includes:
  □ Full prompt templates for verifier
  □ Pilot study results
  □ Per-question-type breakdown for all benchmarks
  □ Additional case studies
  □ Hyperparameter sensitivity details
□ Paper is within page limit (8 pages + references for EMNLP/ACL, 9 for NeurIPS)
```

---
---

## CONTINGENCY PLANS

### If Pilot Study Fails (Verifier < 70%)
→ Remove LLM verifier from pipeline
→ Use rule-based computation verification only
→ Paper becomes: "Chart-Verifiable Causal Rewards" without LLM component
→ Still valid: the causal attribution + Gaussian tolerance are independent of LLM verifier

### If CVR Gains < 2% Over Baseline
→ Reframe as analysis paper: "Understanding Process Errors in Chart Reasoning"
→ Primary contribution: the source vs propagated error taxonomy + analysis
→ Secondary: the reward framework as a diagnostic tool
→ Publishable at EMNLP Findings / ACL Findings tier

### If GRPO Training Diverges
→ Reduce w_process from 0.3 to 0.1, increase w_accuracy to 0.7
→ If still diverging: use CVR only for BoN reranking (no training integration)
→ BoN results are almost certainly positive (VisualPRM precedent)

### If Compute Budget Exceeded
→ Priority: Main experiment (Phase 5) > Ablations (Phase 7)
→ Minimum viable ablations: A2b (causal vs equal) + A1b (R_process isolation) = 2 runs
→ Skip sigma ablation (report default σ=0.10 only with justification)

### If BigCharts-R1 Reproduction Fails (>5% deviation)
→ Debug by comparing data format, hyperparameters, hardware
→ If unfixable: use Qwen2.5-VL-7B + SFT as baseline instead of BigCharts-R1
→ Reframe: "improvement over SFT" instead of "improvement over BigCharts-R1"
→ Still valid but weaker positioning

---

## TIMELINE SUMMARY

```
Day 1-2:   Phase 0 (setup)
Day 3-4:   Phase 1 (pilot study) — CRITICAL GATE
Day 5-7:   Phase 2 (SFT)
Day 8-10:  Phase 3 (GRPO baseline)
Day 11-13: Phase 4 (ChartVCR implementation)
Day 14-16: Phase 5 (ChartVCR-GRPO training) — MAIN EXPERIMENT
Day 17-18: Phase 6 (evaluation)
Day 19-22: Phase 7 (ablations — priority order)
Day 23-24: Phase 8 (analysis + figures)
Day 25-28: Phase 9 (paper writing)
Day 29-30: Phase 10 (polish + submission prep)

Total: 30 calendar days
GPU usage: ~1,400 hours on 8×A100-40GB
API cost: ~$150-300 (CharXiv eval + CoT generation)
```