"""
LLM-based Sentence Verifier using Qwen3-VL-4B-Instruct.

Policy model: Qwen3-VL-8B-Thinking (reasoning, different variant)
Verifier model: Qwen3-VL-4B-Instruct (classification, non-thinking mode)

The verifier receives:
- The chart's data table (CSV content)
- A single reasoning sentence from the model's output
- The original question for context

And classifies it as: correct / incorrect / not_verifiable

This replaces the rule-based number extraction + Gaussian matching
for sentences where rule-based is unreliable (counting, visual estimation, etc.)
"""
import re
import os
import math
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

from code.rewards.reasoning_chain import (
    split_reasoning,
    extract_chart_numbers,
    SentenceAnalysis,
    _CALC_PATTERN,
    gaussian_score,
    best_table_match,
)


# ═══════════════════════════════════════════
# Prompts for LLM Verifier
# ═══════════════════════════════════════════

VERIFIER_SYSTEM_PROMPT = """You are a chart data verification expert.
You verify whether a reasoning step correctly uses values from a data table.
You MUST respond with your classification inside <answer> tags.
Valid labels: correct, incorrect, not_verifiable

Example responses:
<answer>correct</answer>
<answer>incorrect</answer>
<answer>not_verifiable</answer>"""

EXTRACTION_VERIFY_PROMPT = """Data table from the chart:
{csv_content}

A model analyzing this chart wrote the following reasoning step:
"{sentence}"

The question being answered: "{question}"

Classify this reasoning step:
- "correct": The sentence reads/references numeric values from the table AND they match (within 10% tolerance for numbers)
- "incorrect": The sentence reads/references numeric values but they do NOT match the table
- "not_verifiable": The sentence has no verifiable value extraction (text-only reasoning, counting items, describing layout, mentioning years/labels without numeric values)

Put your classification in <answer> tags."""

COMPUTATION_VERIFY_PROMPT = """A model wrote this reasoning step:
"{sentence}"

Classify the mathematical calculation in this step:
- "correct": Contains a calculation AND the math is correct
- "incorrect": Contains a calculation AND the result is wrong
- "not_verifiable": No mathematical calculation present

Put your classification in <answer> tags."""


def _parse_verifier_output(raw: str) -> str:
    """Parse verifier output from <answer>label</answer> format."""
    raw = raw.strip()
    # Try <answer> tag extraction first
    m = re.search(r'<answer>\s*(correct|incorrect|not_verifiable)\s*</answer>', raw, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: keyword matching
    raw_lower = raw.lower()
    if "incorrect" in raw_lower:
        return "incorrect"
    if "correct" in raw_lower and "not" not in raw_lower:
        return "correct"
    return "not_verifiable"


# ═══════════════════════════════════════════
# Verifier Class
# ═══════════════════════════════════════════

class LLMVerifier:
    """
    LLM-based sentence verifier using Qwen3-VL-4B-Instruct.

    Usage:
        verifier = LLMVerifier(model_path="models/qwen3vl-4b-instruct")
        verifier.start(gpu_id=0)
        labels = verifier.verify_sentences(sentences, csv_content, question)
        verifier.stop()

    Or use as vLLM client (if server already running):
        verifier = LLMVerifier(server_url="http://localhost:9200")
    """

    def __init__(self, model_path: str = None, server_url: str = None):
        self.model_path = model_path
        self.server_url = server_url
        self.llm = None
        self.sampling_params = None

    def start(self, gpu_id: int = 0, gpu_memory_utilization: float = 0.85):
        """Start local vLLM instance for the verifier model."""
        if self.server_url:
            return  # Using external server

        from vllm import LLM, SamplingParams
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=30,  # Room for <answer>not_verifiable</answer>
        )

    def stop(self):
        """Release resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            import torch
            torch.cuda.empty_cache()

    def _generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for prompts using chat format with system prompt."""
        if self.llm is not None:
            # Local vLLM — use chat format for better instruction following
            messages_batch = [
                [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
            outputs = self.llm.chat(
                messages_batch,
                self.sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
            return [o.outputs[0].text.strip() for o in outputs]
        elif self.server_url:
            # External server via OpenAI-compatible API
            import requests
            results = []
            for prompt in prompts:
                resp = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json={
                        "model": "verifier",
                        "messages": [
                            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 30,
                        "temperature": 0,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    results.append(resp.json()["choices"][0]["message"]["content"].strip())
                else:
                    results.append("<answer>not_verifiable</answer>")
            return results
        else:
            raise RuntimeError("Verifier not started. Call start() or provide server_url.")

    def verify_sentences(
        self,
        sentences: List[str],
        csv_content: str,
        question: str,
    ) -> List[str]:
        """
        Verify a list of sentences against the chart data table.

        Returns list of labels: "correct", "incorrect", "not_verifiable"
        """
        prompts = []
        prompt_indices = []  # Track which sentences have prompts

        for i, sent in enumerate(sentences):
            # Determine prompt type
            has_calc = bool(_CALC_PATTERN.search(sent))
            chart_nums = extract_chart_numbers(sent)

            if has_calc:
                prompt = COMPUTATION_VERIFY_PROMPT.format(sentence=sent)
            elif chart_nums:
                prompt = EXTRACTION_VERIFY_PROMPT.format(
                    csv_content=csv_content[:1500],  # Truncate large CSVs
                    sentence=sent,
                    question=question,
                )
            else:
                # No numbers, no computation → not verifiable
                prompt_indices.append((i, None))
                continue

            prompts.append(prompt)
            prompt_indices.append((i, len(prompts) - 1))

        # Batch generate
        if prompts:
            raw_outputs = self._generate(prompts)
        else:
            raw_outputs = []

        # Map results back
        labels = []
        for i, prompt_idx in prompt_indices:
            if prompt_idx is None:
                labels.append("not_verifiable")
            else:
                labels.append(_parse_verifier_output(raw_outputs[prompt_idx]))

        return labels


# ═══════════════════════════════════════════
# Process Reward with LLM Verifier
# ═══════════════════════════════════════════

def compute_process_reward_with_verifier(
    response: str,
    csv_path: str,
    question: str,
    verifier: LLMVerifier,
    sigma: float = 0.10,
) -> Tuple[float, List[dict]]:
    """
    Compute process reward using LLM verifier for sentence classification
    + rule-based causal attribution.

    Phase 1 (LLM): Classify each sentence as correct/incorrect/not_verifiable
    Phase 2 (Rule): Apply causal error attribution (source vs propagated)
    Phase 3: Aggregate sentence rewards
    """
    import pandas as pd

    # Load table
    table_vals = set()
    csv_content = ""
    if csv_path and os.path.exists(csv_path):
        try:
            csv_content = open(csv_path).read()
            df = pd.read_csv(csv_path)
            for col in df.columns:
                for v in df[col]:
                    try:
                        table_vals.add(float(v))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    if not table_vals:
        return 0.0, []

    # Split reasoning
    sentences = split_reasoning(response)
    if not sentences:
        return 0.0, []

    # Phase 1: LLM verification
    llm_labels = verifier.verify_sentences(sentences, csv_content, question)

    # Phase 2: Causal attribution
    tainted: Set[float] = set()
    analyses = []

    for i, (sent, label) in enumerate(zip(sentences, llm_labels)):
        chart_nums = extract_chart_numbers(sent)

        analysis = {
            "text": sent,
            "index": i,
            "chart_numbers": chart_nums,
            "llm_label": label,
            "logic_score": 1.0,
            "input_quality": 1.0,
            "sentence_reward": 0.0,
            "label": "skip",
        }

        if label == "not_verifiable":
            analysis["label"] = "skip"
            analyses.append(analysis)
            continue

        if label == "correct":
            analysis["logic_score"] = 1.0
            analysis["input_quality"] = 1.0
            analysis["sentence_reward"] = 1.0
            analysis["label"] = "correct"
        elif label == "incorrect":
            # Check if uses tainted numbers
            uses_tainted = any(
                any(abs(n - t) / max(abs(t), 1e-10) < 0.05 for t in tainted)
                for n in chart_nums
            ) if chart_nums else False

            if uses_tainted:
                # Propagated error: check if logic itself is correct
                has_calc = bool(_CALC_PATTERN.search(sent))
                if has_calc:
                    # Verify arithmetic separately
                    cm = _CALC_PATTERN.search(sent)
                    try:
                        a = float(cm.group(1).replace(',', ''))
                        op = cm.group(2).replace('×', '*').replace('÷', '/')
                        b = float(cm.group(3).replace(',', ''))
                        stated = float(cm.group(4).replace(',', ''))
                        expected = eval(f"{a}{op}{b}") if op in '+-*/' else None
                        analysis["logic_score"] = 1.0 if expected and abs(stated - expected) / max(abs(expected), 1e-10) < 0.05 else 0.0
                    except:
                        analysis["logic_score"] = 0.5
                else:
                    analysis["logic_score"] = 0.5  # Unknown

                analysis["input_quality"] = 0.3  # Tainted input
                analysis["label"] = "propagated_error"
            else:
                # Source error
                analysis["logic_score"] = 0.0
                analysis["input_quality"] = 0.0
                analysis["label"] = "source_error"
                # Taint numbers
                for n in chart_nums:
                    tainted.add(n)

            analysis["sentence_reward"] = analysis["logic_score"] * analysis["input_quality"]

        analyses.append(analysis)

    # Phase 3: Aggregate
    verified = [a["sentence_reward"] for a in analyses if a["label"] != "skip"]
    reward = sum(verified) / len(verified) if verified else 0.0

    return reward, analyses
