"""
LLM Sentence Verifier — OpenAI SDK + vLLM server.

Verifier: Qwen3-VL-4B-Instruct (non-thinking, Instruct mode)
Server: vLLM on port 9200 (OpenAI-compatible API)

모든 검증을 LLM이 수행. Rule-based 없음.
"""
import asyncio
import json
import os
import re
from typing import List, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI

from code.rewards.reasoning_chain import split_reasoning, extract_chart_numbers


# ═══════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise chart data verification assistant.
You verify whether a model's reasoning step is correct by checking against the data table.

RULES:
1. VALUE CHECK: If the sentence cites a numeric value, check if it matches the table (within 10% tolerance)
2. ARITHMETIC CHECK: If the sentence contains a calculation (a+b=c, a-b=c, etc.), compute it yourself and verify
3. NOT VERIFIABLE: If the sentence has no numeric value extraction and no calculation (just text reasoning, counting items, describing layout), classify as not_verifiable

You MUST respond with ONLY <answer>correct</answer>, <answer>incorrect</answer>, or <answer>not_verifiable</answer>. Nothing else."""

VERIFY_PROMPT = """Data table:
{csv_content}

Reasoning step: "{sentence}"
Question: "{question}"

Verify this reasoning step against the data table. Classify in <answer> tags."""


# ═══════════════════════════════════════════
# Verifier
# ═══════════════════════════════════════════

class LLMVerifier:
    """
    Async LLM verifier using OpenAI-compatible vLLM server.

    Usage:
        verifier = LLMVerifier(base_url="http://localhost:9200/v1")
        labels = await verifier.verify_batch(sentences, csv_content, question)
        # or sync:
        labels = verifier.verify_batch_sync(sentences, csv_content, question)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9200/v1",
        model_id: str = "verifier",
        max_tokens: int = 30,
        temperature: float = 0.0,
        max_concurrent: int = 8,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrent)

    @staticmethod
    def parse_label(text: str) -> str:
        """Parse <answer>label</answer> from response."""
        m = re.search(r'<answer>\s*(correct|incorrect|not_verifiable)\s*</answer>', text, re.I)
        if m:
            return m.group(1).lower()
        # Fallback
        t = text.strip().lower()
        if "incorrect" in t:
            return "incorrect"
        if "correct" in t and "not" not in t:
            return "correct"
        return "not_verifiable"

    async def _verify_one(self, sentence: str, csv_content: str, question: str) -> str:
        """Verify a single sentence."""
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": VERIFY_PROMPT.format(
                            csv_content=csv_content[:1500],
                            sentence=sentence[:200],
                            question=question,
                        )},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw = resp.choices[0].message.content or ""
                return self.parse_label(raw)
            except Exception as e:
                return "not_verifiable"

    async def verify_batch(
        self,
        sentences: List[str],
        csv_content: str,
        question: str,
    ) -> List[str]:
        """Verify all sentences concurrently."""
        tasks = []
        for sent in sentences:
            chart_nums = extract_chart_numbers(sent)
            if not chart_nums and not re.search(r'\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=', sent):
                # No numbers and no arithmetic → skip
                tasks.append(asyncio.coroutine(lambda: "not_verifiable")())
            else:
                tasks.append(self._verify_one(sent, csv_content, question))
        return await asyncio.gather(*tasks)

    def verify_batch_sync(
        self,
        sentences: List[str],
        csv_content: str,
        question: str,
    ) -> List[str]:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.verify_batch(sentences, csv_content, question))
        finally:
            loop.close()


# ═══════════════════════════════════════════
# Process Reward with LLM Verifier
# ═══════════════════════════════════════════

@dataclass
class VerifiedSentence:
    text: str
    index: int
    label: str  # correct, incorrect, not_verifiable, source_error, propagated_error
    chart_numbers: list
    sentence_reward: float


def compute_process_reward_llm(
    response: str,
    csv_path: str,
    question: str,
    verifier: LLMVerifier,
) -> Tuple[float, List[VerifiedSentence]]:
    """
    Process reward using LLM verifier only (no rule-based).

    Phase 1: LLM classifies each sentence
    Phase 2: Causal attribution (source vs propagated error)
    Phase 3: Aggregate
    """
    # Load CSV
    csv_content = ""
    if csv_path and os.path.exists(csv_path):
        try:
            csv_content = open(csv_path).read()
        except:
            pass

    if not csv_content:
        return 0.0, []

    # Split reasoning
    sentences = split_reasoning(response)
    if not sentences:
        return 0.0, []

    # Phase 1: LLM verification
    labels = verifier.verify_batch_sync(sentences, csv_content, question)

    # Phase 2: Causal attribution
    tainted = set()
    analyses = []

    for i, (sent, label) in enumerate(zip(sentences, labels)):
        chart_nums = extract_chart_numbers(sent)

        if label == "not_verifiable":
            analyses.append(VerifiedSentence(
                text=sent, index=i, label="skip",
                chart_numbers=chart_nums, sentence_reward=0.0,
            ))
            continue

        if label == "correct":
            reward = 1.0
            final_label = "correct"
        else:
            # incorrect — determine source vs propagated
            uses_tainted = any(
                any(abs(n - t) / max(abs(t), 1e-10) < 0.05 for t in tainted)
                for n in chart_nums
            ) if chart_nums else False

            if uses_tainted:
                reward = 0.3  # Propagated: partial credit for logic
                final_label = "propagated_error"
            else:
                reward = 0.0
                final_label = "source_error"
                for n in chart_nums:
                    tainted.add(n)

        analyses.append(VerifiedSentence(
            text=sent, index=i, label=final_label,
            chart_numbers=chart_nums, sentence_reward=reward,
        ))

    # Phase 3: Aggregate
    verified = [a.sentence_reward for a in analyses if a.label != "skip"]
    total_reward = sum(verified) / len(verified) if verified else 0.0

    return total_reward, analyses
