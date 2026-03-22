"""
ChartVCR LLM Verifier (v5.1)

Architecture:
  LLM: value grounding 판정 (CSV 대조) + computation 추출
  Rule: 산술 재계산 + causal error attribution + scoring

  LLM은 점수를 매기지 않는다.
  LLM은 entity-level factual grounding만 판정하고, computation을 추출한다.
  Rule이 산술 재계산과 causal scoring을 전부 수행한다.

Verifier: Qwen3.5-397B-A17B-FP8 (on-premise)
Server: http://10.1.211.148:8000/v1
"""
import asyncio
import json
import math
import os
import re
from typing import List, Tuple, Optional, Set

import pandas as pd
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI


# ═══════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════

class ValueClaim(BaseModel):
    entity: str              # "Corn" — 모델이 읽었다고 주장하는 대상
    claimed_value: float     # 103.7 — 모델이 주장한 값
    table_value: float       # 103.13 — LLM이 CSV에서 찾은 실제 값
    is_correct: bool         # false — LLM이 판정 (claimed ≈ table?)


class ComputationClaim(BaseModel):
    operands: List[float]    # [103.7, 103.13]
    operator: str            # "-"
    stated_result: float     # 0.57
    # is_correct 없음 — Rule이 판정


class ExtractionResult(BaseModel):
    value_claims: List[ValueClaim] = []
    computation_claims: List[ComputationClaim] = []


# ═══════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════

VERIFIER_SYSTEM = """\
You are a chart reasoning verifier. You receive:
1. A data table (CSV) with the chart's ground-truth values
2. A model's reasoning trace about a chart question

Your job has TWO parts:

## PART A — VALUE CLAIMS (you JUDGE these)
Each time the model states a specific number it claims to read from the chart:
- Find the matching entity in the data table
- Compare the claimed value against the table value
- Set is_correct = true if they match within 5% tolerance

Example:
  Model says: "Corn has a value of 103.7"
  Table shows: Corn = 103.13
  → entity="Corn", claimed_value=103.7, table_value=103.13, is_correct=false

DO NOT extract as value claims:
  - Years used only as time labels ("In 2015" without a data value)
  - List indices ("1. Lamb" — the 1 is an index)
  - Numbers from the question itself
  - The final answer

## PART B — COMPUTATION CLAIMS (you EXTRACT only, do NOT judge)
Each time the model performs arithmetic, extract the operands, operator,
and the result the model claims. Do NOT verify the arithmetic yourself.

Supported operators: +, -, *, /, avg, max, min, count, ratio, %change

Example:
  Model says: "103.7 - 103.13 = 0.57"
  → operands=[103.7, 103.13], operator="-", stated_result=0.57

For fractions: convert stated_result to float (29/12 → 2.4167).
For percentages in values: keep as stated (23% → 23.0).

IMPORTANT: Extract as many claims as you can, even if some parts of the
reasoning are malformed. If you can identify a value reading (e.g.,
"Alpha: 33.4") but the subsequent calculation is unreadable, still
extract the value reading. Do not return empty claims just because
one part of the reasoning is hard to parse.

Entity names may have minor spelling variations — match them to the
closest table entity (e.g., "Alph" → "Alpha", "alpha" → "Alpha").

Respond with ONLY valid JSON matching the schema."""


VERIFIER_USER = """\
## Data Table
{csv_text}

## Question
{question}

## Model's Reasoning Trace
{reasoning}"""


# G-Eval prompt (ablation A2)
GEVAL_SYSTEM = """\
You are a chart reasoning evaluator.
Given the data table and a model's full reasoning trace, score the overall quality on a scale of 1-5:
1 = Completely wrong values and logic
2 = Major errors in values or reasoning
3 = Partially correct, some errors
4 = Mostly correct, minor errors
5 = Fully correct values and computation
Respond with ONLY a single digit (1-5)."""


# ═══════════════════════════════════════════
# Verifier Client
# ═══════════════════════════════════════════

def csv_to_text(csv_path: str, max_rows: int = 50) -> str:
    try:
        df = pd.read_csv(csv_path)
        return df.head(max_rows).to_csv(index=False)
    except Exception:
        return ""


def extract_reasoning(response: str) -> str:
    m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def _parse_json_from_response(raw: str) -> dict:
    """Extract JSON from response that may contain thinking + JSON."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    last_brace = raw.rfind('}')
    if last_brace == -1:
        return {}
    depth = 0
    for i in range(last_brace, -1, -1):
        if raw[i] == '}':
            depth += 1
        elif raw[i] == '{':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[i:last_brace + 1])
                except (json.JSONDecodeError, ValueError):
                    return {}
    return {}


class VerifierClient:
    """Async LLM verifier — 1 call per completion."""

    def __init__(
        self,
        base_url: str = "http://10.1.211.148:8000/v1",
        model_id: str = "Qwen3.5-397B-A17B-FP8",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_concurrent: int = 8,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent

    async def verify_single(
        self, csv_text: str, question: str, reasoning: str,
    ) -> Tuple[ExtractionResult, str]:
        """Verify one completion with retry. Returns (ExtractionResult, raw_response)."""
        for attempt in range(3):  # up to 3 retries
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": VERIFIER_SYSTEM},
                        {"role": "user", "content": VERIFIER_USER.format(
                            csv_text=csv_text,
                            question=question,
                            reasoning=reasoning[:3000],
                        )},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                raw = resp.choices[0].message.content or "{}"
                parsed = _parse_json_from_response(raw)
                result = ExtractionResult.model_validate(parsed)
                # If claims=0, retry (up to limit)
                if not result.value_claims and not result.computation_claims and attempt < 2:
                    continue
                return result, raw
            except (ValidationError, Exception):
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1.0)
                    continue
                return ExtractionResult(), ""
        return ExtractionResult(), ""

    async def verify_batch(
        self, csv_path: str, question: str, completions: List[str],
    ) -> List[Tuple[ExtractionResult, str]]:
        """Verify a batch of completions concurrently."""
        csv_text = csv_to_text(csv_path)
        if not csv_text:
            return [(ExtractionResult(), "")] * len(completions)

        sem = asyncio.Semaphore(self.max_concurrent)

        async def _with_sem(reasoning):
            async with sem:
                return await self.verify_single(csv_text, question, reasoning)

        tasks = [_with_sem(extract_reasoning(c)) for c in completions]
        return await asyncio.gather(*tasks)

    def verify_batch_sync(
        self, csv_path: str, question: str, completions: List[str],
    ) -> List[Tuple[ExtractionResult, str]]:
        """Synchronous wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.verify_batch(csv_path, question, completions)
            )
        finally:
            loop.close()

    # ── G-Eval (Ablation A2) ──

    async def geval_single(
        self, csv_text: str, question: str, reasoning: str,
    ) -> float:
        """G-Eval holistic score (1-5 → 0-1)."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": GEVAL_SYSTEM},
                    {"role": "user", "content": VERIFIER_USER.format(
                        csv_text=csv_text,
                        question=question,
                        reasoning=reasoning[:3000],
                    )},
                ],
                max_tokens=5,
                temperature=0.0,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            raw = resp.choices[0].message.content or ""
            m = re.search(r'[1-5]', raw)
            return (int(m.group()) - 1) / 4.0 if m else 0.0
        except Exception:
            return 0.0

    def geval_batch_sync(
        self, csv_path: str, question: str, completions: List[str],
    ) -> List[float]:
        """Synchronous G-Eval for batch."""
        csv_text = csv_to_text(csv_path)
        if not csv_text:
            return [0.0] * len(completions)

        async def _run():
            sem = asyncio.Semaphore(self.max_concurrent)
            async def _with_sem(reasoning):
                async with sem:
                    return await self.geval_single(csv_text, question, reasoning)
            tasks = [_with_sem(extract_reasoning(c)) for c in completions]
            return await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()


# ═══════════════════════════════════════════
# Rule-Based Scoring
# ═══════════════════════════════════════════

def verify_computation(claim: ComputationClaim) -> Tuple[bool, float]:
    """Python으로 산술 재계산. Returns (is_correct, actual_result)."""
    ops = claim.operands
    op = claim.operator

    try:
        if op == "+":
            actual = sum(ops)
        elif op == "-" and len(ops) == 2:
            actual = ops[0] - ops[1]
        elif op == "*" and len(ops) == 2:
            actual = ops[0] * ops[1]
        elif op == "/" and len(ops) == 2 and ops[1] != 0:
            actual = ops[0] / ops[1]
        elif op == "avg" and ops:
            actual = sum(ops) / len(ops)
        elif op == "max" and ops:
            actual = max(ops)
        elif op == "min" and ops:
            actual = min(ops)
        elif op == "count":
            actual = float(len(ops))
        elif op == "ratio" and len(ops) == 2 and ops[1] != 0:
            actual = ops[0] / ops[1]
        elif op == "%change" and len(ops) == 2 and ops[0] != 0:
            actual = (ops[1] - ops[0]) / abs(ops[0]) * 100
        else:
            return True, 0.0  # 검증 불가 operator → 패널티 안 줌

        if actual == 0:
            is_correct = abs(claim.stated_result) < 0.01
        else:
            is_correct = abs(claim.stated_result - actual) / abs(actual) <= 0.05
        return is_correct, actual
    except Exception:
        return True, 0.0  # 예외 → 패널티 안 줌


def compute_process_reward(extraction: ExtractionResult, reasoning: str = "") -> Tuple[float, dict]:
    """
    Rule-based scoring with causal attribution.

    Returns (reward, details_dict).
    """
    if not extraction.value_claims and not extraction.computation_claims:
        # Our training data is ALL computation questions.
        # claims=0 means: fabrication, parse failure, or no reasoning.
        # None of these deserve a high score.
        has_numbers = bool(re.search(r'\d+\.?\d*', reasoning))
        if has_numbers:
            # Numbers in reasoning but verifier extracted nothing → suspicious
            return 0.2, {"scores": [], "incorrect_values": [], "note": "claims=0 with numbers — possible fabrication or parse failure"}
        else:
            # Computation question but no numbers in reasoning → worst case
            return 0.0, {"scores": [], "incorrect_values": [], "note": "claims=0 no numbers — empty reasoning"}

    scores = []
    details = []
    incorrect_values: Set[float] = set()

    # ── Value claims: LLM 판정 사용 ──
    for vc in extraction.value_claims:
        if vc.is_correct:
            scores.append(1.0)
            details.append({
                "type": "value", "entity": vc.entity,
                "claimed": vc.claimed_value, "table": vc.table_value,
                "verdict": "correct", "score": 1.0,
            })
        else:
            scores.append(0.0)
            incorrect_values.add(vc.claimed_value)
            details.append({
                "type": "value", "entity": vc.entity,
                "claimed": vc.claimed_value, "table": vc.table_value,
                "verdict": "source_error", "score": 0.0,
            })

    # ── Computation claims: Rule 판정 + causal attribution ──
    for cc in extraction.computation_claims:
        is_correct, actual = verify_computation(cc)

        uses_tainted = any(
            any(abs(op - iv) / max(abs(iv), 1e-10) < 0.05
                for iv in incorrect_values)
            for op in cc.operands
        )

        if is_correct and not uses_tainted:
            score = 1.0
            verdict = "correct"
        elif is_correct and uses_tainted:
            score = 0.5
            verdict = "propagated"
        elif not is_correct and uses_tainted:
            score = 0.3
            verdict = "propagated+wrong_calc"
        else:
            score = 0.0
            verdict = "source_error"

        scores.append(score)
        details.append({
            "type": "computation",
            "operands": cc.operands, "operator": cc.operator,
            "stated": cc.stated_result, "actual": round(actual, 6),
            "verdict": verdict, "score": score,
        })

        if not is_correct:
            incorrect_values.add(cc.stated_result)

    reward = sum(scores) / len(scores)
    return reward, {
        "scores": details,
        "incorrect_values": sorted(incorrect_values),
        "reward": reward,
    }


# ═══════════════════════════════════════════
# Reward Functions (GRPO interface)
# ═══════════════════════════════════════════

def extract_answer(response: str) -> str:
    # [answer]...[/answer]
    m = re.search(r'\[answer\](.*?)\[/answer\]', response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # <answer>...</answer> — last occurrence
    matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # After </think>
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        lines = [l.strip() for l in after.split('\n') if l.strip()]
        if lines:
            return lines[-1]
    return response.strip().split('\n')[-1] if response.strip() else ""


def relaxed_match(pred: str, gold: str) -> bool:
    """Binary relaxed accuracy (5% tolerance). Used for evaluation only."""
    p = re.sub(r'[,%$]', '', pred.strip())
    g = re.sub(r'[,%$]', '', gold.strip())
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) / max(abs(gf), 1e-10) <= 0.05 if gf != 0 else abs(pf) < 0.01
    except ValueError:
        return p.lower() == g.lower()


def cerm_accuracy(pred: str, gold: str) -> float:
    """Continuous Error Magnitude Reward (CERM). BigCharts-R1 method.
    Used for GRPO training reward (both baseline and ours).

    Exact match → 1.0, 5% error → 0.95, 50% error → 0.67
    """
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


def reward_outcome_only(completions: List[str], answer: str, **kw) -> List[float]:
    """Baseline: CERM accuracy + format. (BigCharts-R1 equivalent)"""
    results = []
    for r in completions:
        r_acc = cerm_accuracy(extract_answer(r), answer)
        r_fmt = 0.0
        if '<think>' in r and '</think>' in r:
            r_fmt += 0.5
        if '<answer>' in r:
            r_fmt += 0.5
        results.append(r_acc + r_fmt)
    return results


def reward_chartvr(
    completions: List[str], answer: str,
    csv_path: str = "", question: str = "",
    verifier: VerifierClient = None,
    **kw,
) -> List[float]:
    """ChartVCR reward: CERM accuracy + R_process + R_format.
    Same CERM as baseline — only R_process differs."""
    if csv_path and verifier:
        extraction_results = verifier.verify_batch_sync(csv_path, question, completions)
    else:
        extraction_results = [(ExtractionResult(), "")] * len(completions)

    results = []
    for i, resp in enumerate(completions):
        r_acc = cerm_accuracy(extract_answer(resp), answer)
        extraction, _ = extraction_results[i]
        reasoning_text = extract_reasoning(resp)
        r_proc, _ = compute_process_reward(extraction, reasoning_text)
        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.5
        if '<answer>' in resp or '[answer]' in resp:
            r_fmt += 0.5
        results.append(0.5 * r_acc + 0.3 * r_proc + 0.2 * r_fmt)
    return results


def reward_geval(
    completions: List[str], answer: str,
    csv_path: str = "", question: str = "",
    verifier: VerifierClient = None,
    **kw,
) -> List[float]:
    """G-Eval ablation (A2): CERM accuracy + holistic LLM scoring."""
    if csv_path and verifier:
        geval_scores = verifier.geval_batch_sync(csv_path, question, completions)
    else:
        geval_scores = [0.0] * len(completions)

    results = []
    for i, resp in enumerate(completions):
        r_acc = cerm_accuracy(extract_answer(resp), answer)
        r_proc = geval_scores[i]
        r_fmt = 0.0
        if '<think>' in resp and '</think>' in resp:
            r_fmt += 0.5
        if '<answer>' in resp or '[answer]' in resp:
            r_fmt += 0.5
        results.append(0.5 * r_acc + 0.3 * r_proc + 0.2 * r_fmt)
    return results
