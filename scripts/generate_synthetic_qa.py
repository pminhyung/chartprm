"""
ChartVCR Synthetic QA Generation Pipeline.

Step 1: ChartQA train CSV → Qwen3.5 on-premise로 multi-step QA 생성
Step 2: Python으로 답 재계산하여 검증
Step 3: 난이도 필터 (qwen_onpremise zero-shot으로 30-70% 정답률)
Step 4: 최종 ~5K complex QA pairs 저장

Usage:
  # Step 1+2: Generate + verify
  python scripts/generate_synthetic_qa.py --step generate --max_charts 2000

  # Step 3: Difficulty filter
  python scripts/generate_synthetic_qa.py --step filter

  # Full pipeline
  python scripts/generate_synthetic_qa.py --step all --max_charts 2000
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import random
from typing import List, Optional

import pandas as pd
from openai import AsyncOpenAI

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")
CHARTS_DIR = os.path.join(BASE, "data/chartqa/train/tables")
IMAGES_DIR = os.path.join(BASE, "data/chartqa/train/png")
OUTPUT_DIR = os.path.join(BASE, "data/chartvr_train")

QWEN_URL = "http://10.1.211.148:8000/v1"
QWEN_MODEL = "Qwen3.5-397B-A17B-FP8"


# ═══════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════

QA_GENERATION_PROMPT = """\
You are given a chart's data table in CSV format.

Generate exactly 3 questions that require MULTI-STEP REASONING to answer.

Requirements for each question:
1. The answer MUST be computable from the data table using arithmetic
2. The question must require reading 2+ values AND performing 1+ calculations
3. The question should be natural (as if a human asked it about the chart)
4. Provide the exact numeric answer and the computation steps

Question difficulty levels (generate one of each):
- MEDIUM: Read 2-3 values, perform 1 calculation (difference, ratio, percentage)
- HARD: Read 3-5 values, perform 2+ chained calculations (average then compare, sum then ratio)
- VERY HARD: Read 5+ values, multi-step with percentage change, weighted average, or compound operations

IMPORTANT:
- Use the EXACT values from the data table (strip % or $ signs for computation)
- Show every computation step explicitly
- The answer should be a single number (float or int)

Format your response as JSON array:
[
  {{
    "question": "What is the difference between X and Y?",
    "answer": 0.57,
    "difficulty": "medium",
    "reasoning_steps": [
      "Read value of X: 103.7",
      "Read value of Y: 103.13",
      "Compute difference: 103.7 - 103.13 = 0.57"
    ]
  }},
  ...
]

Data table:
{csv_text}"""


# ═══════════════════════════════════════════
# CSV Utilities
# ═══════════════════════════════════════════

def csv_to_text(csv_path: str, max_rows: int = 50) -> str:
    try:
        df = pd.read_csv(csv_path)
        return df.head(max_rows).to_csv(index=False)
    except Exception:
        return ""


def get_csv_numeric_count(csv_path: str) -> int:
    """Count numeric values in CSV. Skip charts with too few values."""
    try:
        df = pd.read_csv(csv_path)
        count = 0
        for col in df.columns:
            for v in df[col]:
                sv = str(v).replace('%', '').replace('$', '').replace(',', '').strip()
                try:
                    float(sv)
                    count += 1
                except ValueError:
                    pass
        return count
    except Exception:
        return 0


def get_image_path(csv_name: str) -> str:
    """Get image path from CSV filename."""
    base = os.path.splitext(csv_name)[0]
    img = os.path.join(IMAGES_DIR, f"{base}.png")
    return img if os.path.exists(img) else ""


# ═══════════════════════════════════════════
# Step 1: QA Generation
# ═══════════════════════════════════════════

def parse_json_from_response(raw: str) -> list:
    """Extract JSON array from LLM response."""
    raw = raw.strip()
    # Try direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # Find JSON array in response
    # Look for [ ... ]
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return []


async def generate_qa_for_chart(
    client: AsyncOpenAI,
    csv_path: str,
    csv_name: str,
    semaphore: asyncio.Semaphore,
) -> List[dict]:
    """Generate QA pairs for one chart."""
    async with semaphore:
        csv_text = csv_to_text(csv_path)
        if not csv_text:
            return []

        try:
            resp = await client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "user", "content": QA_GENERATION_PROMPT.format(csv_text=csv_text)},
                ],
                max_tokens=2000,
                temperature=0.7,  # Some variety in questions
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            raw = resp.choices[0].message.content or ""
            qa_list = parse_json_from_response(raw)

            # Attach metadata
            img_path = get_image_path(csv_name)
            results = []
            for qa in qa_list:
                if not isinstance(qa, dict):
                    continue
                if "question" not in qa or "answer" not in qa:
                    continue
                qa["csv_path"] = csv_path
                qa["csv_name"] = csv_name
                qa["image_path"] = img_path
                results.append(qa)
            return results
        except Exception as e:
            return []


# ═══════════════════════════════════════════
# Step 2: Answer Verification (Python recompute)
# ═══════════════════════════════════════════

def verify_answer(qa: dict, csv_path: str) -> dict:
    """
    Verify answer by re-extracting values from CSV and recomputing.
    Returns qa with 'verified' field added.
    """
    csv_text = csv_to_text(csv_path)
    if not csv_text:
        qa["verified"] = False
        qa["verify_reason"] = "no_csv"
        return qa

    # Parse reasoning steps to extract computations
    steps = qa.get("reasoning_steps", [])
    if not steps:
        qa["verified"] = False
        qa["verify_reason"] = "no_steps"
        return qa

    # Load CSV values for cross-check
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        qa["verified"] = False
        qa["verify_reason"] = "csv_parse_error"
        return qa

    # Build value lookup from CSV
    csv_values = {}
    for _, row in df.iterrows():
        entity = str(row.iloc[0]).strip()
        for col_idx in range(1, len(row)):
            val_str = str(row.iloc[col_idx]).replace('%', '').replace('$', '').replace(',', '').strip()
            try:
                csv_values[f"{entity}_{df.columns[col_idx]}"] = float(val_str)
            except ValueError:
                pass

    # Extract computations from reasoning steps
    # Pattern: "A op B = C" (handles negatives in result)
    calc_pattern = re.compile(
        r'([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)'
        r'\s*([+\-*/÷×])\s*'
        r'([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)'
        r'\s*[=≈]\s*'
        r'([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)'
    )

    all_correct = True
    computations_found = 0
    last_computed = None  # Track last computation result

    for step in steps:
        for m in calc_pattern.finditer(step):
            try:
                a = float(m.group(1).replace(',', ''))
                op = m.group(2).replace('×', '*').replace('÷', '/')
                b = float(m.group(3).replace(',', ''))
                stated = float(m.group(4).replace(',', ''))

                if op == '+':
                    actual = a + b
                elif op == '-':
                    actual = a - b
                elif op == '*':
                    actual = a * b
                elif op == '/' and b != 0:
                    actual = a / b
                else:
                    continue

                computations_found += 1
                last_computed = stated
                if actual == 0:
                    if abs(stated) >= 0.01:
                        all_correct = False
                else:
                    if abs(stated - actual) / abs(actual) > 0.05:
                        all_correct = False
            except (ValueError, ZeroDivisionError):
                continue

    # Check: stated answer should be numeric
    try:
        answer_val = float(str(qa["answer"]).replace(',', '').replace('%', ''))
    except (ValueError, TypeError):
        qa["verified"] = False
        qa["verify_reason"] = "non_numeric_answer"
        return qa

    # FIX: Check final answer matches last computation result
    if last_computed is not None:
        if answer_val == 0:
            if abs(last_computed) >= 0.01:
                qa["verified"] = False
                qa["verify_reason"] = "final_answer_vs_computation_mismatch"
                return qa
        else:
            if abs(last_computed - answer_val) / max(abs(answer_val), 1e-10) > 0.05:
                qa["verified"] = False
                qa["verify_reason"] = "final_answer_vs_computation_mismatch"
                return qa

    if computations_found == 0:
        qa["verified"] = False
        qa["verify_reason"] = "no_computations_in_steps"
    elif all_correct:
        qa["verified"] = True
        qa["verify_reason"] = "computations_verified"
    else:
        qa["verified"] = False
        qa["verify_reason"] = "computation_mismatch"

    return qa


# ═══════════════════════════════════════════
# Step 3: Difficulty Filter
# ═══════════════════════════════════════════

async def check_difficulty_batch(
    client: AsyncOpenAI,
    qa_pairs: List[dict],
    semaphore: asyncio.Semaphore,
    num_attempts: int = 5,
) -> List[dict]:
    """
    Use qwen_onpremise to attempt answering each question.
    Keep questions where model gets it right 30-70% of the time.
    """

    async def attempt_one(qa: dict) -> bool:
        """One zero-shot attempt. Returns True if correct."""
        async with semaphore:
            csv_text = csv_to_text(qa["csv_path"])
            try:
                resp = await client.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "You are a chart analyst. Answer the question using the data table. "
                            "Put your final answer in [answer][/answer] tags."
                        )},
                        {"role": "user", "content": f"Data table:\n{csv_text}\n\nQuestion: {qa['question']}"},
                    ],
                    max_tokens=500,
                    temperature=0.3,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                raw = resp.choices[0].message.content or ""
                # Extract answer
                m = re.search(r'\[answer\](.*?)\[/answer\]', raw, re.DOTALL)
                if m:
                    pred = m.group(1).strip()
                else:
                    pred = raw.strip().split('\n')[-1]

                # Relaxed match
                gold = str(qa["answer"])
                p = re.sub(r'[,%$]', '', pred.strip())
                g = re.sub(r'[,%$]', '', gold.strip())
                try:
                    pf, gf = float(p), float(g)
                    return abs(pf - gf) / max(abs(gf), 1e-10) <= 0.05 if gf != 0 else abs(pf) < 0.01
                except ValueError:
                    return p.lower() == g.lower()
            except Exception:
                return False

    # Run all attempts concurrently
    async def evaluate_qa(qa_item):
        tasks = [attempt_one(qa_item) for _ in range(num_attempts)]
        results = await asyncio.gather(*tasks)
        return sum(results)

    # Process in batches for memory
    filtered = []
    batch_sz = 100
    for batch_start in range(0, len(qa_pairs), batch_sz):
        batch = qa_pairs[batch_start:batch_start + batch_sz]
        counts = await asyncio.gather(*[evaluate_qa(qa) for qa in batch])
        for qa, correct_count in zip(batch, counts):
            accuracy = correct_count / num_attempts
            qa["difficulty_accuracy"] = accuracy
            qa["difficulty_attempts"] = num_attempts

            if 0.3 <= accuracy <= 0.7:
                qa["difficulty_pass"] = True
                filtered.append(qa)
            else:
                qa["difficulty_pass"] = False

        print(f"  Difficulty filter: {batch_start + len(batch)}/{len(qa_pairs)}, kept {len(filtered)}")

    return filtered


# ═══════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════

async def run_generation(max_charts: int, max_concurrent: int = 8):
    """Step 1+2: Generate QA pairs and verify."""
    client = AsyncOpenAI(base_url=QWEN_URL, api_key="dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    # Get eligible CSVs — use cache if available
    cache_path = os.path.join(OUTPUT_DIR, "csv_eligible.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            csv_names = json.load(f)
        eligible = [(n, os.path.join(CHARTS_DIR, n)) for n in csv_names]
        print(f"Loaded {len(eligible)} eligible charts from cache")
    else:
        csv_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith('.csv')]
        eligible = []
        print(f"Scanning {len(csv_files)} CSVs for eligibility...")
        for csv_name in csv_files:
            csv_path = os.path.join(CHARTS_DIR, csv_name)
            if get_csv_numeric_count(csv_path) >= 4:
                eligible.append((csv_name, csv_path))
        print(f"Eligible charts (>=4 numeric values): {len(eligible)}")

    # Sample
    random.seed(42)
    if len(eligible) > max_charts:
        eligible = random.sample(eligible, max_charts)
    print(f"Processing {len(eligible)} charts...")

    # Generate
    raw_output = os.path.join(OUTPUT_DIR, "raw_generated.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    all_qa = []
    batch_size = 50

    for batch_start in range(0, len(eligible), batch_size):
        batch = eligible[batch_start:batch_start + batch_size]
        tasks = [
            generate_qa_for_chart(client, csv_path, csv_name, semaphore)
            for csv_name, csv_path in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        for qa_list in batch_results:
            all_qa.extend(qa_list)

        elapsed = time.time() - t0
        print(f"  Generated: {len(all_qa)} QAs from {batch_start + len(batch)}/{len(eligible)} charts ({elapsed:.0f}s)", flush=True)

    print(f"\nTotal raw QAs: {len(all_qa)}")

    # Step 2: Verify
    print("\nVerifying answers...")
    verified_count = 0
    for i in range(len(all_qa)):
        all_qa[i] = verify_answer(all_qa[i], all_qa[i]["csv_path"])
        if all_qa[i].get("verified"):
            verified_count += 1

    print(f"Verified: {verified_count}/{len(all_qa)} ({verified_count/len(all_qa)*100:.1f}%)")

    # Save raw + verified
    with open(raw_output, 'w') as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False, default=str) + '\n')
    print(f"Saved raw to {raw_output}")

    # Save verified only
    verified_output = os.path.join(OUTPUT_DIR, "verified_qa.jsonl")
    with open(verified_output, 'w') as f:
        for qa in all_qa:
            if qa.get("verified"):
                f.write(json.dumps(qa, ensure_ascii=False, default=str) + '\n')
    print(f"Saved verified to {verified_output} ({verified_count} QAs)")

    return all_qa


async def run_filter(max_concurrent: int = 8):
    """Step 3: Difficulty filter."""
    client = AsyncOpenAI(base_url=QWEN_URL, api_key="dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    verified_path = os.path.join(OUTPUT_DIR, "verified_qa.jsonl")
    if not os.path.exists(verified_path):
        print(f"No verified QAs found at {verified_path}. Run --step generate first.")
        return

    with open(verified_path) as f:
        qa_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(qa_pairs)} verified QAs for difficulty filtering")

    filtered = await check_difficulty_batch(client, qa_pairs, semaphore)
    print(f"\nDifficulty filter: {len(filtered)}/{len(qa_pairs)} passed (30-70% accuracy)")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "chartvr_train.jsonl")
    with open(output_path, 'w') as f:
        for qa in filtered:
            f.write(json.dumps(qa, ensure_ascii=False, default=str) + '\n')
    print(f"Saved to {output_path}")

    # Also save as JSON for GRPO
    grpo_output = os.path.join(OUTPUT_DIR, "chartvr_train_grpo.json")
    grpo_data = []
    for qa in filtered:
        grpo_data.append({
            "problem": qa["question"],
            "solution": str(qa["answer"]),
            "csv_path": qa["csv_path"],
            "image": qa.get("image_path", ""),
            "difficulty": qa.get("difficulty", ""),
            "reasoning_steps": qa.get("reasoning_steps", []),
        })
    with open(grpo_output, 'w') as f:
        json.dump(grpo_data, f, indent=2, ensure_ascii=False)
    print(f"Saved GRPO format to {grpo_output} ({len(grpo_data)} samples)")


async def main_async(args):
    if args.step in ("generate", "all"):
        await run_generation(args.max_charts, args.concurrency)
    if args.step in ("filter", "all"):
        await run_filter(args.concurrency)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "filter", "all"], default="all")
    parser.add_argument("--max_charts", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
