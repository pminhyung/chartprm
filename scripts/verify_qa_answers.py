#!/usr/bin/env python
"""Answer verification using LLM Verifier (Qwen3.5 on-premise).

Verifies each generated QA's reasoning_steps against CSV data:
- LLM checks ValueClaims (is_correct) against CSV
- Rule checks ComputationClaims (Python arithmetic)
- Both must pass for QA to be verified

Usage:
    python scripts/verify_qa_answers.py
    python scripts/verify_qa_answers.py --input data/chartvr_train/verified_qa.jsonl
"""
import asyncio
import json
import os
import re
import sys
import time

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.rewards.llm_verifier import (
    VerifierClient,
    ExtractionResult,
    verify_computation,
    compute_process_reward,
    csv_to_text,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
    _parse_json_from_response,
)

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")
QWEN_URL = "http://10.1.211.148:8000/v1"
MODEL = "Qwen3.5-397B-A17B-FP8"
MAX_CONCURRENT = 4

INPUT_PATH = os.path.join(BASE, "data/chartvr_train/verified_qa.jsonl")
OUTPUT_PATH = os.path.join(BASE, "data/chartvr_train/answer_verified.jsonl")


async def verify_one_qa(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    qa: dict,
) -> dict:
    """Verify one QA using LLM verifier + Rule arithmetic."""
    async with sem:
        csv_text = csv_to_text(qa.get("csv_path", ""))
        if not csv_text:
            return {**qa, "llm_verified": False, "verify_reason": "no_csv"}

        # Build reasoning text from steps
        steps = qa.get("reasoning_steps", qa.get("steps", []))
        reasoning = "\n".join(steps) if steps else ""
        if not reasoning:
            return {**qa, "llm_verified": False, "verify_reason": "no_steps"}

        # Call verifier
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user", "content": VERIFIER_USER.format(
                        csv_text=csv_text,
                        question=qa.get("question", ""),
                        reasoning=reasoning[:3000],
                    )},
                ],
                temperature=0.0,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = _parse_json_from_response(raw)
            extraction = ExtractionResult.model_validate(parsed)
        except Exception as e:
            return {**qa, "llm_verified": False, "verify_reason": f"api_error: {str(e)[:100]}"}

        # Check all value claims
        all_values_ok = True
        value_details = []
        for vc in extraction.value_claims:
            value_details.append({
                "entity": vc.entity,
                "claimed": vc.claimed_value,
                "table": vc.table_value,
                "correct": vc.is_correct,
            })
            if not vc.is_correct:
                all_values_ok = False

        # Check all computation claims (Rule)
        all_comps_ok = True
        comp_details = []
        for cc in extraction.computation_claims:
            is_correct, actual = verify_computation(cc)
            comp_details.append({
                "operands": cc.operands,
                "operator": cc.operator,
                "stated": cc.stated_result,
                "actual": round(actual, 6),
                "correct": is_correct,
            })
            if not is_correct:
                all_comps_ok = False

        # Overall verdict
        has_claims = len(extraction.value_claims) > 0 or len(extraction.computation_claims) > 0
        if not has_claims:
            verdict = "no_claims"
            llm_verified = False
        elif all_values_ok and all_comps_ok:
            verdict = "pass"
            llm_verified = True
        elif not all_values_ok:
            verdict = "value_mismatch"
            llm_verified = False
        else:
            verdict = "computation_error"
            llm_verified = False

        return {
            **qa,
            "llm_verified": llm_verified,
            "verify_reason": verdict,
            "value_details": value_details,
            "comp_details": comp_details,
            "verifier_raw": raw[:500],
        }


async def main():
    # Load input
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}")
        return

    with open(INPUT_PATH) as f:
        qa_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(qa_pairs)} QAs for verification", flush=True)

    # Resume: load already verified
    done_questions = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_questions.add(rec.get("question", ""))
                except Exception:
                    pass
    print(f"Already verified: {len(done_questions)}", flush=True)

    pending = [qa for qa in qa_pairs if qa.get("question", "") not in done_questions]
    print(f"Pending: {len(pending)}", flush=True)

    if not pending:
        print("All done!", flush=True)
        return

    client = AsyncOpenAI(base_url=QWEN_URL, api_key="dummy", timeout=120.0)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    out_f = open(OUTPUT_PATH, "a")
    stats = {"pass": 0, "fail": 0, "error": 0}

    # Create tasks
    tasks = [verify_one_qa(client, sem, qa) for qa in pending]

    pbar = tqdm(total=len(tasks), desc="Answer Verification", unit="qa")
    for coro in asyncio.as_completed(tasks):
        result = await coro
        pbar.update(1)

        if result.get("llm_verified"):
            stats["pass"] += 1
        elif result.get("verify_reason", "").startswith("api_error"):
            stats["error"] += 1
        else:
            stats["fail"] += 1

        # Save all results (pass and fail) for inspection
        out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        out_f.flush()

        pbar.set_postfix(
            ok=stats["pass"],
            fail=stats["fail"],
            err=stats["error"],
        )
    pbar.close()
    out_f.close()

    # Summary
    total = stats["pass"] + stats["fail"] + stats["error"]
    print(f"\nVerification complete:", flush=True)
    print(f"  PASS: {stats['pass']}/{total} ({stats['pass']/max(total,1)*100:.1f}%)", flush=True)
    print(f"  FAIL: {stats['fail']}/{total}", flush=True)
    print(f"  ERROR: {stats['error']}/{total}", flush=True)

    # Save pass-only file
    pass_only_path = os.path.join(BASE, "data/chartvr_train/answer_verified_pass.jsonl")
    with open(OUTPUT_PATH) as f:
        with open(pass_only_path, "w") as out:
            for line in f:
                rec = json.loads(line)
                if rec.get("llm_verified"):
                    out.write(line)

    pass_count = sum(1 for _ in open(pass_only_path))
    print(f"  Pass-only saved: {pass_only_path} ({pass_count} QAs)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
