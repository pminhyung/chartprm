#!/usr/bin/env python
"""Async QA generation + answer verification with tqdm, dedup, resume."""
import asyncio
import json
import os
import re
import sys
import time
import random

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

BASE = "/ex_disk2/mhpark/poc/chartvr"
CHARTS_DIR = os.path.join(BASE, "data/chartqa/train/tables")
OUTPUT_PATH = os.path.join(BASE, "data/chartvr_train/verified_qa.jsonl")
ELIGIBLE_CACHE = os.path.join(BASE, "data/chartvr_train/csv_eligible.json")

QWEN_URL = "http://10.1.211.148:8000/v1"
MODEL = "Qwen3.5-397B-A17B-FP8"
MAX_CONCURRENT = 4
TARGET_QAS = 2200

PROMPT_TEMPLATE = (
    "Generate exactly 3 multi-step reasoning QA pairs as JSON array from this data table.\n"
    "Each question: read 2+ values, perform 1+ calculation, exact numeric answer.\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values. Show steps.\n"
    'Format: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["..."]}]\n\n'
    "Data table:\n"
)

calc_re = re.compile(
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
    r"\s*([+\-*/])\s*"
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
    r"\s*[=≈]\s*"
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
)


def verify_qa(qa: dict) -> bool:
    """Python arithmetic verification of a single QA."""
    steps = qa.get("reasoning_steps", qa.get("steps", []))
    if not steps:
        return False

    all_correct = True
    last_computed = None
    comps_found = 0

    for step in steps:
        for m in calc_re.finditer(step):
            try:
                a = float(m.group(1).replace(",", ""))
                op = m.group(2)
                b = float(m.group(3).replace(",", ""))
                stated = float(m.group(4).replace(",", ""))
                if op == "+": actual = a + b
                elif op == "-": actual = a - b
                elif op == "*": actual = a * b
                elif op == "/" and b != 0: actual = a / b
                else: continue
                comps_found += 1
                last_computed = stated
                if actual != 0 and abs(stated - actual) / abs(actual) > 0.05:
                    all_correct = False
                elif actual == 0 and abs(stated) >= 0.01:
                    all_correct = False
            except Exception:
                continue

    try:
        answer_val = float(str(qa["answer"]).replace(",", "").replace("%", ""))
    except (ValueError, TypeError):
        return False

    if last_computed is not None:
        if answer_val != 0 and abs(last_computed - answer_val) / max(abs(answer_val), 1e-10) > 0.05:
            return False
        elif answer_val == 0 and abs(last_computed) >= 0.01:
            return False

    return comps_found > 0 and all_correct


async def generate_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    csv_name: str,
    lock: asyncio.Lock,
    out_f,
    written_csvs: set,
    stats: dict,
):
    """Generate QAs for one chart, verify, and save."""
    async with sem:
        csv_path = os.path.join(CHARTS_DIR, csv_name)
        try:
            df = pd.read_csv(csv_path)
            csv_text = df.head(50).to_csv(index=False)
        except Exception:
            stats["errors"] += 1
            return

        img_base = os.path.splitext(csv_name)[0]
        img_path = os.path.join(BASE, "data/chartqa/train/png", img_base + ".png")
        prompt = PROMPT_TEMPLATE + csv_text

        # Retry
        raw = ""
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = resp.choices[0].message.content or ""
                break
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    stats["errors"] += 1

        if not raw:
            return

        # Parse JSON
        s, e = raw.find("["), raw.rfind("]")
        try:
            qa_list = json.loads(raw[s:e + 1]) if s != -1 and e != -1 else []
        except (json.JSONDecodeError, ValueError):
            qa_list = []

        # Verify and save
        for qa in qa_list:
            if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                continue
            if not verify_qa(qa):
                continue

            qa.update({
                "csv_path": csv_path,
                "csv_name": csv_name,
                "image_path": img_path if os.path.exists(img_path) else "",
                "verified": True,
            })

            async with lock:
                if csv_name not in written_csvs or True:  # allow multiple QAs per CSV
                    out_f.write(json.dumps(qa, ensure_ascii=False, default=str) + "\n")
                    out_f.flush()
                    stats["new_qa"] += 1


async def main():
    # Load eligible CSVs
    with open(ELIGIBLE_CACHE) as f:
        eligible = json.load(f)

    # Load already-done CSVs
    done_csvs = set()
    existing_qa = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_csvs.add(rec.get("csv_name", ""))
                    existing_qa += 1
                except Exception:
                    pass

    print(f"Existing: {existing_qa} QAs from {len(done_csvs)} CSVs", flush=True)

    if existing_qa >= TARGET_QAS:
        print(f"Already at target ({TARGET_QAS}). Done.", flush=True)
        return

    # Select remaining charts
    remaining = [c for c in eligible if c not in done_csvs]
    random.seed(43)
    random.shuffle(remaining)

    # Estimate how many charts needed
    needed_qa = TARGET_QAS - existing_qa
    # ~0.7 QA/chart yield
    charts_needed = min(int(needed_qa / 0.7 * 1.5), len(remaining))
    selected = remaining[:charts_needed]
    print(f"Selected {len(selected)} new charts (need ~{needed_qa} more QAs)", flush=True)

    client = AsyncOpenAI(base_url=QWEN_URL, api_key="dummy", timeout=120.0)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()
    stats = {"new_qa": 0, "errors": 0}
    written_csvs = set(done_csvs)

    out_f = open(OUTPUT_PATH, "a")

    # Create tasks
    tasks = []
    for csv_name in selected:
        tasks.append(
            generate_one(client, sem, csv_name, lock, out_f, written_csvs, stats)
        )

    # Run with tqdm
    pbar = tqdm(total=len(tasks), desc="QA Generation", unit="chart")
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
        pbar.set_postfix(
            new=stats["new_qa"],
            total=existing_qa + stats["new_qa"],
            err=stats["errors"],
        )
        # Early stop if target reached
        if existing_qa + stats["new_qa"] >= TARGET_QAS:
            pbar.set_description("Target reached!")
            break
    pbar.close()

    out_f.close()

    # Final count
    with open(OUTPUT_PATH) as f:
        total = sum(1 for _ in f)
    print(f"\nDone: +{stats['new_qa']} new, total={total} QAs (errors={stats['errors']})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
