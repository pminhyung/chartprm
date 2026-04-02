#!/usr/bin/env python
"""Async QA generation for new_charts (OWID/WorldBank/Synthetic).

Multi-host round-robin + async parallel requests + per-host concurrency limit.
Supports resume, tqdm, append-mode JSONL.
"""
import argparse
import asyncio
import json
import os
import re
import time

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

BASE = "/ex_disk2/mhpark/poc/chartvr"
ELIGIBLE_PATH = os.path.join(BASE, "data/new_charts/qa_gen_eligible.json")
OUTPUT_PATH = os.path.join(BASE, "data/new_charts/generated_qa.jsonl")

DEFAULT_HOSTS = [
    "http://10.1.211.147:8000/v1",
    "http://10.1.211.148:8000/v1",
    "http://10.1.211.169:8000/v1",
    "http://10.1.211.170:8000/v1",
]
MODEL = "Qwen3.5-397B-A17B-FP8"
MAX_CONCURRENT_PER_HOST = 5

PROMPT_TEMPLATE = (
    "Generate exactly 3 multi-step reasoning QA pairs as JSON array from this data table.\n"
    "Each question: read 2+ values, perform 1+ calculation, exact numeric answer.\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values. Show computation steps.\n"
    'Format: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["Read X: val","Compute: a+b=c"]}]\n\n'
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
    """Python arithmetic verification."""
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
    clients: list,
    idx: int,
    sem: asyncio.Semaphore,
    chart: dict,
    lock: asyncio.Lock,
    out_f,
    written_keys: set,
    stats: dict,
):
    """Generate QAs for one chart."""
    client = clients[idx % len(clients)]  # Round-robin load balancing
    async with sem:
        csv_path = chart["csv_path"]
        try:
            df = pd.read_csv(csv_path)
            csv_text = df.head(50).to_csv(index=False)
        except Exception:
            stats["errors"] += 1
            return

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

            # Dedup key
            key = f"{chart['slug']}_{qa['question'][:50]}"
            qa.update({
                "csv_path": csv_path,
                "image_path": chart.get("image_path", ""),
                "source": chart.get("source", ""),
                "chart_slug": chart.get("slug", ""),
                "verified": True,
            })

            async with lock:
                if key not in written_keys:
                    written_keys.add(key)
                    out_f.write(json.dumps(qa, ensure_ascii=False, default=str) + "\n")
                    out_f.flush()
                    stats["new_qa"] += 1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=str, default=None,
                        help="Comma-separated host URLs (default: 4 on-premise hosts)")
    parser.add_argument("--eligible", type=str, default=ELIGIBLE_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--max_concurrent_per_host", type=int, default=MAX_CONCURRENT_PER_HOST)
    args = parser.parse_args()

    hosts = args.hosts.split(",") if args.hosts else DEFAULT_HOSTS
    output_path = args.output

    with open(args.eligible) as f:
        charts = json.load(f)
    print(f"Loaded {len(charts)} charts for QA generation", flush=True)

    # Resume
    written_keys = set()
    existing_qa = 0
    done_slugs = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:50]}"
                    written_keys.add(key)
                    done_slugs.add(rec.get("chart_slug", ""))
                    existing_qa += 1
                except Exception:
                    pass

    pending = [c for c in charts if c.get("slug", "") not in done_slugs]

    if not pending:
        print("All done!", flush=True)
        return

    # Multi-host setup
    clients = [AsyncOpenAI(base_url=url, api_key="dummy", timeout=120.0) for url in hosts]
    sem = asyncio.Semaphore(len(hosts) * args.max_concurrent_per_host)
    lock = asyncio.Lock()
    stats = {"new_qa": 0, "errors": 0}

    print(f"Hosts: {len(hosts)}, Max concurrent: {len(hosts) * args.max_concurrent_per_host}", flush=True)
    print(f"Existing: {existing_qa} QAs, Pending: {len(pending)} charts", flush=True)

    out_f = open(output_path, "a")
    tasks = [generate_one(clients, i, sem, c, lock, out_f, written_keys, stats)
             for i, c in enumerate(pending)]

    pbar = tqdm(total=len(tasks), desc="QA Generation", unit="chart")
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
        pbar.set_postfix(new=stats["new_qa"], total=existing_qa + stats["new_qa"], err=stats["errors"])
    pbar.close()
    out_f.close()

    total = sum(1 for _ in open(output_path))
    print(f"\nDone: +{stats['new_qa']} new, total={total} QAs (errors={stats['errors']})", flush=True)

    # Stats by source
    with open(output_path) as f:
        sources = {}
        for line in f:
            rec = json.loads(line)
            s = rec.get("source", "?")
            sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
