#!/usr/bin/env python
"""Unified QA generation script.

Multi-host round-robin + async parallel + per-host concurrency.
Supports resume, tqdm, append-mode JSONL.

Usage:
  # Generate QA from eligible charts
  python scripts/generate_qa.py generate --eligible data/charts_v2/qa_gen_eligible.json

  # Generate with specific prompt type (text, scientific, template, edge_case)
  python scripts/generate_qa.py generate --eligible data/block_c.json --prompt scientific

  # Difficulty filter (keep 30-70% accuracy)
  python scripts/generate_qa.py filter --input data/generated_qa.jsonl --output data/filtered.jsonl

  # Custom hosts
  python scripts/generate_qa.py generate --hosts "http://host1:8000/v1,http://host2:8000/v1"
"""
import argparse
import asyncio
import json
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.config import ONPREM_HOSTS, ONPREM_MODEL, ONPREM_MAX_CONCURRENT_PER_HOST
from chartvr.llm_client import MultiHostClient
from chartvr.verification import verify_qa
from chartvr.prompts import get_prompt


# ═══════════════════════════════════════════
# Generate
# ═══════════════════════════════════════════

async def generate_one(client, chart, lock, out_f, written_keys, stats, prompt_text):
    """Generate QAs for one chart."""
    csv_path = chart.get("csv_path", "")
    try:
        df = pd.read_csv(csv_path)
        csv_text = df.head(50).to_csv(index=False)
    except Exception:
        stats["errors"] += 1
        return

    prompt = prompt_text + csv_text
    try:
        resp = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        stats["errors"] += 1
        return

    if not raw:
        return

    # Parse JSON array
    s, e = raw.find("["), raw.rfind("]")
    try:
        qa_list = json.loads(raw[s:e + 1]) if s != -1 and e != -1 else []
    except (json.JSONDecodeError, ValueError):
        qa_list = []

    for qa in qa_list:
        if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
            continue
        if not verify_qa(qa):
            continue

        try:
            qa["answer"] = round(float(str(qa["answer"]).replace(",", "").replace("%", "")), 2)
        except (ValueError, TypeError):
            continue

        key = f"{chart.get('slug', '')}_{qa['question'][:50]}"
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


async def run_generate(args):
    with open(args.eligible) as f:
        charts = json.load(f)
    print(f"Loaded {len(charts)} charts", flush=True)

    # Resume
    written_keys, done_slugs, existing_qa = set(), set(), 0
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    written_keys.add(f"{rec.get('chart_slug', '')}_{rec.get('question', '')[:50]}")
                    done_slugs.add(rec.get("chart_slug", ""))
                    existing_qa += 1
                except Exception:
                    pass

    pending = [c for c in charts if c.get("slug", "") not in done_slugs]
    if not pending:
        print("All done!", flush=True)
        return

    # Client setup
    hosts = args.hosts.split(",") if args.hosts else None
    client = MultiHostClient(hosts=hosts, max_concurrent_per_host=args.max_concurrent_per_host)
    prompt_text = get_prompt(args.prompt)
    lock = asyncio.Lock()
    stats = {"new_qa": 0, "errors": 0}

    print(f"Hosts: {client.n_hosts}, Max concurrent: {client.max_concurrent}", flush=True)
    print(f"Prompt: {args.prompt}, Existing: {existing_qa}, Pending: {len(pending)}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a")
    tasks = [generate_one(client, c, lock, out_f, written_keys, stats, prompt_text)
             for c in pending]

    pbar = tqdm(total=len(tasks), desc=f"QA Gen ({args.prompt})", unit="chart")
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
        pbar.set_postfix(new=stats["new_qa"], total=existing_qa + stats["new_qa"], err=stats["errors"])
    pbar.close()
    out_f.close()

    total = sum(1 for _ in open(args.output))
    print(f"\nDone: +{stats['new_qa']}, total={total} QAs (errors={stats['errors']})", flush=True)

    # Stats by source
    with open(args.output) as f:
        sources = {}
        for line in f:
            try:
                rec = json.loads(line)
                s = rec.get("source", "?")
                sources[s] = sources.get(s, 0) + 1
            except:
                pass
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}", flush=True)


# ═══════════════════════════════════════════
# Difficulty Filter
# ═══════════════════════════════════════════

async def run_filter(args):
    """Keep questions where model accuracy is 30-70%."""
    with open(args.input) as f:
        qa_pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(qa_pairs)} QAs for difficulty filtering", flush=True)

    hosts = args.hosts.split(",") if args.hosts else None
    client = MultiHostClient(hosts=hosts, max_concurrent_per_host=args.max_concurrent_per_host)
    num_attempts = args.num_attempts

    _counter = {"n": 0}

    async def attempt_one(qa):
        """One zero-shot attempt."""
        _counter["n"] += 1
        csv_text = ""
        try:
            df = pd.read_csv(qa["csv_path"])
            csv_text = df.head(50).to_csv(index=False)
        except:
            return False

        try:
            resp = await client.chat(
                messages=[
                    {"role": "system", "content": "You are a chart analyst. Answer using the data. Put answer in [answer][/answer] tags."},
                    {"role": "user", "content": f"Data table:\n{csv_text}\n\nQuestion: {qa['question']}"},
                ],
                max_tokens=500, temperature=0.3,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            m = re.search(r'\[answer\](.*?)\[/answer\]', raw, re.DOTALL)
            pred = m.group(1).strip() if m else raw.strip().split('\n')[-1]

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

    async def evaluate_qa(qa):
        results = await asyncio.gather(*[attempt_one(qa) for _ in range(num_attempts)])
        return sum(results)

    filtered = []
    batch_sz = 100
    for batch_start in range(0, len(qa_pairs), batch_sz):
        batch = qa_pairs[batch_start:batch_start + batch_sz]
        counts = await asyncio.gather(*[evaluate_qa(qa) for qa in batch])
        for qa, correct_count in zip(batch, counts):
            accuracy = correct_count / num_attempts
            qa["difficulty_accuracy"] = accuracy
            if 0.3 <= accuracy <= 0.7:
                filtered.append(qa)
        print(f"  Filter: {batch_start + len(batch)}/{len(qa_pairs)}, kept {len(filtered)}", flush=True)

    print(f"\nDifficulty filter: {len(filtered)}/{len(qa_pairs)} passed", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        for qa in filtered:
            f.write(json.dumps(qa, ensure_ascii=False, default=str) + '\n')
    print(f"Saved to {args.output}", flush=True)


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ChartVCR QA Generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate QA pairs from charts")
    gen.add_argument("--eligible", required=True, help="Path to eligible charts JSON")
    gen.add_argument("--output", default="data/generated_qa.jsonl")
    gen.add_argument("--prompt", default="numeric", choices=["numeric", "text", "template", "scientific", "edge_case"])
    gen.add_argument("--hosts", default=None, help="Comma-separated host URLs")
    gen.add_argument("--max_concurrent_per_host", type=int, default=ONPREM_MAX_CONCURRENT_PER_HOST)

    # filter
    filt = sub.add_parser("filter", help="Difficulty filter (keep 30-70% accuracy)")
    filt.add_argument("--input", required=True, help="Input JSONL")
    filt.add_argument("--output", required=True, help="Output JSONL")
    filt.add_argument("--num_attempts", type=int, default=5)
    filt.add_argument("--hosts", default=None)
    filt.add_argument("--max_concurrent_per_host", type=int, default=ONPREM_MAX_CONCURRENT_PER_HOST)

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(run_generate(args))
    elif args.command == "filter":
        asyncio.run(run_filter(args))


if __name__ == "__main__":
    main()
