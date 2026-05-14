"""A/B test to isolate cause of "Wait" self-correction contamination in GT.

Sends the same legacy gen_qa_v2.py prompt to two stable Qwen hosts (148, 9201)
with 3 parameter cases each and counts "Wait, let" occurrences in the response.

Case A (original gen_qa_v2 buggy): temp=0.7, enable_thinking=False, no top_p/top_k/min_p/presence_penalty
Case B (Qwen instruct compliant):  Case A + top_p=0.8, top_k=20, min_p=0
Case C (compliant + pp):           Case B + presence_penalty=1.0

Result grid tells us:
- If Case A has "Wait" but B/C don't → sampling params are root cause
- If Case B/C still have "Wait" → base model characteristic, filtering required
- If 148 vs 9201 differ → fp8 vs GPTQ-int4 quantization effect
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# Original gen_qa_v2 prompt template (exact replica from deprecated script)
PROMPT_TEMPLATE = (
    "Generate exactly 3 multi-step reasoning QA pairs as JSON array from this data table.\n"
    "Each question: read 2+ values, perform 1+ calculation, exact numeric answer.\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values. Show computation steps.\n"
    "IMPORTANT: answers must be between 0.01 and 10000. Round to 2 decimal places max.\n"
    'Format: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["Read X: val","Compute: a+b=c"]}]\n\n'
    "Data table:\n"
)

HOSTS = {
    "148": "http://10.1.211.148:8000/v1",
    "9201": "http://localhost:9201/v1",
}
MODEL = "Qwen3.5-397B-A17B-FP8"

CASES = {
    "A_original": {
        "temperature": 0.7,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "B_qwen_compliant": {
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "min_p": 0.0,
        },
    },
    "C_compliant_with_pp": {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "min_p": 0.0,
        },
    },
}

# Sample csvs — use 3 owid tables that exist
SAMPLE_CSVS = [
    "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/owid/tables/5-year-survival-rate-of-cancers-among-female-patients-in-england.csv",
    "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/owid/tables/above-ground-biomass-in-forest-per-hectare.csv",
    "/ex_disk2/mhpark/poc/chartvr/data/charts_v2/owid/tables/absolute-change-co2.csv",
]

# Patterns to detect in response reasoning
WAIT_PATTERNS = [
    re.compile(r"wait,\s*let", re.IGNORECASE),
    re.compile(r"wait,\s*re-?read", re.IGNORECASE),
    re.compile(r"wait[,.]\s+re-?verif", re.IGNORECASE),
    re.compile(r"correction\s*(during|:)", re.IGNORECASE),
    re.compile(r"let me re-?read", re.IGNORECASE),
    re.compile(r"let me re-?verify", re.IGNORECASE),
    re.compile(r"actually[,.]?\s+let", re.IGNORECASE),
    re.compile(r"hmm[,.]", re.IGNORECASE),
]

def count_patterns(text: str) -> dict:
    """Count occurrences of each contamination pattern."""
    return {p.pattern: len(p.findall(text)) for p in WAIT_PATTERNS}

def text_has_any_pattern(text: str) -> bool:
    return any(p.search(text) for p in WAIT_PATTERNS)


def load_csv_text(csv_path: str, max_rows: int = 50) -> str:
    df = pd.read_csv(csv_path)
    return df.head(max_rows).to_csv(index=False)


async def call_one(client: AsyncOpenAI, csv_text: str, params: dict):
    """Single API call with given params. Returns (raw_response, reasoning_concat)."""
    prompt = PROMPT_TEMPLATE + csv_text
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            **params,
        )
        raw = resp.choices[0].message.content or ""
        # Parse reasoning_steps from JSON array in response
        s, e = raw.find("["), raw.rfind("]")
        qa_list = []
        if s != -1 and e != -1:
            try:
                qa_list = json.loads(raw[s:e + 1])
            except Exception:
                pass
        reasoning_concat = ""
        for qa in qa_list:
            if isinstance(qa, dict):
                rs = qa.get("reasoning_steps", [])
                if isinstance(rs, list):
                    reasoning_concat += " " + " ".join(str(s) for s in rs)
                elif isinstance(rs, str):
                    reasoning_concat += " " + rs
        return raw, reasoning_concat.strip()
    except Exception as e:
        return f"ERR:{type(e).__name__}:{e}", ""


async def run_test():
    # Load csvs
    csvs = []
    for p in SAMPLE_CSVS:
        try:
            csvs.append((os.path.basename(p), load_csv_text(p)))
        except Exception as e:
            print(f"  [warn] could not load {p}: {e}")
    if len(csvs) < 1:
        print("No CSVs loaded, abort")
        sys.exit(1)
    print(f"Loaded {len(csvs)} csvs")

    REPEAT = 3  # 3 calls per csv for statistical variance
    REQUESTS_PER_CASE = len(csvs) * REPEAT  # 9 per (host, case)
    TOTAL = REQUESTS_PER_CASE * len(HOSTS) * len(CASES)
    print(f"Total calls: {TOTAL}")

    results = []  # (host, case, csv_name, reasoning, has_pattern, pattern_counts, raw_len)

    # Build tasks
    clients = {name: AsyncOpenAI(base_url=url, api_key="dummy", timeout=180) for name, url in HOSTS.items()}
    sem_per_host = {name: asyncio.Semaphore(2) for name in HOSTS}  # 2 concurrent per host

    async def one_task(host_name, case_name, csv_name, csv_text, rep):
        async with sem_per_host[host_name]:
            raw, reasoning = await call_one(clients[host_name], csv_text, CASES[case_name])
            has = text_has_any_pattern(reasoning)
            counts = count_patterns(reasoning)
            results.append({
                "host": host_name,
                "case": case_name,
                "csv": csv_name,
                "rep": rep,
                "has_pattern": has,
                "counts": counts,
                "reasoning_len": len(reasoning),
                "raw_snippet": raw[:500] if isinstance(raw, str) else str(raw)[:500],
                "reasoning_snippet": reasoning[:500],
            })
            print(f"  [{host_name}|{case_name}|{csv_name[:30]}|rep{rep}] has_pattern={has} rlen={len(reasoning)}", flush=True)

    tasks = []
    for host in HOSTS:
        for case in CASES:
            for (csv_name, csv_text) in csvs:
                for rep in range(REPEAT):
                    tasks.append(one_task(host, case, csv_name, csv_text, rep))

    print(f"\nRunning {len(tasks)} tasks ({len(HOSTS)} hosts × {len(CASES)} cases × {len(csvs)} csvs × {REPEAT} reps)...")
    await asyncio.gather(*tasks)

    # Summary grid
    print("\n" + "=" * 72)
    print("SUMMARY: % of calls with 'Wait/Correction/etc' pattern in reasoning_steps")
    print("=" * 72)
    print(f"{'host':<6s} {'case':<22s} {'n':>4s} {'has_pat':>9s} {'pat_rate':>9s}")
    print("-" * 72)

    grid = {}
    for r in results:
        k = (r["host"], r["case"])
        grid.setdefault(k, []).append(r)
    for (host, case), rs in sorted(grid.items()):
        n = len(rs)
        has = sum(1 for r in rs if r["has_pattern"])
        pct = has * 100 / n if n else 0
        print(f"{host:<6s} {case:<22s} {n:>4d} {has:>9d} {pct:>8.1f}%")

    # Save full results
    out = "/tmp/ab_test_contamination_results.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nFull results: {out}")

    # Show one sample from each case where pattern was detected
    print("\n" + "=" * 72)
    print("PATTERN-POSITIVE SAMPLES (first per host×case)")
    print("=" * 72)
    seen = set()
    for r in results:
        k = (r["host"], r["case"])
        if r["has_pattern"] and k not in seen:
            seen.add(k)
            print(f"\n[{r['host']}|{r['case']}|{r['csv'][:40]}|rep{r['rep']}]")
            print(f"  reasoning_snippet: {r['reasoning_snippet'][:400]}")


if __name__ == "__main__":
    asyncio.run(run_test())
