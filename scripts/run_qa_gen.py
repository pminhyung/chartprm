#!/usr/bin/env python
"""Sequential QA generation with retry + incremental save."""
import json, os, re, time, random, sys
import pandas as pd
from openai import OpenAI

BASE = "/ex_disk2/mhpark/poc/chartvr"
CHARTS_DIR = os.path.join(BASE, "data/chartqa/train/tables")
OUTPUT_DIR = os.path.join(BASE, "data/chartvr_train")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "verified_qa.jsonl")

MAX_RETRIES = 3
RETRY_DELAY = 5

client = OpenAI(base_url="http://10.1.211.148:8000/v1", api_key="dummy", timeout=120.0)
MODEL = "Qwen3.5-397B-A17B-FP8"

with open(os.path.join(OUTPUT_DIR, "csv_eligible.json")) as f:
    csv_names = json.load(f)

MAX_CHARTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
random.seed(42)
selected = random.sample(csv_names, min(MAX_CHARTS, len(csv_names)))

# Resume: skip already processed CSVs
done_csvs = set()
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done_csvs.add(rec.get("csv_name", ""))
            except:
                pass
    print(f"Resuming: {len(done_csvs)} CSVs already done", flush=True)

remaining = [c for c in selected if c not in done_csvs]
print(f"Processing {len(remaining)} charts (skipped {len(selected)-len(remaining)})...", flush=True)

calc_re = re.compile(
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
    r"\s*([+\-*/])\s*"
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
    r"\s*[=≈]\s*"
    r"([-+]?\d+(?:,?\d{3})*(?:\.\d+)?)"
)

PROMPT_TEMPLATE = (
    "Generate exactly 3 multi-step reasoning QA pairs as JSON array from this data table.\n"
    "Each question: read 2+ values, perform 1+ calculation, exact numeric answer.\n"
    "Difficulties: medium, hard, very_hard. Use EXACT table values. Show computation steps.\n"
    'JSON: [{"question":"...","answer":number,"difficulty":"...","reasoning_steps":["Read X: val","Compute: a+b=c"]}]\n\n'
    "Data table:\n"
)

verified_count = len(done_csvs)
errors = 0
t0 = time.time()

# Open output in append mode
out_f = open(OUTPUT_PATH, "a")

for idx, csv_name in enumerate(remaining):
    csv_path = os.path.join(CHARTS_DIR, csv_name)
    try:
        df = pd.read_csv(csv_path)
        csv_text = df.head(50).to_csv(index=False)
    except Exception:
        errors += 1
        continue

    img_base = os.path.splitext(csv_name)[0]
    img_path = os.path.join(BASE, "data/chartqa/train/png", img_base + ".png")
    prompt = PROMPT_TEMPLATE + csv_text

    # Retry loop
    raw = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                errors += 1

    if not raw:
        continue

    # Parse JSON
    start = raw.find("[")
    end = raw.rfind("]")
    try:
        qa_list = json.loads(raw[start:end + 1]) if start != -1 and end != -1 else []
    except (json.JSONDecodeError, ValueError):
        qa_list = []

    # Verify each QA
    for qa in qa_list:
        if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
            continue
        steps = qa.get("reasoning_steps", [])
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
            continue

        if last_computed is not None:
            if answer_val != 0 and abs(last_computed - answer_val) / max(abs(answer_val), 1e-10) > 0.05:
                continue
            elif answer_val == 0 and abs(last_computed) >= 0.01:
                continue

        if comps_found > 0 and all_correct:
            qa.update({
                "csv_path": csv_path,
                "csv_name": csv_name,
                "image_path": img_path if os.path.exists(img_path) else "",
                "verified": True,
            })
            out_f.write(json.dumps(qa, ensure_ascii=False, default=str) + "\n")
            out_f.flush()
            verified_count += 1

    if (idx + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - idx - 1) / rate if rate > 0 else 0
        print(
            f"  [{idx+1:4d}/{len(remaining)}] verified={verified_count} errors={errors} "
            f"({elapsed:.0f}s, {rate:.1f} charts/s, ETA {eta/60:.0f}min)",
            flush=True,
        )

out_f.close()
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s: {verified_count} verified QAs (errors={errors})", flush=True)

# Count final
with open(OUTPUT_PATH) as f:
    total = sum(1 for _ in f)
print(f"Total in {OUTPUT_PATH}: {total}", flush=True)
