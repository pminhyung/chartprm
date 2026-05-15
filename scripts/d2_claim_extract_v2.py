"""D2 claim extractor v2 — LLM-primary with chart image + strict entity validation.

Replaces regex Tier 1 from v1 with Qwen3.6-27B prompt-based extraction.
Goal: spurious claim ratio 30-50% → <10% (Issue 1 §3 of failure analysis).

Multi-host async, sample-level append, resume-by-id.

Input:  data/d2_pilot/segmented_v3.jsonl  (or d1_pilot/segmented_v3.jsonl)
Output: data/d2_pilot/claims_v2.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from PIL import Image

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

EXTRACT_PROMPT = """You are extracting structured claims from a single chart-reasoning step.

Given the chart image and the reasoning step text below, extract ALL numeric or categorical claims this step makes about VISIBLE CHART ENTITIES.

STRICT RULES — reject and OMIT any claim that violates these:
1. Entity MUST be a label/category that visibly appears on the chart axes, legend, or data labels.
2. REJECT pronouns ("It", "The answer", "This value", "The result").
3. REJECT math operation names as entities ("Difference", "Average", "Ratio", "Sum", "Total", "Diff", "Mean", "Median") — these are computed results, not chart entities.
4. REJECT positional descriptors ("Rightmost bar", "Bottom", "Topmost", "Second column", "Leftmost").
5. REJECT comparative adjectives as entities ("Smaller value", "Bigger", "Smallest", "Highest", "Lowest").
6. REJECT sentence fragments. Decompose "Mali is 146.58" → entity="Mali", value=146.58. NOT entity="Second lowest is Mali".
7. If you cannot verify the entity exists ON THE CHART, OMIT that claim entirely.

Reasoning step:
\"\"\"
{step_text}
\"\"\"

Output ONLY a JSON list (no preamble, no explanation):
[{{"entity": "<exact chart label>", "value": <number or null>, "type": "value" | "categorical"}}]

If no valid claim, output [].
JSON:"""


EXTRACT_SAMPLING = {"temperature": 0.0, "max_tokens": 512}


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_claims(raw: str) -> list[dict]:
    if not raw:
        return []
    raw = raw.strip()
    # find first [...] block
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for c in data:
        if not isinstance(c, dict):
            continue
        entity = c.get("entity")
        if not entity or not isinstance(entity, str):
            continue
        entity = entity.strip()
        if not entity:
            continue
        value = c.get("value")
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
        ctype = c.get("type", "value")
        if ctype not in ("value", "categorical"):
            ctype = "value" if value is not None else "categorical"
        out.append({"entity": entity, "value": value, "type": ctype})
    return out


async def extract_step(
    client: AsyncOpenAI, model: str, step_text: str, img_b64: str,
    sem: asyncio.Semaphore,
) -> tuple[list[dict], str]:
    prompt = EXTRACT_PROMPT.format(step_text=step_text)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=EXTRACT_SAMPLING["temperature"],
                max_tokens=EXTRACT_SAMPLING["max_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return [], f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"
    claims = parse_claims(text)
    return claims, text


async def process_sample(
    rec: dict, client_picker, model: str, sem: asyncio.Semaphore,
    file_lock: asyncio.Lock, out_path: Path,
) -> int:
    try:
        img_b64 = image_to_b64(rec["image_path"])
    except Exception as e:
        out_rec = dict(rec)
        out_rec["step_claims"] = [[] for _ in rec["steps"]]
        out_rec["extract_v2_error"] = f"image_load: {type(e).__name__}: {str(e)[:160]}"
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        return 0

    futures = [extract_step(client_picker(), model, st, img_b64, sem) for st in rec["steps"]]
    results = await asyncio.gather(*futures)
    step_claims = [r[0] for r in results]

    out = dict(rec)
    out["step_claims"] = step_claims
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return sum(len(c) for c in step_claims)


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


async def run(input_path: Path, output_path: Path, ports: list[int], model: str,
              concurrent_per_host: int):
    records = [json.loads(l) for l in open(input_path) if l.strip()]
    done = load_done(output_path)
    todo = [r for r in records if r["id"] not in done]
    print(f"[extract_v2] total={len(records)} done={len(done)} todo={len(todo)}")
    if not todo:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0)
               for p in ports]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    file_lock = asyncio.Lock()
    cnt = {"i": 0}
    def pick():
        c = clients[cnt["i"] % len(clients)]
        cnt["i"] += 1
        return c

    print(f"[extract_v2] sampling={EXTRACT_SAMPLING}  hosts={len(clients)}  conc/host={concurrent_per_host}")

    tasks = [process_sample(r, pick, model, sem, file_lock, output_path) for r in todo]
    n_done = 0
    n_claims = 0
    for f in asyncio.as_completed(tasks):
        nc = await f
        n_done += 1
        n_claims += nc
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)}  claims_so_far={n_claims}", flush=True)
    # Final stats
    all_recs = [json.loads(l) for l in open(output_path) if l.strip()]
    total_claims = sum(len(c) for r in all_recs for c in r.get("step_claims", []))
    total_steps = sum(len(r.get("steps", [])) for r in all_recs)
    empty_steps = sum(1 for r in all_recs for c in r.get("step_claims", []) if not c)
    print(f"\n[extract_v2] total_records={len(all_recs)}")
    print(f"  total_claims={total_claims}  avg/step={total_claims/max(total_steps,1):.2f}")
    print(f"  empty-claim step rate={empty_steps/max(total_steps,1)*100:.1f}%")
    print(f"  -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(BASE / "data/d1_pilot/segmented_v3.jsonl"))
    ap.add_argument("--output", default=str(BASE / "data/d2_pilot/claims_v2.jsonl"))
    ap.add_argument("--ports", required=True, help="comma-separated extractor vLLM ports")
    ap.add_argument("--model", default="qwen3_6_27b_extractor")
    ap.add_argument("--concurrent_per_host", type=int, default=6)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(Path(args.input), Path(args.output), ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
