"""Day 1 verifier reliability check — CSV-based ground truth.

Picks N ChartQA-train samples with CSV ground truth, queries verifier with K entities each,
checks agreement with CSV value at 10% tolerance.

PASS criteria: coverage ≥80% AND agreement ≥85% (10% rel diff).

Multi-host async, sample-level append, resume by (sample_id, entity).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
CHARTQA_TRAIN_TABLES = BASE / "data/chartqa/train/tables"
CHARTQA_TRAIN_PNG = BASE / "data/chartqa/train/png"


VERIFY_PROMPT = """Looking at this chart, locate the entity labeled exactly '{entity}' on the chart.

If you find it, reply with ONLY its associated numeric value (no units, no explanation).
If '{entity}' is not visible on the chart, reply EXACTLY: NOT_FOUND

Reply:"""


SAMPLING = {"temperature": 0.0, "max_tokens": 64}


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_value(cell: str) -> float | None:
    if cell is None:
        return None
    try:
        return float(str(cell).strip().replace(",", "").replace("%", "").replace("$", ""))
    except (ValueError, AttributeError):
        return None


def load_csv_entities(csv_path: Path, max_entities: int = 4,
                      rng: random.Random | None = None) -> list[tuple[str, float]]:
    """Extract (entity, value) pairs. Long format if 2 cols; wide format else."""
    try:
        rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    except Exception:
        return []
    if len(rows) < 2:
        return []
    headers = rows[0]
    pairs: list[tuple[str, float]] = []
    if len(headers) == 2:
        for r in rows[1:]:
            if len(r) >= 2:
                v = parse_value(r[1])
                if v is not None and r[0].strip():
                    pairs.append((r[0].strip(), v))
    else:
        for col_idx in range(1, len(headers)):
            col_label = headers[col_idx].strip()
            for r in rows[1:]:
                if len(r) <= col_idx:
                    continue
                v = parse_value(r[col_idx])
                row_label = r[0].strip() if r else ""
                if v is not None and (row_label or col_label):
                    if row_label and col_label:
                        entity = f"{row_label} ({col_label})"
                    else:
                        entity = row_label or col_label
                    pairs.append((entity, v))
    if rng is not None:
        rng.shuffle(pairs)
    return pairs[:max_entities]


async def query_one(client: AsyncOpenAI, model: str, entity: str, img_b64: str,
                    sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": VERIFY_PROMPT.format(entity=entity)},
                ]}],
                temperature=SAMPLING["temperature"],
                max_tokens=SAMPLING["max_tokens"],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"


async def process_sample(sid: str, image_path: Path, entities: list[tuple[str, float]],
                         client_picker, model: str, sem: asyncio.Semaphore,
                         file_lock: asyncio.Lock, out_path: Path):
    try:
        img_b64 = image_to_b64(str(image_path))
    except Exception as e:
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps({"sample_id": sid, "error": f"image_load: {e}"}) + "\n")
        return

    futures = [query_one(client_picker(), model, ent, img_b64, sem) for ent, _ in entities]
    responses = await asyncio.gather(*futures)
    rows = []
    for (ent, gt), raw in zip(entities, responses):
        up = raw.upper()
        if "NOT_FOUND" in up or "NOT FOUND" in up:
            rows.append({"sample_id": sid, "entity": ent, "gt": gt, "response": raw,
                         "extracted": None, "status": "not_found", "agree": None})
            continue
        if raw.startswith("__ERROR__"):
            rows.append({"sample_id": sid, "entity": ent, "gt": gt, "response": raw,
                         "extracted": None, "status": "error", "agree": None})
            continue
        ext = parse_number(raw)
        if ext is None:
            rows.append({"sample_id": sid, "entity": ent, "gt": gt, "response": raw,
                         "extracted": None, "status": "parse_fail", "agree": None})
            continue
        rel_diff = abs(ext - gt) / max(abs(gt), 1e-10)
        abs_diff = abs(ext - gt)
        agree = (rel_diff <= 0.10) or (abs_diff <= 1.0)
        rows.append({"sample_id": sid, "entity": ent, "gt": gt, "response": raw,
                     "extracted": ext, "rel_diff": rel_diff, "abs_diff": abs_diff,
                     "status": "ok", "agree": bool(agree)})
    async with file_lock:
        with open(out_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
    return done


async def run(n_samples: int, max_entities: int, ports: list[int], model: str,
              concurrent_per_host: int, output_path: Path, seed: int = 42):
    rng = random.Random(seed)
    all_csv = sorted(CHARTQA_TRAIN_TABLES.glob("*.csv"))
    if not all_csv:
        print(f"[reliability] FATAL: no CSV under {CHARTQA_TRAIN_TABLES}")
        return
    rng.shuffle(all_csv)

    todo: list[tuple[str, Path, list[tuple[str, float]]]] = []
    done = load_done(output_path)
    for csv_path in all_csv:
        sid = csv_path.stem
        full_id = f"chartqa_train_{sid}"
        if full_id in done:
            continue
        img = CHARTQA_TRAIN_PNG / f"{sid}.png"
        if not img.exists():
            continue
        ents = load_csv_entities(csv_path, max_entities=max_entities, rng=rng)
        if not ents:
            continue
        todo.append((full_id, img, ents))
        if len(todo) >= n_samples - len(done):
            break

    print(f"[reliability] target_samples={n_samples} done={len(done)} todo={len(todo)} (entities/sample≤{max_entities})")
    if not todo:
        analyze(output_path)
        return

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

    print(f"[reliability] hosts={len(clients)} conc/host={concurrent_per_host}")
    tasks = [process_sample(sid, img, ents, pick, model, sem, file_lock, output_path)
             for (sid, img, ents) in todo]
    n_done = 0
    for f in asyncio.as_completed(tasks):
        await f
        n_done += 1
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)}", flush=True)

    analyze(output_path)


def analyze(out_path: Path):
    rows = [json.loads(l) for l in open(out_path) if l.strip()]
    rows = [r for r in rows if "status" in r]
    if not rows:
        print("[reliability] no rows to analyze")
        return
    n_total = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    nf = [r for r in rows if r["status"] == "not_found"]
    pf = [r for r in rows if r["status"] == "parse_fail"]
    er = [r for r in rows if r["status"] == "error"]
    coverage = len(ok) / n_total * 100
    agreement = sum(1 for r in ok if r["agree"]) / max(len(ok), 1) * 100
    not_found_pct = len(nf) / n_total * 100
    pass_check = (coverage >= 80.0) and (agreement >= 85.0)
    summary = {
        "n_total": n_total,
        "ok": len(ok),
        "not_found": len(nf),
        "parse_fail": len(pf),
        "error": len(er),
        "coverage_pct": coverage,
        "agreement_pct": agreement,
        "not_found_pct": not_found_pct,
        "pass": pass_check,
        "criteria": "coverage>=80 AND agreement>=85 (10% rel or 1.0 abs)",
    }
    print(f"\n[reliability] N={n_total}")
    print(f"  OK (parsed): {len(ok)} ({coverage:.1f}%)")
    print(f"  NOT_FOUND  : {len(nf)} ({not_found_pct:.1f}%)")
    print(f"  parse_fail : {len(pf)}")
    print(f"  error      : {len(er)}")
    print(f"  agreement (on parsed): {agreement:.1f}%")
    print(f"  VERDICT: {'PASS' if pass_check else 'FAIL'}  (coverage>=80, agreement>=85)")
    out_json = out_path.with_suffix(".summary.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"  summary -> {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="comma-separated verifier vLLM ports")
    ap.add_argument("--model", default="internvl_verifier")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--max_entities", type=int, default=4)
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    ap.add_argument("--output", default=str(BASE / "data/d1_pilot/verifier_reliability.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(args.n_samples, args.max_entities, ports, args.model,
                    args.concurrent_per_host, Path(args.output), args.seed))


if __name__ == "__main__":
    sys.exit(main() or 0)
