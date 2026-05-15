"""D2 perception verifier — image re-query with Qwen3.5-VL-9B (2-host).

Guide §3.6: for each extracted claim, query the chart image with a separate
verifier VLM, parse the response, compare against the claimed value with
tolerance (5% rel or 0.5 abs → 1.0, ≤15% rel → 0.5, else 0.0).

Input:  data/d2_pilot/claims.jsonl
Output: data/d2_pilot/perception_results.jsonl
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

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL = BASE / "data/d2_pilot/claims.jsonl"
OUT_JSONL = BASE / "data/d2_pilot/perception_results.jsonl"


VERIFIER_SAMPLING = {
    "temperature": 0.0,
    "max_tokens": 128,
}


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def parse_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def verify_match(claim: dict, response: str) -> float:
    if claim["type"] == "value":
        extracted = parse_number(response)
        if extracted is None:
            return 0.5
        truth = claim.get("value")
        if truth is None:
            return 0.5
        rel_diff = abs(extracted - truth) / max(abs(truth), 1e-10)
        abs_diff = abs(extracted - truth)
        if rel_diff <= 0.05 or abs_diff <= 0.5:
            return 1.0
        if rel_diff <= 0.15:
            return 0.5
        return 0.0
    # categorical
    target = normalize_text(claim.get("entity") or "")
    norm = normalize_text(response)
    if not target:
        return 0.5
    return 1.0 if target in norm or norm in target else 0.0


def build_query(claim: dict) -> str:
    if claim["type"] == "value":
        return (
            f"Looking at this chart, what is the value associated with "
            f"'{claim['entity']}'? Reply with ONLY the number, no units, no explanation."
        )
    # categorical
    comp = claim.get("comparator", "highest")
    return (
        f"Looking at this chart, what entity has the {comp} value? "
        f"Reply with the entity name only."
    )


async def verify_claim(
    client: AsyncOpenAI,
    model: str,
    claim: dict,
    img_b64: str,
    sem: asyncio.Semaphore,
) -> tuple[float, str]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": build_query(claim)},
                ]}],
                temperature=VERIFIER_SAMPLING["temperature"],
                max_tokens=VERIFIER_SAMPLING["max_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return 0.5, f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"
    return verify_match(claim, text), text


async def process_sample(
    rec: dict, client_picker, model: str, sem: asyncio.Semaphore,
    file_lock: asyncio.Lock, out_path: Path,
) -> dict:
    img_b64 = image_to_b64(rec["image_path"])

    step_perception = []
    for si, step_claims in enumerate(rec["step_claims"]):
        if not step_claims:
            step_perception.append(None)
            continue
        scores = []
        responses = []
        # Parallel within a step
        futures = [verify_claim(client_picker(), model, c, img_b64, sem) for c in step_claims]
        results = await asyncio.gather(*futures)
        for (sc, resp) in results:
            scores.append(sc)
            responses.append(resp)
        step_perception.append({
            "claim_scores": scores,
            "claim_responses": responses,
            "mean_score": sum(scores) / len(scores),
        })

    out = dict(rec)
    out["step_perception_scores"] = step_perception
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


async def run(ports: list[int], model: str, concurrent_per_host: int = 5):
    records = [json.loads(l) for l in open(IN_JSONL) if l.strip()]
    done = load_done(OUT_JSONL)
    todo = [r for r in records if r["id"] not in done]
    print(f"[perception] total={len(records)} done={len(done)} todo={len(todo)}")
    if not todo:
        return 0

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0) for p in ports]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    file_lock = asyncio.Lock()
    cnt = {"i": 0}
    def pick():
        c = clients[cnt["i"] % len(clients)]
        cnt["i"] += 1
        return c

    print(f"[perception] sampling={VERIFIER_SAMPLING} hosts={len(clients)} concurrent/host={concurrent_per_host}")

    tasks = [process_sample(r, pick, model, sem, file_lock, OUT_JSONL) for r in todo]
    n_done = 0
    for f in asyncio.as_completed(tasks):
        await f
        n_done += 1
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)}", flush=True)

    # Stats
    all_recs = [json.loads(l) for l in open(OUT_JSONL)]
    pscores = [s["mean_score"] for r in all_recs
               for s in r["step_perception_scores"] if s is not None]
    if pscores:
        print(f"\n[perception] score dist: median={np.median(pscores):.3f}  "
              f"p25={np.percentile(pscores,25):.3f}  p75={np.percentile(pscores,75):.3f}")
    print(f"[perception] -> {OUT_JSONL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="comma-separated 9B verifier ports")
    ap.add_argument("--model", default="Qwen3.5-VL-9B")
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
