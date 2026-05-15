"""D2 perception verifier v2 — NOT_FOUND fallback + cross-family verifier + relaxed tolerance.

Replaces same-family Qwen2.5-VL-9B verifier with InternVL2.5-26B (cross-family).
Adds NOT_FOUND escape (Spec B) and 10%/25% tolerance (Spec C).

Multi-host async, sample-level append, resume-by-id.

Input:  data/d2_pilot/claims_v2.jsonl
Output: data/d2_pilot/perception_v2.jsonl
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


VERIFY_VALUE_PROMPT = """Looking at this chart, locate the entity labeled exactly '{entity}' on the chart's axes, legend, or data labels.

If you find it, reply with ONLY its associated numeric value (no units, no explanation).

If '{entity}' does NOT appear as a chart label (it's a derived value, math result, pronoun, or not visible on the chart), reply EXACTLY with: NOT_FOUND

Do not infer or compute. Read directly from the chart.

Reply:"""

VERIFY_CATEGORICAL_PROMPT = """Looking at this chart, identify which entity matches the description '{entity}'.

If you find a matching entity among the chart labels, reply with the entity name only.

If no chart entity matches '{entity}' (it's vague, derived, or not visible), reply EXACTLY: NOT_FOUND

Reply:"""


VERIFIER_SAMPLING = {"temperature": 0.0, "max_tokens": 64}

# Spec C — relaxed tolerance (chart visual reading noise)
TOL_REL_FULL = 0.10   # rel diff ≤ 10% → 1.0
TOL_ABS_FULL = 1.0    # abs diff ≤ 1.0 → 1.0
TOL_REL_PARTIAL = 0.25  # rel diff ≤ 25% → 0.5


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


def is_not_found(text: str) -> bool:
    if not text:
        return False
    up = text.upper()
    return "NOT_FOUND" in up or "NOT FOUND" in up


def score_value(claim_value: float | None, response: str) -> tuple[float | None, float | None]:
    """Returns (score, extracted_value). score None = NOT_FOUND, else 0/0.5/1."""
    if is_not_found(response):
        return None, None
    extracted = parse_number(response)
    if extracted is None or claim_value is None:
        return None, extracted  # parse failure → treat as NOT_FOUND
    rel_diff = abs(extracted - claim_value) / max(abs(claim_value), 1e-10)
    abs_diff = abs(extracted - claim_value)
    if rel_diff <= TOL_REL_FULL or abs_diff <= TOL_ABS_FULL:
        return 1.0, extracted
    if rel_diff <= TOL_REL_PARTIAL:
        return 0.5, extracted
    return 0.0, extracted


def score_categorical(claim_entity: str, response: str) -> float | None:
    if is_not_found(response):
        return None
    target = normalize_text(claim_entity)
    norm = normalize_text(response)
    if not target or not norm:
        return None
    return 1.0 if target in norm or norm in target else 0.0


async def verify_claim(
    client: AsyncOpenAI, model: str, claim: dict, img_b64: str,
    sem: asyncio.Semaphore,
) -> dict:
    ctype = claim.get("type", "value")
    entity = claim.get("entity", "")
    if ctype == "value" and claim.get("value") is not None:
        prompt = VERIFY_VALUE_PROMPT.format(entity=entity)
    else:
        prompt = VERIFY_CATEGORICAL_PROMPT.format(entity=entity)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=VERIFIER_SAMPLING["temperature"],
                max_tokens=VERIFIER_SAMPLING["max_tokens"],
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return {"score": None, "response": f"__ERROR__: {type(e).__name__}: {str(e)[:160]}",
                    "extracted": None, "matched": False}
    if ctype == "value":
        sc, extracted = score_value(claim.get("value"), text)
        return {"score": sc, "response": text, "extracted": extracted,
                "matched": sc is not None}
    sc = score_categorical(entity, text)
    return {"score": sc, "response": text, "matched": sc is not None}


async def process_sample(
    rec: dict, client_picker, model: str, sem: asyncio.Semaphore,
    file_lock: asyncio.Lock, out_path: Path,
) -> int:
    try:
        img_b64 = image_to_b64(rec["image_path"])
    except Exception as e:
        out_rec = dict(rec)
        out_rec["step_perception_scores"] = [None] * len(rec.get("step_claims", []))
        out_rec["verify_v2_error"] = f"image_load: {type(e).__name__}: {str(e)[:160]}"
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        return 0

    step_perception = []
    n_verified = 0
    for step_claims in rec.get("step_claims", []):
        if not step_claims:
            step_perception.append(None)
            continue
        futures = [verify_claim(client_picker(), model, c, img_b64, sem) for c in step_claims]
        results = await asyncio.gather(*futures)
        valid_scores = [r["score"] for r in results if r["score"] is not None]
        nf_count = sum(1 for r in results if r["score"] is None)
        step_perception.append({
            "verifications": results,
            "valid_scores": valid_scores,
            "mean_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
            "n_total": len(results),
            "n_not_found": nf_count,
            "not_found_rate": nf_count / max(len(results), 1),
        })
        n_verified += len(valid_scores)

    out = dict(rec)
    out["step_perception_scores"] = step_perception
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return n_verified


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
    print(f"[verify_v2] total={len(records)} done={len(done)} todo={len(todo)}")
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

    print(f"[verify_v2] sampling={VERIFIER_SAMPLING}  hosts={len(clients)}  conc/host={concurrent_per_host}")
    print(f"[verify_v2] tolerance: full={TOL_REL_FULL*100:.0f}%/{TOL_ABS_FULL} abs, partial=≤{TOL_REL_PARTIAL*100:.0f}%")

    tasks = [process_sample(r, pick, model, sem, file_lock, output_path) for r in todo]
    n_done = 0
    n_verified = 0
    for f in asyncio.as_completed(tasks):
        nv = await f
        n_done += 1
        n_verified += nv
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)}  verified={n_verified}", flush=True)
    # Final stats
    all_recs = [json.loads(l) for l in open(output_path) if l.strip()]
    pscores = [s["mean_score"] for r in all_recs
               for s in r.get("step_perception_scores", []) if s is not None and s.get("mean_score") is not None]
    nf_rates = [s["not_found_rate"] for r in all_recs
                for s in r.get("step_perception_scores", []) if s is not None]
    print(f"\n[verify_v2] total_records={len(all_recs)}")
    if pscores:
        print(f"  perception mean_score: median={np.median(pscores):.3f}  mean={np.mean(pscores):.3f}")
        print(f"  score distribution: =0: {sum(1 for s in pscores if s==0)/len(pscores)*100:.1f}%  "
              f"=0.5: {sum(1 for s in pscores if s==0.5)/len(pscores)*100:.1f}%  "
              f"=1: {sum(1 for s in pscores if s==1)/len(pscores)*100:.1f}%  "
              f"middle: {sum(1 for s in pscores if 0<s<1)/len(pscores)*100:.1f}%")
    if nf_rates:
        print(f"  NOT_FOUND rate per step: mean={np.mean(nf_rates)*100:.1f}%")
    print(f"  -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(BASE / "data/d2_pilot/claims_v2.jsonl"))
    ap.add_argument("--output", default=str(BASE / "data/d2_pilot/perception_v2.jsonl"))
    ap.add_argument("--ports", required=True, help="comma-separated verifier vLLM ports")
    ap.add_argument("--model", default="internvl_verifier")
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(Path(args.input), Path(args.output), ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
