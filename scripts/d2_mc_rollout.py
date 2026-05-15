"""D2 MC outcome rollout — Math-Shepherd style step-value estimation.

For each (sample, step_idx), generate K=8 continuations starting from the end of
that step and check how many reach the correct final answer. step_value = mean
correctness ∈ [0,1].

Pattern: prefix prompt via tokenizer.apply_chat_template + raw `/v1/completions`
endpoint (assistant continuation). This is more robust than chat endpoint when
we want to inject `<think>\\n{prefix_steps}\\n\\n` as the assistant prefix.

Input:  data/d2_pilot/samples.jsonl
Output: data/d2_pilot/mc_results.jsonl  (per-sample, append + resume)

Sampling: guide §3.3 spec (temperature=0.7, top_p=0.9, max_tokens=2048, K=8).
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

from openai import AsyncOpenAI
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chartvr.extraction import extract_answer, relaxed_accuracy  # noqa: E402

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL = BASE / "data/d2_pilot/samples.jsonl"
OUT_JSONL = BASE / "data/d2_pilot/mc_results.jsonl"
TOKENIZER_PATH = BASE / "models/qwen3.5-4b"

SYSTEM_PROMPT = (
    "You are a chart reasoning assistant. Look at the chart carefully and "
    "think step by step before giving your final answer."
)

# Guide §3.3 continuation sampling
MC_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "n": 8,  # K continuations per step
}


def image_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_prefix_prompt(tokenizer, sample: dict, step_idx: int, img_b64: str) -> str:
    """Build a chat-templated prompt that ends with <think>\\n{prefix_steps}\\n\\n
    so vLLM continues from the partial thinking."""
    prefix_steps = "\n\n".join(sample["steps"][: step_idx + 1])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Question: {sample['question']}"},
        ]},
    ]
    # add_generation_prompt=True adds the assistant header; we then inject
    # <think>\n{prefix}\n\n so generation continues from there.
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )
    return base + f"<think>\n{prefix_steps}\n\n"


def extract_final_answer(continuation: str) -> str:
    """Pull post-think answer out of a continuation."""
    m = re.search(r"</think>\s*(.+)", continuation, re.DOTALL)
    if m:
        tail = m.group(1).strip()
        return extract_answer(tail) or tail
    # No </think> closure → fallback to last numeric or extract_answer over full
    return extract_answer(continuation) or ""


async def gen_step_continuations(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    """K=8 continuations for one (sample, step). Uses /v1/completions endpoint."""
    async with sem:
        try:
            resp = await client.completions.create(
                model=model,
                prompt=prompt,
                temperature=MC_SAMPLING["temperature"],
                top_p=MC_SAMPLING["top_p"],
                max_tokens=MC_SAMPLING["max_tokens"],
                n=MC_SAMPLING["n"],
                extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.1},
            )
            return [c.text for c in resp.choices]
        except Exception as e:
            return [f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"] * MC_SAMPLING["n"]


async def process_sample(
    sample: dict,
    client_picker,
    model: str,
    tokenizer,
    sem: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    out_path: Path,
) -> dict:
    img_b64 = image_to_b64(sample["image_path"])
    step_outcome_scores = []
    for step_idx in range(sample["n_steps"]):
        prompt = build_prefix_prompt(tokenizer, sample, step_idx, img_b64)
        client = client_picker()
        completions = await gen_step_continuations(client, model, prompt, sem)
        # Score each continuation
        scores = []
        finals = []
        for c in completions:
            if c.startswith("__ERROR__"):
                scores.append(0.0)
                finals.append("")
                continue
            final = extract_final_answer(c)
            finals.append(final)
            scores.append(float(relaxed_accuracy(final, sample["gold_answer"])))
        mean_score = sum(scores) / len(scores)
        # Variance
        m = mean_score
        var = sum((s - m) ** 2 for s in scores) / len(scores)
        step_outcome_scores.append({
            "step_idx": step_idx,
            "individual_scores": scores,
            "final_answers": finals,
            "mean_score": mean_score,
            "variance": var,
        })

    result = {
        "id": sample["id"],
        "source": sample["source"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "n_steps": sample["n_steps"],
        "steps": sample["steps"],
        "step_outcome_scores": step_outcome_scores,
    }
    async with file_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


async def run(ports: list[int], model: str, concurrent_per_host: int = 5) -> int:
    samples = [json.loads(l) for l in open(IN_JSONL) if l.strip()]
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [s for s in samples if s["id"] not in done]
    print(f"[d2-mc] total={len(samples)} done={len(done)} todo={len(todo)}")
    if not todo:
        return 0

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0) for p in ports]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    file_lock = asyncio.Lock()

    # Round-robin client picker
    counter = {"i": 0}
    def pick():
        c = clients[counter["i"] % len(clients)]
        counter["i"] += 1
        return c

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[d2-mc] sampling={MC_SAMPLING} | hosts={len(clients)} | concurrent/host={concurrent_per_host}")

    tasks = [process_sample(s, pick, model, tokenizer, sem, file_lock, OUT_JSONL) for s in todo]
    n_done = 0
    n_err = 0
    for f in asyncio.as_completed(tasks):
        r = await f
        n_done += 1
        n_step_err = sum(
            1 for sos in r["step_outcome_scores"]
            for fa in sos["final_answers"] if not fa
        )
        if n_step_err:
            n_err += 1
        if n_done % 5 == 0 or n_done == len(todo):
            print(f"  progress: {n_done}/{len(todo)} (samples with some empty finals: {n_err})", flush=True)
    print(f"[d2-mc] complete -> {OUT_JSONL}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="comma-separated 4B policy ports")
    ap.add_argument("--model", default="Qwen3.5-VL-4B")
    ap.add_argument("--concurrent_per_host", type=int, default=5)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    return asyncio.run(run(ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
