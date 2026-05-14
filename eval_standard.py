"""Standard-protocol eval runner.

Uses official prompts/extraction/scoring per benchmark, layered on top of
our multi-host vLLM + concurrent + append-save infrastructure.

Per-bench protocol (sources in `third_party/`):
  - chartqa_human / chartqa_augmented / chartqa_pro:
      * Prompt: terse "Answer the question with a single word." (LMMs-Eval)
      * Generation: greedy, max_tokens=64
      * Scoring: deterministic relaxed_correctness (5% numeric tol, exact text)
  - charxiv_reasoning:
      * Prompt: per-inst_category instruction (CharXiv REASONING_RESP_INST)
      * Generation: greedy, max_tokens=512
      * Scoring: GPT-4o judge (run separately via score_standard.py)
  - chartmuseum:
      * Prompt: official thinking-style <think>/<answer> (ChartMuseum prompt.py QA_PROMPT)
      * Generation: temperature=0.6, top_p=0.95, max_tokens=4096 (thinking mode)
      * Extraction: regex <answer>(.*?)</answer>
      * Scoring: GPT-4.1-mini judge (run separately via score_standard.py)

Inference outputs raw output text + sample_id + bench. Scoring is deferred to
score_standard.py so judges can be batched / re-run independently.
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
from typing import Any

from PIL import Image
from openai import AsyncOpenAI

BASE = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ─────────────── Standard prompts (verbatim from official repos) ───────────────

# LMMs-Eval ChartQA / ChartQAPro post_prompt
CHARTQA_POST = "\nAnswer the question with a single word."

# ChartMuseum prompt.py QA_PROMPT (verbatim)
CHARTMUSEUM_PROMPT = """Please answer the question using the chart image.

Question: [QUESTION]

Please first generate your reasoning process and then provide the user with the answer. Use the following format:

<think>
... your thinking process here ...
</think>
<answer>
... your final answer (entity(s) or number) ...
</answer>"""

# CharXiv reasoning REASONING_RESP_INST (verbatim from src/constants.py)
CHARXIV_REASONING_INST = {
    1: "{}\n    * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.\n    * If you need to answer multiple terms, separate them with commas.\n    * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.\n    ",
    2: "{}\n    * If there are options in the question, your final answer must conform to one of the options.\n    * If there are additional instructions in the question, follow them accordingly.\n    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only.\n    ",
    3: "{}\n    * Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.\n    * You are allowed to extract numbers within some text when needed.\n    ",
    4: "{}\n    {}\n    ",
}

def _charxiv_number_instruction(answer: str) -> str:
    """For inst_category=4: derive decimal-place instruction from gold answer."""
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if decimal is None:
        return "* Your final answer must be an exact integer."
    return f"* Your final answer must be a number with {len(decimal)} decimal places."


# ─────────────── Benchmark configs ───────────────

def _load_chartqa_split(split: str, thinking: bool = False) -> list[dict]:
    """LMMs-Eval style chartqa. thinking flag selects on/off variant."""
    path = f"{BASE}/data/chartqa/test/test_{split}.json"
    with open(path) as f:
        data = json.load(f)
    samples = []
    for i, item in enumerate(data):
        img_path = f"{BASE}/data/chartqa/test/png/{item['imgname']}"
        if not os.path.exists(img_path):
            continue
        samples.append({
            "sample_id": f"chartqa_{split}_{i}",
            "bench": f"chartqa_{split}",
            "image_path": img_path,
            "question": item["query"],
            "gold_answer": str(item["label"]),
            "prompt_text": item["query"] + CHARTQA_POST,
            "thinking": thinking,
            "max_tokens": 16384 if thinking else 64,
            "temperature": 0.6 if thinking else 0.0,
            **({"top_p": 0.95} if thinking else {}),
            "scoring": "relaxed_correctness",
        })
    return samples


def _load_chartqa_pro(thinking: bool = False) -> list[dict]:
    path = f"{BASE}/data/chartqa_pro/test.json"
    with open(path) as f:
        data = json.load(f)
    samples = []
    for i, item in enumerate(data):
        img_path = f"{BASE}/data/chartqa_pro/images/{item['imgname']}"
        if not os.path.exists(img_path):
            continue
        samples.append({
            "sample_id": f"chartqa_pro_{i}",
            "bench": "chartqa_pro",
            "image_path": img_path,
            "question": item["query"],
            "gold_answer": str(item["label"]),
            "prompt_text": item["query"] + CHARTQA_POST,
            "thinking": thinking,
            "max_tokens": 16384 if thinking else 64,
            "temperature": 0.6 if thinking else 0.0,
            **({"top_p": 0.95} if thinking else {}),
            "scoring": "relaxed_correctness",
            "question_type": item.get("question_type"),
        })
    return samples


def _load_charxiv_reasoning(split: str = "val", thinking: bool = False) -> list[dict]:
    """CharXiv official reasoning_val.json — has inst_category for per-cat prompt."""
    path = f"{BASE}/third_party/CharXiv/data/reasoning_{split}.json"
    with open(path) as f:
        data = json.load(f)
    # Image source priority: official jpg from images.zip (full coverage) > local png mirror
    samples = []
    for fid, d in data.items():
        img_jpg_full = f"{BASE}/data/charxiv/images_full/{fid}.jpg"
        img_png = f"{BASE}/data/charxiv/images/{fid}.png"
        img_jpg = f"{BASE}/data/charxiv/images/{fid}.jpg"
        img_path = next((p for p in (img_jpg_full, img_png, img_jpg) if os.path.exists(p)), None)
        if not img_path:
            continue
        ic = d["inst_category"]
        if ic in (1, 2, 3):
            ptext = CHARXIV_REASONING_INST[ic].format(d["query"])
        elif ic == 4:
            ptext = CHARXIV_REASONING_INST[4].format(
                d["query"], _charxiv_number_instruction(str(d["answer"]))
            )
        else:
            continue
        samples.append({
            "sample_id": f"charxiv_{fid}",
            "bench": "charxiv_reasoning",
            "figure_id": fid,
            "image_path": img_path,
            "question": d["query"],
            "gold_answer": str(d["answer"]),
            "inst_category": ic,
            "prompt_text": ptext,
            "thinking": thinking,
            "max_tokens": 16384 if thinking else 512,
            "temperature": 0.6 if thinking else 0.0,
            **({"top_p": 0.95} if thinking else {}),
            "scoring": "charxiv_judge",
        })
    return samples


def _load_chartmuseum(split: str = "test", thinking: bool = True) -> list[dict]:
    path = f"{BASE}/data/chartmuseum/test.json"
    with open(path) as f:
        data = json.load(f)
    samples = []
    for i, item in enumerate(data):
        img_path = f"{BASE}/data/chartmuseum/hf_data/images/{item['imgname']}"
        if not os.path.exists(img_path):
            continue
        ptext = CHARTMUSEUM_PROMPT.replace("[QUESTION]", item["query"])
        samples.append({
            "sample_id": f"chartmuseum_{i}",
            "bench": "chartmuseum",
            "image_path": img_path,
            "question": item["query"],
            "gold_answer": str(item["label"]),
            "prompt_text": ptext,
            "thinking": thinking,
            "max_tokens": 16384 if thinking else 256,
            "temperature": 0.6 if thinking else 0.0,
            **({"top_p": 0.95} if thinking else {}),
            "scoring": "chartmuseum_judge",
            "reasoning_type": item.get("reasoning_type"),
        })
    return samples


BENCH_LOADERS = {
    "chartqa_human":     lambda thinking=False: _load_chartqa_split("human", thinking),
    "chartqa_augmented": lambda thinking=False: _load_chartqa_split("augmented", thinking),
    "chartqa_pro":       lambda thinking=False: _load_chartqa_pro(thinking),
    "charxiv_reasoning": lambda thinking=False: _load_charxiv_reasoning("val", thinking),
    "chartmuseum":       lambda thinking=True:  _load_chartmuseum("test", thinking),
}


# ─────────────── Inference (multi-host vLLM, concurrent, append) ───────────────

def _img_to_b64(image_path: str, max_side: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    s = max(img.size)
    if s > max_side:
        scale = max_side / s
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_done_ids(out_path: str) -> set[str]:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


async def _infer_sample(client: AsyncOpenAI, model_id: str, sample: dict) -> dict:
    img_b64 = _img_to_b64(sample["image_path"])
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": sample["prompt_text"]},
        ],
    }]
    extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": sample["thinking"]}}
    kwargs = {
        "model": model_id,
        "messages": messages,
        "max_tokens": sample["max_tokens"],
        "temperature": sample["temperature"],
        "extra_body": extra,
    }
    if "top_p" in sample:
        kwargs["top_p"] = sample["top_p"]
    try:
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
        reasoning = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
        err = ""
    except Exception as e:
        content, reasoning, err = "", "", f"ERROR: {e}"
    return {"content": content, "reasoning_content": reasoning, "error": err}


async def run_eval(bench: str, server_urls: list[str], model_id: str, out_dir: str,
                   per_server_concurrency: int = 5, thinking: bool | None = None) -> None:
    loader = BENCH_LOADERS[bench]
    samples = loader(thinking) if thinking is not None else loader()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{bench}.jsonl")

    done = _load_done_ids(out_path)
    todo = [s for s in samples if s["sample_id"] not in done]
    print(f"\n=== {bench} ===  total={len(samples)} done={len(done)} todo={len(todo)} servers={len(server_urls)}")
    if not todo:
        return

    clients = [AsyncOpenAI(base_url=u, api_key="dummy") for u in server_urls]
    sem = asyncio.Semaphore(len(server_urls) * per_server_concurrency)
    file_lock = asyncio.Lock()
    n_done = 0

    async def task(idx: int, s: dict) -> None:
        nonlocal n_done
        async with sem:
            client = clients[idx % len(clients)]
            inf = await _infer_sample(client, model_id, s)
        rec = {
            "sample_id": s["sample_id"],
            "bench": s["bench"],
            "thinking": s["thinking"],
            "question": s["question"],
            "gold_answer": s["gold_answer"],
            "image_path": s["image_path"],
            "content": inf["content"],
            "reasoning_content": inf["reasoning_content"],
            "error": inf["error"],
            "scoring": s["scoring"],
        }
        # carry per-bench metadata for downstream scoring
        for k in ("inst_category", "figure_id", "question_type", "reasoning_type"):
            if k in s:
                rec[k] = s[k]
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{bench}] {n_done}/{len(todo)}", flush=True)

    await asyncio.gather(*(task(i, s) for i, s in enumerate(todo)))
    print(f"  [{bench}] complete -> {out_path}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ports", help="Comma-separated vLLM ports (e.g., 8000,8001,...)")
    p.add_argument("model_id", help="Model path/ID served by vLLM")
    p.add_argument("output_dir", help="Output dir for *.jsonl")
    p.add_argument("benchmarks", help="Comma-separated bench names")
    p.add_argument("--concurrency", type=int, default=5, help="Concurrent reqs per server")
    p.add_argument("--host", default="localhost")
    p.add_argument("--thinking", choices=["on", "off", "default"], default="default",
                   help="Force thinking on/off across all benches; 'default' uses each bench's default")
    args = p.parse_args()

    urls = [f"http://{args.host}:{port}/v1" for port in args.ports.split(",")]
    benches = args.benchmarks.split(",")
    for b in benches:
        if b not in BENCH_LOADERS:
            sys.exit(f"unknown benchmark: {b}; valid={list(BENCH_LOADERS)}")
    flag: bool | None
    if args.thinking == "on":
        flag = True
    elif args.thinking == "off":
        flag = False
    else:
        flag = None
    for b in benches:
        await run_eval(b, urls, args.model_id, args.output_dir, args.concurrency, flag)


if __name__ == "__main__":
    asyncio.run(main())
