"""Score standard-protocol eval outputs.

Reads *.jsonl produced by eval_standard.py and writes *_scored.jsonl with
`pred`, `score`, and `judge_raw` (where applicable).

Scorers:
  - relaxed_correctness  : LMMs-Eval ChartQA scorer (deterministic, no API)
  - charxiv_judge        : LLM judge with REASONING_GRADING prompts (verbatim)
  - chartmuseum_judge    : LLM judge with COMPARE_ANSWER_PROMPT (verbatim)

Judge backend (env CHARTVR_JUDGE_BACKEND):
  - "qwen" (default): local Qwen3.6-27B vLLM hosts, 3-way round-robin
    Hosts: env CHARTVR_QWEN_HOSTS (default http://localhost:9101/v1,9102,9103)
    enable_thinking=False forced via chat_template_kwargs.
  - "claude": legacy `claude -p` Sonnet OAuth (kept for fallback).

Usage:
  # deterministic only (ChartQA-H/A/Pro)
  python score_standard.py results/standard/<run>/ --benches chartqa_human,chartqa_augmented,chartqa_pro

  # with judges (CharXiv, ChartMuseum) via Qwen3.6-27B
  python score_standard.py results/standard/<run>/ --benches charxiv_reasoning,chartmuseum --use_judge --concurrency 18
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import sys
from typing import Any

# ─────────────── Deterministic: LMMs-Eval relaxed_correctness ───────────────

def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> float:
    """Verbatim from third_party/lmms-eval/lmms_eval/tasks/chartqa/utils.py."""
    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None
    p, t = prediction.strip(), target.strip()
    pf, tf = _to_float(p), _to_float(t)
    if pf is not None and tf:
        return 1.0 if abs(pf - tf) / abs(tf) <= max_relative_change else 0.0
    return 1.0 if p.lower() == t.lower() else 0.0


# ─────────────── Output extraction (uniform across benches) ───────────────

def extract_post_think(record: dict) -> str:
    """Pull the post-thinking answer from a record produced by eval_standard.py.

    With vLLM --reasoning-parser qwen3:
      - thinking on:  <think>...</think> → reasoning_content; post-think → content
      - thinking off: empty <think>\\n\\n</think>\\n\\n injected; full output → content
    Either way, `content` carries the answer text.

    Strategy: take the LAST non-empty line of `content`. For base models that
    already follow "answer with a single word" instruction this is identical to
    the full content, but for SFT/RL models that emit a verbose explanation
    followed by a final-answer line (teacher-distillation prose pattern), this
    isolates the answer the model actually committed to.

    Fallback: if content is empty, try last non-empty line of reasoning_content
    (handles cases where reasoning-parser failed and everything went to
    reasoning_content).
    """
    c = (record.get("content") or "")
    lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    rc = (record.get("reasoning_content") or "")
    if "</think>" in rc:
        tail = rc.rsplit("</think>", 1)[1]
        tlines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        if tlines:
            return tlines[-1]
    rlines = [ln.strip() for ln in rc.splitlines() if ln.strip()]
    return rlines[-1] if rlines else ""


def chartmuseum_extract(record_or_text) -> str:
    """ChartMuseum-style extraction: <answer>(.*?)</answer> regex, lenient close.

    Runs the regex on the FULL `content` (multi-line aware via re.DOTALL),
    not just the last line — earlier behavior of `extract_post_think` returned
    `</answer>` alone when the model emitted `<answer>\\n...\\n</answer>`,
    which slipped past the regex and scored 0. We pick the LAST `<answer>` tag
    so trailing prose / multiple tags don't shadow the final answer.
    """
    if isinstance(record_or_text, str):
        text = record_or_text
    else:
        text = (record_or_text.get("content") or "")
        if not text.strip():
            # thinking-off / parser fallback: try reasoning_content tail after </think>
            rc = record_or_text.get("reasoning_content") or ""
            if "</think>" in rc:
                text = rc.rsplit("</think>", 1)[1]
            else:
                text = rc
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text + "</answer>", re.DOTALL))
    if matches:
        pred = matches[-1].group(1).strip()
        if pred:
            return pred
    # No tag (or empty body) → fall back to last non-empty line of full content.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ─────────────── CharXiv judge prompts (verbatim) ───────────────

CHARXIV_PREFIX = """
You will be given a question, an ground truth answer and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0.

Your response must follow json formats with keys [extract_answer, score] where the value of the score is an interger in [0, 1]. You must follow the scoring rules:\n"""

CHARXIV_INST = {
    1: """
    ### Rules ###
    * Give a score of 1 if and only if the final answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha). It's also acceptable to have different orders of the terms when question asks for multiple terms.
    * Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different between the final answer and the ground truth.

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>
    """,
    2: """
    ### Rules ###
    * If there are predefined options in the question:
        * Give a score of 1 if the final answer matches the ground truth answer exactly.
        * Give a score of 0 otherwise.
    * If there are no predefined options:
        * Give a score of 1 if the final answer shares the same semantic meaning with the ground truth (e.g., "increasing then decreasing" and "moving up then down").
        * Give a score of 0 if it has different meaning.

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>
    """,
    3: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. Different notations are acceptable (e.g., 0.01 and 10^-2).
    * Give a score of 0 if values differ.

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>
    """,
    4: """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values.
    * Give a score of 0 otherwise.

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>
    """,
}


# ─────────────── ChartMuseum judge prompt (verbatim) ───────────────

CHARTMUSEUM_COMPARE_PROMPT = """You are provided with a question and two answers. Please determine if these answers are equivalent. Follow these guidelines:

1. Numerical Comparison:
   - For decimal numbers, consider them as equivalent if their relative difference is sufficiently small.
   For example: 32.35 and 32.34 are equivalent; 32.35 and 35.25 are NOT.
   Note that if the question asks for years or dates, do exact match.

2. Unit Handling:
   - If only one answer includes units, ignore units and compare numerical values.

3. Text Comparison:
   - Ignore differences in capitalization.
   - Treat equivalent mathematical expressions as the same.

Question: [QUESTION]
Answer 1: [ANSWER1]
Answer 2: [ANSWER2]

Please respond with:
- "Yes" if the answers are equivalent
- "No" if the answers are different"""


# ─────────────── Judge backends ───────────────

JUDGE_BACKEND = os.environ.get("CHARTVR_JUDGE_BACKEND", "qwen")  # qwen|claude
JUDGE_TIMEOUT = int(os.environ.get("CHARTVR_JUDGE_TIMEOUT", "120"))

# --- Qwen3.6 vLLM backend (default) ---
QWEN_HOSTS = [h.strip() for h in os.environ.get(
    "CHARTVR_QWEN_HOSTS",
    "http://localhost:9101/v1,http://localhost:9102/v1,http://localhost:9103/v1",
).split(",") if h.strip()]
QWEN_MODEL = os.environ.get("CHARTVR_QWEN_MODEL", "Qwen3.6-27B")
QWEN_PER_HOST = int(os.environ.get("CHARTVR_QWEN_PER_HOST", "6"))

_qwen_clients: dict[str, Any] = {}
_qwen_sems: dict[str, asyncio.Semaphore] = {}
_qwen_rr = itertools.count()


def _qwen_init() -> None:
    """Lazy init AsyncOpenAI clients + per-host semaphores."""
    if _qwen_clients:
        return
    from openai import AsyncOpenAI  # type: ignore
    for h in QWEN_HOSTS:
        _qwen_clients[h] = AsyncOpenAI(base_url=h, api_key="EMPTY", timeout=JUDGE_TIMEOUT)
        _qwen_sems[h] = asyncio.Semaphore(QWEN_PER_HOST)


async def _qwen_p(prompt: str, max_tokens: int = 256) -> str:
    """Call Qwen3.6-27B vLLM with enable_thinking=False (forced).
    Round-robin across hosts; per-host semaphore caps concurrency.
    """
    _qwen_init()
    host = QWEN_HOSTS[next(_qwen_rr) % len(QWEN_HOSTS)]
    client = _qwen_clients[host]
    async with _qwen_sems[host]:
        try:
            resp = await client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {str(e)[:200]}"
    txt = (resp.choices[0].message.content or "").strip()
    return txt


# --- Claude `claude -p` legacy backend ---
CLAUDE_MODEL = os.environ.get("CHARTVR_JUDGE_MODEL", "claude-sonnet-4-6")


async def _claude_p(prompt: str) -> str:
    """Call `claude -p --model <CLAUDE_MODEL>` non-interactively.
    cwd=/tmp so project CLAUDE.md does not pollute judge output.
    """
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", "--disable-slash-commands", "--model", CLAUDE_MODEL,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/tmp",
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")), timeout=JUDGE_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        return f"ERROR: timeout after {JUDGE_TIMEOUT}s"
    if proc.returncode != 0:
        return f"ERROR: rc={proc.returncode}: {stderr.decode('utf-8', errors='replace')[:200]}"
    return stdout.decode("utf-8", errors="replace").strip()


async def _judge_call(prompt: str, max_tokens: int = 256) -> str:
    if JUDGE_BACKEND == "qwen":
        return await _qwen_p(prompt, max_tokens=max_tokens)
    if JUDGE_BACKEND == "claude":
        return await _claude_p(prompt)
    return f"ERROR: unknown JUDGE_BACKEND={JUDGE_BACKEND}"


def _parse_json_loose(text: str) -> dict | None:
    """Extract first {...} JSON object from text (handles surrounding chatter)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


async def _judge_charxiv(question: str, gold: str, response: str, ic: int) -> tuple[str, int, str]:
    inst = CHARXIV_INST.get(ic, CHARXIV_INST[1])
    prompt = CHARXIV_PREFIX + inst.replace("<|question|>", question)\
        .replace("<|ground_truth|>", gold).replace("<|response|>", response)
    prompt += '\n\nRespond with ONLY a JSON object: {"extracted_answer": "...", "score": 0 or 1}. No other text.'
    raw = await _judge_call(prompt, max_tokens=256)
    if raw.startswith("ERROR"):
        return "", 0, raw
    d = _parse_json_loose(raw)
    if not d:
        return "", 0, raw
    try:
        return str(d.get("extracted_answer", "")), int(d.get("score", 0)), raw
    except (ValueError, TypeError):
        return "", 0, raw


async def _judge_chartmuseum(question: str, gold: str, pred: str) -> tuple[int, str]:
    prompt = CHARTMUSEUM_COMPARE_PROMPT.replace("[QUESTION]", question)\
        .replace("[ANSWER1]", gold).replace("[ANSWER2]", pred)
    prompt += '\n\nRespond with ONLY "Yes" or "No". No other text.'
    raw = await _judge_call(prompt, max_tokens=8)
    if raw.startswith("ERROR"):
        return 0, raw
    return (1 if "yes" in raw.lower().strip().split() or raw.lower().strip().startswith("yes") else 0), raw


# ─────────────── Main ───────────────

def _load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def score_file(in_path: str, out_path: str, use_judge: bool, concurrency: int = 8) -> dict:
    rows = _load_jsonl(in_path)
    bench = rows[0]["bench"] if rows else "?"
    scoring = rows[0].get("scoring", "?")
    print(f"\n=== {bench} ({scoring}) ===  n={len(rows)}")

    # Resume: skip already-scored sample_ids; rows whose judge_raw starts with
    # "ERROR" are treated as unfinished (caller may need to clean the file
    # first if the prior backend left bulk failures — we just don't re-add them).
    done = set()
    if os.path.exists(out_path):
        for r in _load_jsonl(out_path):
            jr = (r.get("judge_raw") or "")
            if isinstance(jr, str) and jr.startswith("ERROR"):
                continue
            done.add(r["sample_id"])
    todo = [r for r in rows if r["sample_id"] not in done]
    if not todo:
        print(f"  already scored: {len(rows)}/{len(rows)}")
        scored = _load_jsonl(out_path)
        return {"bench": bench, "n": len(scored), "score": sum(r.get("score", 0) for r in scored) / max(1, len(scored))}

    if scoring == "relaxed_correctness":
        with open(out_path, "a") as fout:
            for r in todo:
                pred = extract_post_think(r)
                score = relaxed_correctness(pred, r["gold_answer"])
                fout.write(json.dumps({**r, "pred": pred, "score": score}, ensure_ascii=False) + "\n")
        scored = _load_jsonl(out_path)
        acc = sum(r.get("score", 0) for r in scored) / max(1, len(scored))
        print(f"  acc={acc:.4f}")
        return {"bench": bench, "n": len(scored), "score": acc}

    if not use_judge:
        print(f"  scoring={scoring} requires --use_judge; skipping")
        return {"bench": bench, "n": 0, "score": None}

    sem = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    n_done = 0

    async def one(r: dict) -> None:
        nonlocal n_done
        async with sem:
            if scoring == "charxiv_judge":
                resp = extract_post_think(r)
                ext, score, raw = await _judge_charxiv(
                    r["question"], r["gold_answer"], resp, int(r.get("inst_category", 1))
                )
                row = {**r, "pred": ext, "score": score, "judge_raw": raw}
            elif scoring == "chartmuseum_judge":
                pred = chartmuseum_extract(r)
                score, raw = await _judge_chartmuseum(r["question"], r["gold_answer"], pred)
                row = {**r, "pred": pred, "score": score, "judge_raw": raw}
            else:
                row = {**r, "score": None, "judge_raw": f"unknown scoring {scoring}"}
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 25 == 0:
            print(f"  [{bench}] judged {n_done}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    scored = _load_jsonl(out_path)
    acc = sum(r.get("score", 0) for r in scored if r.get("score") is not None) / max(1, len(scored))
    print(f"  acc={acc:.4f}")
    return {"bench": bench, "n": len(scored), "score": acc}


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--benches", default="chartqa_human,chartqa_augmented,chartqa_pro,charxiv_reasoning,chartmuseum")
    p.add_argument("--use_judge", action="store_true", help="enable GPT-4o/4.1-mini judge")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    summary = []
    for b in args.benches.split(","):
        ip = os.path.join(args.run_dir, f"{b}.jsonl")
        op = os.path.join(args.run_dir, f"{b}_scored.jsonl")
        if not os.path.exists(ip):
            print(f"missing: {ip}"); continue
        s = await score_file(ip, op, args.use_judge, args.concurrency)
        summary.append(s)
    print("\n=== SUMMARY ===")
    for s in summary:
        sc = s["score"]
        print(f"  {s['bench']:<22} n={s['n']:<5} score={sc:.4f}" if sc is not None else f"  {s['bench']:<22} (skipped)")


if __name__ == "__main__":
    asyncio.run(main())
