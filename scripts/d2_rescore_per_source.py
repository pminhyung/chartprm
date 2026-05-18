"""D2 — per-source outcome rescore.

For each pilot sample, applies the BENCHMARK-STANDARD scorer:
- chartqa_train  → LMMs-Eval relaxed_correctness (verbatim, deterministic)
- reachqa_train  → ReachQA official LLM judge (verbatim Answer_Judge_Prompt)

Multi-host async, sample-level append, resume by id.

Reads <perception_file> + extracts model prediction from segmented_v3 (main)
or raw_response (baselines).

Output: <out_path> jsonl with id, outcome_old, outcome_new, source, scorer, judge_raw.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
sys.path.insert(0, str(BASE))
from chartvr.extraction import extract_answer, relaxed_accuracy  # noqa: E402


# ─── LMMs-Eval verbatim ────────────────────────────────────────────────────
def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> float:
    """Verbatim from third_party/lmms-eval/lmms_eval/tasks/chartqa/utils.py
    via score_standard.py."""
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


# ─── ReachQA official Answer_Judge_Prompt (verbatim) ───────────────────────
REACHQA_JUDGE_PROMPT = """Compare the ground truth with the prediction from AI model and determine if the prediction is correct. The question is about an image, which we have not given here. You need to determine whether the model's prediction is consistent with the ground truth. No points will be awarded for wrong answers, over answers or under answers. The reasoning process in the prediction does not need to be considered too much, you only need to determine if the final answer is consistent. There are times when the answer may have a different form of expression and some variation is acceptable.

## Question: {question}
## Ground Truth: {answer}
## Prediction: {prediction}

Now, let's take a analysis and then provide your judgement. Your response must follow the format below:
Analysis: (analyze the correctness briefly)
Correctness: (Yes or No)"""


def parse_judge_response(text: str) -> int:
    m = re.search(r"Correctness:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return -1
    ans = m.group(1).strip().lower()
    if "yes" in ans:
        return 1
    if "no" in ans:
        return 0
    return -1


async def judge_one(client: AsyncOpenAI, model: str, q: str, gold: str, pred: str,
                    sem: asyncio.Semaphore) -> tuple[int, str]:
    prompt = REACHQA_JUDGE_PROMPT.format(question=q, answer=gold, prediction=pred)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return -1, f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"
    return parse_judge_response(text), text


def get_prediction(rec: dict, src_seg: dict | None) -> str:
    """Extract model's prediction from a sample record.
    - If rec has raw_response (baseline_inference output), use that.
    - Else look up the matching segmented_v3 record (main d2_pilot 27B trace).
    """
    raw = rec.get("raw_response", "")
    if raw and not raw.startswith("__ERROR__") and not raw.startswith("__IMG_ERROR__"):
        ans = extract_answer(raw) or raw.strip()
        return ans
    if src_seg is not None:
        content = src_seg.get("content", "") or ""
        if content:
            return extract_answer(content) or content.strip()
        reasoning = src_seg.get("reasoning_content", "") or ""
        return extract_answer(reasoning) or reasoning.strip()
    return ""


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


async def run(perception_path: Path, output_path: Path, ports: list[int], model: str,
              concurrent_per_host: int):
    perc = [json.loads(l) for l in open(perception_path) if l.strip()]
    seg = {r["id"]: r for r in (json.loads(l) for l in open(BASE / "data/d1_pilot/segmented_v3.jsonl") if l.strip())}
    done = load_done(output_path)
    todo = [r for r in perc if r["id"] not in done]
    print(f"[rescore] perception={perception_path.name}  total={len(perc)} done={len(done)} todo={len(todo)}")
    if not todo:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate by source
    chartqa = [r for r in todo if r["id"].startswith("chartqa_")]
    reachqa = [r for r in todo if r["id"].startswith("reachqa_")]
    print(f"  chartqa={len(chartqa)}  reachqa={len(reachqa)}")

    file_lock = asyncio.Lock()

    # Deterministic chartqa
    for rec in chartqa:
        sid = rec["id"]
        src_seg = seg.get(sid)
        pred = get_prediction(rec, src_seg)
        gold = rec["gold_answer"]
        score_old = float(relaxed_accuracy(pred, gold))
        score_new = float(relaxed_correctness(pred, gold))
        out_rec = {"id": sid, "source": "chartqa_train", "scorer": "relaxed_correctness",
                   "prediction": pred[:300], "gold": gold,
                   "outcome_old": int(score_old), "outcome_new": int(score_new)}
        async with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    print(f"  chartqa rescored ({len(chartqa)} samples)")

    # LLM judge reachqa
    if reachqa:
        clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0)
                   for p in ports]
        sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
        cnt = {"i": 0}
        def pick():
            c = clients[cnt["i"] % len(clients)]
            cnt["i"] += 1
            return c

        async def judge_sample(rec):
            sid = rec["id"]
            src_seg = seg.get(sid)
            pred = get_prediction(rec, src_seg)
            gold = rec["gold_answer"]
            score_old = float(relaxed_accuracy(pred, gold))
            client = pick()
            judge, judge_raw = await judge_one(client, model, rec["question"], gold, pred, sem)
            out_rec = {"id": sid, "source": "reachqa_train", "scorer": "reachqa_llm_judge",
                       "prediction": pred[:300], "gold": gold[:500],
                       "outcome_old": int(score_old), "outcome_new": int(judge) if judge != -1 else None,
                       "judge_raw": judge_raw[:500]}
            async with file_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        tasks = [judge_sample(r) for r in reachqa]
        n_done = 0
        for f in asyncio.as_completed(tasks):
            await f
            n_done += 1
            if n_done % 5 == 0 or n_done == len(reachqa):
                print(f"  reachqa judge: {n_done}/{len(reachqa)}", flush=True)

    # Final stats
    all_recs = [json.loads(l) for l in open(output_path) if l.strip()]
    cq = [r for r in all_recs if r["source"] == "chartqa_train"]
    rq = [r for r in all_recs if r["source"] == "reachqa_train"]
    print(f"\n[rescore] outcome_new pass rate:")
    if cq:
        print(f"  chartqa  : old={sum(r['outcome_old'] for r in cq)/len(cq)*100:.1f}%  new={sum(r['outcome_new'] for r in cq)/len(cq)*100:.1f}%  (Δ={sum(r['outcome_new']-r['outcome_old'] for r in cq):+d})")
    if rq:
        valid = [r for r in rq if r["outcome_new"] is not None]
        if valid:
            print(f"  reachqa  : old={sum(r['outcome_old'] for r in rq)/len(rq)*100:.1f}%  new={sum(r['outcome_new'] for r in valid)/len(valid)*100:.1f}%  (Δ={sum((r['outcome_new'] or 0)-r['outcome_old'] for r in valid):+d})")
    print(f"  -> {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True, help="perception_v2.jsonl or baseline_*_perception.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ports", default="8400")
    ap.add_argument("--model", default="qwen3_6_27b_judge")
    ap.add_argument("--concurrent_per_host", type=int, default=8)
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(Path(args.perception), Path(args.output), ports, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
