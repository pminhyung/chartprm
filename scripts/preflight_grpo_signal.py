"""GRPO learning-signal diagnostic.

For each of N prompts, generate K fresh rollouts using policy model (qwen3vl-4b-instruct).
For each rollout, compute:
  - perception_score: mean of 4-axis VLM verdict over all steps in the rollout
  - outcome_score: 397B judge on final answer (0/1)
  - combined_reward: perception × outcome (multiplicative)

Aggregate per-prompt:
  - std of outcome over K rollouts
  - std of perception over K rollouts
  - std of combined over K rollouts
  - frac_zero_std: fraction of prompts where all K rollouts have identical reward

Goal: combined reward should have LOWER frac_zero_std than outcome-only → GRPO learning signal present.
"""
from __future__ import annotations

import argparse, asyncio, base64, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

from preflight_reward_v1 import (
    split_steps, vlm_verify_step,
    relaxed_correctness, CHARXIV_J, CHARTMUSEUM_J, judge_remote, img_to_b64
)

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


CHARTQA_POST = "Answer with a short phrase or value. Format: <answer>X</answer>"
CHARXIV_POST = "Answer the question based on the chart. Format: <answer>X</answer>"
CHARTMUSEUM_POST = "Think step-by-step inside <think>...</think>, then provide final answer inside <answer>X</answer>."


def build_prompt(question: str, bench: str) -> str:
    if bench == "chartqa_pro":
        return f"{question}\n\n{CHARTQA_POST}"
    if bench == "charxiv_reasoning":
        return f"{question}\n\n{CHARXIV_POST}"
    return f"{question}\n\n{CHARTMUSEUM_POST}"


async def policy_rollout(client, model, image_b64, prompt_text, K, sem) -> list[str]:
    """Generate K rollouts (full completions) for the same prompt."""
    async with sem:
        try:
            messages = [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                {"type":"text","text":prompt_text}]}]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.8, top_p=0.95, max_tokens=2048, n=K)
            outs = []
            for c in resp.choices:
                msg = c.message
                content = msg.content or ""
                reasoning = (getattr(msg, "reasoning_content", "") or "")
                outs.append((reasoning + "\n" + content).strip())
            return outs
        except Exception as e:
            return [""] * K


def extract_final_answer(trace: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", trace, re.S | re.I)
    if m: return m.group(1).strip()
    return trace.strip().split("\n")[-1][:200]


async def evaluate_rollout(rollout_trace, image_b64, vlm_client, vlm_model,
                            judge_client, judge_model, question, gold, bench,
                            vlm_sem, judge_sem):
    """Compute perception, outcome, combined reward for one rollout."""
    if not rollout_trace.strip():
        return {"perception": None, "outcome": None, "combined": None,
                "pred": "", "n_steps": 0}
    steps = split_steps(rollout_trace, max_steps=6)
    if not steps:
        return {"perception": None, "outcome": None, "combined": None,
                "pred": extract_final_answer(rollout_trace), "n_steps": 0}
    # Per-step 4-axis verifier (parallel)
    axis_tasks = [vlm_verify_step(vlm_client, vlm_model, image_b64, st, vlm_sem)
                  for st in steps]
    axis_results_list = await asyncio.gather(*axis_tasks)
    step_percs = []
    for axr in axis_results_list:
        scored = [v for v in axr.values() if v is not None]
        if scored:
            step_percs.append(sum(scored)/len(scored))
    perception = (sum(step_percs)/len(step_percs)) if step_percs else None
    # Outcome
    pred = extract_final_answer(rollout_trace)
    if bench == "chartqa_pro":
        outcome = relaxed_correctness(pred, gold)
    else:
        tmpl = CHARXIV_J if bench == "charxiv_reasoning" else CHARTMUSEUM_J
        prompt = tmpl.format(q=question, a=gold, r=pred)
        outcome = await judge_remote(judge_client, judge_model, prompt, judge_sem)
    combined = None
    if perception is not None and outcome is not None:
        combined = perception * outcome
    return {"perception": perception, "outcome": outcome, "combined": combined,
            "pred": pred[:200], "n_steps": len(steps),
            "axis_per_step": axis_results_list}


async def process_prompt(s, K, vlm_client, vlm_model, policy_client, policy_model,
                          judge_client, judge_model, policy_sem, vlm_sem, judge_sem):
    image_b64 = img_to_b64(s["image_path"])
    if not image_b64:
        return {**{k:v for k,v in s.items() if k!="trace"}, "error":"no_image"}
    bench = s["bench"]
    prompt_text = build_prompt(s["question"], bench)
    rollouts = await policy_rollout(policy_client, policy_model, image_b64,
                                      prompt_text, K, policy_sem)
    # Evaluate each rollout
    eval_tasks = [evaluate_rollout(r, image_b64, vlm_client, vlm_model,
                                     judge_client, judge_model,
                                     s["question"], s["gold"], bench,
                                     vlm_sem, judge_sem) for r in rollouts]
    evals = await asyncio.gather(*eval_tasks)
    return {**{k:v for k,v in s.items() if k!="trace"},
            "K": K, "rollouts": [
                {"trace_head": r[:300], "trace_tail": r[-300:], **ev}
                for r, ev in zip(rollouts, evals)
            ]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_url", default="http://localhost:8500/v1")
    ap.add_argument("--vlm_model", default="qwen3vl_verifier")
    ap.add_argument("--policy_url", default="http://localhost:8501/v1")
    ap.add_argument("--policy_model", default="qwen3vl_policy")
    ap.add_argument("--judge_urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--input", default=str(BASE/"data/d2_hardbench/reports/preflight_50.jsonl"))
    ap.add_argument("--output", default=str(BASE/"data/d2_hardbench/reports/preflight_grpo_signal.jsonl"))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--vlm_concurrency", type=int, default=8)
    ap.add_argument("--policy_concurrency", type=int, default=4)
    ap.add_argument("--judge_concurrency", type=int, default=8)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"loaded {len(samples)} prompts; K={args.K}")

    done = set()
    if Path(args.output).exists():
        for line in open(args.output):
            try: done.add(json.loads(line)["id"]+"||"+json.loads(line)["combo"])
            except Exception: pass
    todo = [s for s in samples if (s["id"]+"||"+s["combo"]) not in done]
    print(f"done={len(done)}, todo={len(todo)}")
    if not todo: return

    vlm_client = AsyncOpenAI(base_url=args.vlm_url, api_key="dummy", timeout=300.0)
    policy_client = AsyncOpenAI(base_url=args.policy_url, api_key="dummy", timeout=300.0)
    judge_client = AsyncOpenAI(base_url=args.judge_urls.split(",")[0], api_key="dummy", timeout=300.0)
    policy_sem = asyncio.Semaphore(args.policy_concurrency)
    vlm_sem = asyncio.Semaphore(args.vlm_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    file_lock = asyncio.Lock()
    cnt_d = [0]
    async def go(s):
        rec = await process_prompt(s, args.K, vlm_client, args.vlm_model,
                                     policy_client, args.policy_model,
                                     judge_client, args.judge_model,
                                     policy_sem, vlm_sem, judge_sem)
        async with file_lock:
            with open(args.output, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        cnt_d[0] += 1
        if cnt_d[0] % 5 == 0:
            print(f"  processed {cnt_d[0]}/{len(todo)}")

    await asyncio.gather(*[go(s) for s in todo])
    print(f"done: {len(todo)} prompts -> {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
