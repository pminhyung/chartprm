"""Math-Shepherd step-MC sub-rollout on dead samples (NO perception gating).

For each dead sample (outcome std=0 across K=4 outer rollouts):
  1. Segment baseline trace into steps.
  2. Per step k:
     a. 4-axis perception verification (qwen3vl-8b-thinking) → step_perception.
     b. K' MC sub-rollouts from cumulative prefix [steps 0..k] using our policy (qwen3vl-4b-instruct).
        ── NO perception gating; runs on every step regardless of perception score.
     c. Score each sub-rollout outcome (relaxed for chartqa_pro, 397B judge for chartmuseum/charxiv).
     d. mc_value = mean(sub_outcomes), mc_var = std(sub_outcomes).

Saves per-step: step_text, axis_results (claim verdicts), step_perception,
  sub_rollouts (K' full completion texts), sub_outcomes (K' 0/1), mc_value, mc_var.

Output: data/d2_hardbench/reports/math_shepherd_dead.jsonl
"""
from __future__ import annotations
import argparse, asyncio, base64, json, os, re, statistics, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
sys.path.insert(0, str(BASE / ".worktrees/research-chartprm/scripts"))
from preflight_reward_v1 import (split_steps, STEP_4AXIS_PROMPT, parse_4axis,
                                  img_to_b64, relaxed_correctness,
                                  CHARXIV_J, CHARTMUSEUM_J)


async def vlm_verify_step(client, model, image_b64, step_text, sem):
    async with sem:
        prompt = STEP_4AXIS_PROMPT.format(step=step_text[:1500])
        try:
            messages = [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                {"type":"text","text":prompt}]}]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=1024,
                extra_body={"chat_template_kwargs":{"enable_thinking":True}})
            content = resp.choices[0].message.content or ""
            return parse_4axis(content)
        except Exception as e:
            return {"VALUE":None,"LABEL":None,"COUNT":None,"STRUCTURE":None,"_err":str(e)[:80]}


async def mc_sub_rollout(policy_client, model, image_b64, question, prefix, K, sem):
    """Generate K continuations from prefix. Returns list[str] of completion texts."""
    async with sem:
        try:
            messages = [
                {"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                    {"type":"text","text":question}
                ]},
                {"role":"assistant","content": prefix},
            ]
            resp = await policy_client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.7, top_p=0.95, max_tokens=1024, n=K,
                extra_body={"continue_final_message": True,
                            "add_generation_prompt": False,
                            "chat_template_kwargs":{"enable_thinking":False}})
            return [c.message.content or "" for c in resp.choices]
        except Exception as e:
            return [f"<gen_error:{str(e)[:60]}>" for _ in range(K)]


async def judge_remote(judge_client, model, prompt, sem):
    async with sem:
        try:
            resp = await judge_client.chat.completions.create(
                model=model, messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=64,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            t = (resp.choices[0].message.content or "").strip().lower()
            if "verdict" in t:
                return 1 if "correct" in t.split("verdict")[-1] else 0
            if t.startswith("yes"): return 1
            if t.startswith("no"): return 0
            return None
        except Exception:
            return None


def extract_pred(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if m: return m.group(1).strip()[:200]
    # fallback: last non-empty line
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return (lines[-1] if lines else text[:200])[:200]


async def score_subrollout(text, sample, judge_client, judge_model, judge_sem):
    """Returns 0/1 outcome. Empty pred → 0 (skip judge to avoid empty-prompt bias)."""
    pred = extract_pred(text)
    if not pred.strip():
        return 0, pred
    bench = sample['bench']
    if bench == "chartqa_pro":
        return relaxed_correctness(pred, sample['gold']), pred
    tmpl = CHARXIV_J if bench == "charxiv_reasoning" else CHARTMUSEUM_J
    prompt = tmpl.format(q=sample['question'], a=sample['gold'], r=pred)
    o = await judge_remote(judge_client, judge_model, prompt, judge_sem)
    return o, pred


async def process_step(k, step_txt, cum_prefix, sample, image_b64,
                       vlm_client, vlm_model, policy_client, policy_model,
                       judge_client, judge_model, K_prime,
                       vlm_sem, mc_sem, judge_sem):
    # 1. perception (4-axis)
    axis_results = await vlm_verify_step(vlm_client, vlm_model, image_b64, step_txt, vlm_sem)
    scored = [v for v in axis_results.values() if isinstance(v, int)]
    step_perception = (sum(scored)/len(scored)) if scored else None
    # 2. MC sub-rollout (NO gating)
    sub_completions = await mc_sub_rollout(policy_client, policy_model, image_b64,
                                            sample['question'], cum_prefix, K_prime, mc_sem)
    # 3. score each sub-rollout
    score_tasks = [score_subrollout(c, sample, judge_client, judge_model, judge_sem)
                   for c in sub_completions]
    score_results = await asyncio.gather(*score_tasks)
    sub_outcomes = [r[0] for r in score_results]
    sub_preds = [r[1] for r in score_results]
    valid_outs = [o for o in sub_outcomes if o is not None]
    mc_value = (sum(valid_outs)/len(valid_outs)) if valid_outs else None
    mc_var = statistics.pstdev(valid_outs) if len(valid_outs) >= 2 else 0.0
    return {
        "k": k, "step_text": step_txt[:500],
        "axis_results": axis_results, "step_perception": step_perception,
        "sub_rollouts": [s[:600] for s in sub_completions],
        "sub_preds": sub_preds,
        "sub_outcomes": sub_outcomes,
        "mc_value": mc_value, "mc_var": mc_var,
        "n_valid_outcomes": len(valid_outs),
    }


async def process_sample(s, vlm_client, vlm_model, policy_client, policy_model,
                          judge_client, judge_model, K_prime,
                          vlm_sem, mc_sem, judge_sem):
    image_b64 = img_to_b64(s["image_path"])
    if not image_b64:
        return {**{k:v for k,v in s.items() if k != "trace"}, "error": "no_image"}
    steps = split_steps(s["trace"], max_steps=8)
    if not steps:
        return {**{k:v for k,v in s.items() if k != "trace"}, "error": "no_steps"}
    # Process steps SEQUENTIALLY per sample (each step depends on prior prefix); cross-sample parallel.
    step_records = []
    cum_prefix = ""
    for k, step_txt in enumerate(steps):
        cum_prefix = (cum_prefix + "\n" + step_txt).strip() if cum_prefix else step_txt.strip()
        rec = await process_step(k, step_txt, cum_prefix, s, image_b64,
                                  vlm_client, vlm_model, policy_client, policy_model,
                                  judge_client, judge_model, K_prime,
                                  vlm_sem, mc_sem, judge_sem)
        step_records.append(rec)
    return {
        **{k:v for k,v in s.items() if k != "trace" and k != "rollouts"},
        "n_steps": len(steps), "K_prime": K_prime,
        "step_records": step_records,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_url", default="http://localhost:8500/v1")
    ap.add_argument("--vlm_model", default="qwen3vl_verifier")
    ap.add_argument("--policy_url", default="http://localhost:8501/v1")
    ap.add_argument("--policy_model", default="qwen3vl_policy")
    ap.add_argument("--judge_url", default="http://10.1.211.147:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--input", default=str(BASE/"data/d2_hardbench/reports/mc_input.jsonl"))
    ap.add_argument("--output", default=str(BASE/"data/d2_hardbench/reports/math_shepherd_dead.jsonl"))
    ap.add_argument("--K_prime", type=int, default=6)
    ap.add_argument("--vlm_concurrency", type=int, default=6)
    ap.add_argument("--mc_concurrency", type=int, default=4)
    ap.add_argument("--judge_concurrency", type=int, default=8)
    ap.add_argument("--sample_concurrency", type=int, default=3)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Loaded {len(samples)} dead samples; K'={args.K_prime}")

    done = set()
    if Path(args.output).exists():
        for line in open(args.output):
            try:
                d = json.loads(line)
                done.add(f"{d['id']}||{d['combo']}")
            except Exception: pass
    todo = [s for s in samples if f"{s['id']}||{s['combo']}" not in done]
    print(f"  done: {len(done)}, todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    vlm_client = AsyncOpenAI(base_url=args.vlm_url, api_key="dummy", timeout=600.0)
    policy_client = AsyncOpenAI(base_url=args.policy_url, api_key="dummy", timeout=600.0)
    judge_client = AsyncOpenAI(base_url=args.judge_url, api_key="dummy", timeout=600.0)
    vlm_sem = asyncio.Semaphore(args.vlm_concurrency)
    mc_sem = asyncio.Semaphore(args.mc_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    sample_sem = asyncio.Semaphore(args.sample_concurrency)

    file_lock = asyncio.Lock()
    cnt_d = [0]

    async def go(s):
        async with sample_sem:
            try:
                rec = await process_sample(s, vlm_client, args.vlm_model,
                                            policy_client, args.policy_model,
                                            judge_client, args.judge_model,
                                            args.K_prime,
                                            vlm_sem, mc_sem, judge_sem)
            except Exception as e:
                rec = {**{k:v for k,v in s.items() if k != "trace" and k != "rollouts"},
                       "error": f"process_failed: {str(e)[:120]}"}
        async with file_lock:
            with open(args.output, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cnt_d[0] += 1
        print(f"  [{cnt_d[0]}/{len(todo)}] done: {s['id']} ({s['combo']})")

    await asyncio.gather(*[go(s) for s in todo])
    print(f"\nDone: {len(todo)} samples -> {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
