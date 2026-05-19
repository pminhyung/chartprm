"""Tier 1-B OC-VDM measurement: 12-combo image_dep on wrong samples.

For each wrong sample (outcome=0):
  Use baseline trace (from perception/<model>/<bench>.jsonl, joined reasoning+content),
  segment into steps via split_steps().
  Per step k:
    mc_with(k):    K' sub-rollouts from cum_prefix WITH image
    mc_without(k): K' sub-rollouts from cum_prefix WITHOUT image (text-only)
    visual_dep(k) = mc_with - mc_without
  No VLM 4-axis verification (paradigm dropped per OC-VDM action guide §0.1).

Policy: qwen3.5-vl-4b-instruct (single endpoint; the model we will train via GRPO).
Judge:  397B remote.

Source of wrong samples: data/d2_hardbench/reports/root_cause.jsonl (combo, id).
Source of traces:        data/d2_hardbench/perception/<model>/<bench>.jsonl.

Output schema (per sample):
{
  combo, id, bench, scoring, question, gold, baseline_model,
  n_steps, K_prime,
  step_records: [
    {k, step_text,
     sub_rollouts_with, sub_preds_with, sub_outcomes_with, mc_value, mc_var,
     sub_rollouts_without, sub_preds_without, sub_outcomes_without,
     mc_without_value, mc_without_var, visual_dep}
  ]
}
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, statistics, sys
from collections import defaultdict
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
sys.path.insert(0, str(BASE / ".worktrees/research-chartprm/scripts"))
from preflight_reward_v1 import (split_steps, img_to_b64,
                                  relaxed_correctness, CHARXIV_J, CHARTMUSEUM_J)

ALL_MODELS = ["qwen3vl_4b", "qwen3vl_8b_thinking", "chart_r1", "chartgemma"]
ALL_BENCHES = ["chartqa_pro", "charxiv_reasoning", "chartmuseum"]


def reconstruct_trace(s: dict) -> str:
    """Trace = reasoning_content (think) + content (answer) for thinking models;
    or content alone for instruct models. Fallback to raw_response."""
    parts = []
    rc = (s.get('reasoning_content') or '').strip()
    if rc:
        parts.append(rc)
    c = (s.get('content') or '').strip()
    if c:
        parts.append(c)
    if not parts:
        return (s.get('raw_response') or '').strip()
    return "\n\n".join(parts)


def load_wrong_sample_ids(root_cause_path: Path) -> dict[str, list[str]]:
    """Returns combo (`bench__model`) → ordered list of wrong sample IDs."""
    out = defaultdict(list)
    for line in open(root_cause_path):
        d = json.loads(line)
        out[d['combo']].append(d['id'])
    return dict(out)


def load_perception(model: str, bench: str) -> dict[str, dict]:
    """Returns {id: full_sample_dict} from perception/<model>/<bench>.jsonl."""
    p = BASE / f"data/d2_hardbench/perception/{model}/{bench}.jsonl"
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}


async def mc_rollout_with_image(client, model, image_b64, question, prefix, K, sem):
    async with sem:
        try:
            messages = [
                {"role":"user","content":[
                    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                    {"type":"text","text":question}
                ]},
                {"role":"assistant","content": prefix},
            ]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.7, top_p=0.95, n=K,
                extra_body={"continue_final_message": True,
                            "add_generation_prompt": False,
                            "chat_template_kwargs":{"enable_thinking":False}})
            return [c.message.content or "" for c in resp.choices]
        except Exception as e:
            return [f"<gen_error:{str(e)[:80]}>" for _ in range(K)]


async def mc_rollout_without_image(client, model, question, prefix, K, sem):
    """Text-only user message: '[Chart image hidden] {question}'."""
    async with sem:
        try:
            messages = [
                {"role":"user","content": f"[Chart image hidden]\n{question}"},
                {"role":"assistant","content": prefix},
            ]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.7, top_p=0.95, n=K,
                extra_body={"continue_final_message": True,
                            "add_generation_prompt": False,
                            "chat_template_kwargs":{"enable_thinking":False}})
            return [c.message.content or "" for c in resp.choices]
        except Exception as e:
            return [f"<gen_error:{str(e)[:80]}>" for _ in range(K)]


async def judge_remote(client, model, prompt, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model, messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=64,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            t = (resp.choices[0].message.content or "").strip().lower()
            if "verdict" in t:
                suffix = t.split("verdict")[-1]
                if "incorrect" in suffix:
                    return 0
                if "correct" in suffix:
                    return 1
                return None
            if t.startswith("yes"): return 1
            if t.startswith("no"): return 0
            return None
        except Exception:
            return None


def extract_pred(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if m: return m.group(1).strip()[:200]
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    return (lines[-1] if lines else text[:200])[:200]


async def score_subrollout(text, bench, gold, question, judge_client, judge_model, judge_sem):
    pred = extract_pred(text)
    if not pred.strip():
        return 0, pred
    if bench == "chartqa_pro":
        return relaxed_correctness(pred, gold), pred
    tmpl = CHARXIV_J if bench == "charxiv_reasoning" else CHARTMUSEUM_J
    prompt = tmpl.format(q=question, a=gold, r=pred)
    o = await judge_remote(judge_client, judge_model, prompt, judge_sem)
    return o, pred


async def process_step(k, step_txt, cum_prefix, sample, image_b64,
                        policy_client, policy_model, judge_client, judge_model,
                        K_prime, mc_sem, judge_sem):
    bench, gold, question = sample['bench'], sample['gold'], sample['question']
    # Run mc_with and mc_without in parallel
    with_task = mc_rollout_with_image(policy_client, policy_model, image_b64, question,
                                       cum_prefix, K_prime, mc_sem)
    without_task = mc_rollout_without_image(policy_client, policy_model, question,
                                             cum_prefix, K_prime, mc_sem)
    with_completions, without_completions = await asyncio.gather(with_task, without_task)

    # Score in parallel
    with_score_tasks = [score_subrollout(c, bench, gold, question, judge_client, judge_model, judge_sem)
                         for c in with_completions]
    without_score_tasks = [score_subrollout(c, bench, gold, question, judge_client, judge_model, judge_sem)
                            for c in without_completions]
    with_results, without_results = await asyncio.gather(
        asyncio.gather(*with_score_tasks),
        asyncio.gather(*without_score_tasks),
    )

    def agg(score_results):
        outs = [r[0] for r in score_results]
        preds = [r[1] for r in score_results]
        valid = [o for o in outs if o is not None]
        mv = (sum(valid)/len(valid)) if valid else None
        mvar = statistics.pstdev(valid) if len(valid) >= 2 else 0.0
        return outs, preds, mv, mvar

    w_outs, w_preds, mc_w, mc_w_var = agg(with_results)
    wo_outs, wo_preds, mc_wo, mc_wo_var = agg(without_results)
    visual_dep = (mc_w - mc_wo) if (mc_w is not None and mc_wo is not None) else None

    return {
        "k": k, "step_text": step_txt[:400],
        "sub_rollouts_with":  [c[:400] for c in with_completions],
        "sub_preds_with":     w_preds,
        "sub_outcomes_with":  w_outs,
        "mc_value":           mc_w,
        "mc_var":             mc_w_var,
        "sub_rollouts_without": [c[:400] for c in without_completions],
        "sub_preds_without":    wo_preds,
        "sub_outcomes_without": wo_outs,
        "mc_without_value":     mc_wo,
        "mc_without_var":       mc_wo_var,
        "visual_dep":           visual_dep,
    }


async def process_sample(s, policy_client, policy_model, judge_client, judge_model,
                          K_prime, mc_sem, judge_sem):
    image_b64 = img_to_b64(s["image_path"])
    if not image_b64:
        return {**{k:v for k,v in s.items() if k not in ("trace","content","reasoning_content","raw_response","steps","step_claims","step_perception_scores")},
                "error": "no_image"}
    trace = reconstruct_trace(s)
    steps = split_steps(trace, max_steps=8)
    if not steps:
        return {**{k:v for k,v in s.items() if k not in ("trace","content","reasoning_content","raw_response","steps","step_claims","step_perception_scores")},
                "error": "no_steps"}

    step_records = []
    cum_prefix = ""
    for k, step_txt in enumerate(steps):
        cum_prefix = (cum_prefix + "\n" + step_txt).strip() if cum_prefix else step_txt.strip()
        rec = await process_step(k, step_txt, cum_prefix, s, image_b64,
                                  policy_client, policy_model, judge_client, judge_model,
                                  K_prime, mc_sem, judge_sem)
        step_records.append(rec)

    keep_keys = ("id","combo","bench","scoring","question","gold","baseline_model","image_path")
    base = {k:s.get(k) for k in keep_keys if k in s}
    base["n_steps"] = len(steps)
    base["K_prime"] = K_prime
    base["step_records"] = step_records
    return base


async def run_combo(model: str, bench: str, wrong_ids: list[str],
                    args, policy_client, judge_client, mc_sem, judge_sem):
    combo = f"{bench}__{model}"
    out_path = BASE / f"data/d2_hardbench/reports/image_dep_12combo_no_cap/{model}_{bench}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    perception = load_perception(model, bench)
    selected_ids = wrong_ids[:args.limit]
    samples = []
    for sid in selected_ids:
        if sid not in perception:
            continue
        s = perception[sid]
        # Normalize fields the pipeline expects
        s = {**s,
             "combo": combo,
             "gold": s.get('gold_answer'),
             "baseline_model": model}
        samples.append(s)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add(d['id'])
            except Exception:
                pass
    todo = [s for s in samples if s['id'] not in done]
    print(f"[{combo}] selected={len(samples)} done={len(done)} todo={len(todo)}")
    if not todo:
        return

    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    file_lock = asyncio.Lock()
    cnt = [0]

    async def go(s):
        async with sample_sem:
            try:
                rec = await process_sample(s, policy_client, args.policy_model,
                                            judge_client, args.judge_model,
                                            args.K_prime, mc_sem, judge_sem)
            except Exception as e:
                rec = {"combo": combo, "id": s['id'], "bench": bench,
                       "error": f"process_failed: {str(e)[:120]}"}
        async with file_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cnt[0] += 1
        if cnt[0] % 5 == 0 or cnt[0] == len(todo):
            print(f"  [{combo}] {cnt[0]}/{len(todo)}")

    await asyncio.gather(*[go(s) for s in todo])
    print(f"  [{combo}] DONE -> {out_path}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_url", default="http://localhost:8501/v1")
    ap.add_argument("--policy_model", default="qwen3vl_policy")
    ap.add_argument("--judge_url", default="http://10.1.211.147:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--K_prime", type=int, default=6)
    ap.add_argument("--mc_concurrency", type=int, default=6)
    ap.add_argument("--judge_concurrency", type=int, default=12)
    ap.add_argument("--sample_concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    ap.add_argument("--root_cause", default=str(BASE/"data/d2_hardbench/reports/root_cause.jsonl"))
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    benches = [b for b in args.benches.split(",") if b]
    wrong_by_combo = load_wrong_sample_ids(Path(args.root_cause))

    policy_client = AsyncOpenAI(base_url=args.policy_url, api_key="dummy", timeout=600.0)
    judge_client = AsyncOpenAI(base_url=args.judge_url, api_key="dummy", timeout=600.0)
    mc_sem = asyncio.Semaphore(args.mc_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    for bench in benches:
        for model in models:
            combo = f"{bench}__{model}"
            wrong_ids = wrong_by_combo.get(combo, [])
            if not wrong_ids:
                print(f"[{combo}] no wrong samples in root_cause; skip")
                continue
            await run_combo(model, bench, wrong_ids, args,
                            policy_client, judge_client, mc_sem, judge_sem)


if __name__ == "__main__":
    asyncio.run(main())
