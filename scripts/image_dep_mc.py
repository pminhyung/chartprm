"""Image-dependency MC: add mc_without_image to existing 64 step records.

Reuses mc_with_image data from math_shepherd_dead.jsonl (policy unchanged).
For each step, runs K'=6 sub-rollouts from same prefix but WITHOUT chart image
(text-only user msg). Scores outcomes via same judges. Computes visual_dep.

Output: data/d2_hardbench/reports/image_dep_dead.jsonl
Schema per step adds:
  mc_without_outcomes: list[int] of K'=6 outcomes
  mc_without_value: mean
  mc_without_var: std
  mc_without_preds: list[str]
  mc_without_completions: list[str] (first 600c each)
  visual_dep: mc_with - mc_without
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, statistics, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
sys.path.insert(0, str(BASE / ".worktrees/research-chartprm/scripts"))
from preflight_reward_v1 import relaxed_correctness, CHARXIV_J, CHARTMUSEUM_J


async def mc_without_image(policy_client, model, question, prefix, K, sem):
    """K continuations from prefix WITHOUT chart image (text-only user msg)."""
    async with sem:
        try:
            messages = [
                {"role": "user", "content": question},     # text only, no image_url
                {"role": "assistant", "content": prefix},
            ]
            resp = await policy_client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.7, top_p=0.95, max_tokens=1024, n=K,
                extra_body={"continue_final_message": True,
                            "add_generation_prompt": False,
                            "chat_template_kwargs":{"enable_thinking":False}})
            return [c.message.content or "" for c in resp.choices]
        except Exception as e:
            return [f"<gen_error:{str(e)[:80]}>" for _ in range(K)]


async def judge_remote(judge_client, model, prompt, sem):
    async with sem:
        try:
            resp = await judge_client.chat.completions.create(
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


async def score(text, sample, judge_client, judge_model, judge_sem):
    pred = extract_pred(text)
    if not pred.strip():
        return 0, pred
    bench = sample['bench']
    if bench == "chartqa_pro":
        return relaxed_correctness(pred, sample['gold']), pred
    tmpl = CHARXIV_J if bench == "charxiv_reasoning" else CHARTMUSEUM_J
    prompt = tmpl.format(q=sample['question'], a=sample['gold'], r=pred)
    return (await judge_remote(judge_client, judge_model, prompt, judge_sem)), pred


def reconstruct_cum_prefix(step_records, target_k):
    """Best-effort: cum_prefix at step k = concat of step_text[0..k] joined by newline."""
    parts = [sr['step_text'] for sr in step_records if sr['k'] <= target_k]
    return "\n".join(parts).strip()


async def process_record(rec, policy_client, policy_model, judge_client, judge_model,
                          K_prime, mc_sem, judge_sem):
    if 'error' in rec:
        return rec
    sample = {"id": rec['id'], "combo": rec['combo'], "question": rec['question'],
              "gold": rec['gold'], "bench": rec['bench']}
    for sr in rec['step_records']:
        cum_prefix = reconstruct_cum_prefix(rec['step_records'], sr['k'])
        sub_completions = await mc_without_image(policy_client, policy_model,
                                                   rec['question'], cum_prefix,
                                                   K_prime, mc_sem)
        score_tasks = [score(c, sample, judge_client, judge_model, judge_sem)
                       for c in sub_completions]
        score_results = await asyncio.gather(*score_tasks)
        outs = [r[0] for r in score_results]
        preds = [r[1] for r in score_results]
        valid_outs = [o for o in outs if o is not None]
        mc_without_value = (sum(valid_outs)/len(valid_outs)) if valid_outs else None
        mc_without_var = statistics.pstdev(valid_outs) if len(valid_outs) >= 2 else 0.0
        sr["mc_without_completions"] = [s[:600] for s in sub_completions]
        sr["mc_without_preds"] = preds
        sr["mc_without_outcomes"] = outs
        sr["mc_without_value"] = mc_without_value
        sr["mc_without_var"] = mc_without_var
        if sr.get('mc_value') is not None and mc_without_value is not None:
            sr["visual_dep"] = sr['mc_value'] - mc_without_value
        else:
            sr["visual_dep"] = None
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_url", default="http://localhost:8501/v1")
    ap.add_argument("--policy_model", default="qwen3vl_policy")
    ap.add_argument("--judge_url", default="http://10.1.211.147:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--input", default=str(BASE/"data/d2_hardbench/reports/math_shepherd_dead.jsonl"))
    ap.add_argument("--output", default=str(BASE/"data/d2_hardbench/reports/image_dep_dead.jsonl"))
    ap.add_argument("--K_prime", type=int, default=6)
    ap.add_argument("--mc_concurrency", type=int, default=5)
    ap.add_argument("--judge_concurrency", type=int, default=8)
    ap.add_argument("--sample_concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    if args.limit > 0: records = records[:args.limit]
    print(f"Loaded {len(records)} records; K_prime={args.K_prime}")

    done_keys = set()
    if Path(args.output).exists():
        for line in open(args.output):
            try:
                d = json.loads(line)
                done_keys.add(f"{d['id']}||{d['combo']}")
            except Exception: pass
    todo = [r for r in records if f"{r['id']}||{r['combo']}" not in done_keys]
    print(f"  done: {len(done_keys)}, todo: {len(todo)}")
    if not todo: return

    policy_client = AsyncOpenAI(base_url=args.policy_url, api_key="dummy", timeout=600.0)
    judge_client = AsyncOpenAI(base_url=args.judge_url, api_key="dummy", timeout=600.0)
    mc_sem = asyncio.Semaphore(args.mc_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    sample_sem = asyncio.Semaphore(args.sample_concurrency)
    file_lock = asyncio.Lock()
    cnt = [0]

    async def go(rec):
        async with sample_sem:
            try:
                out = await process_record(rec, policy_client, args.policy_model,
                                            judge_client, args.judge_model,
                                            args.K_prime, mc_sem, judge_sem)
            except Exception as e:
                out = {**rec, "error_image_dep": f"failed: {str(e)[:120]}"}
        async with file_lock:
            with open(args.output, "a") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        cnt[0] += 1
        print(f"  [{cnt[0]}/{len(todo)}] done: {rec['id']} ({rec['combo']})")

    await asyncio.gather(*[go(r) for r in todo])


if __name__ == "__main__":
    asyncio.run(main())
