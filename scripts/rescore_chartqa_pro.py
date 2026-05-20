"""M2: chartqa_pro re-scoring with judge LLM for open-ended gold answers.

Re-uses existing sub_preds_with / sub_preds_without (saved during measurement).
Routes by gold type:
  - bool (true/false/yes/no): exact-match
  - numeric (digit + optional %, ., -, M, B): relaxed_correctness
  - open-ended: judge LLM (Yes/No format)

Output: same schema as input, but with re-scored mc_value, mc_without_value,
        sub_outcomes_with, sub_outcomes_without per step.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, statistics, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
sys.path.insert(0, str(BASE / ".worktrees/research-chartprm/scripts"))
from preflight_reward_v1 import relaxed_correctness

CHARTQA_PRO_JUDGE = """Is the model's answer semantically equivalent to the ground truth?

Q: {q}
GT: {a}
Model: {r}

Consider:
- For open-ended answers, accept paraphrases that convey the same meaning
- For "Unanswerable"/"Not available"/"Cannot determine" type answers, accept any equivalent phrasing
- For lists/comparisons, accept if all key entities match

Reply only Yes or No."""


def gold_type(gold: str) -> str:
    g = (gold or "").strip()
    gl = g.lower()
    if gl in ("true", "false", "yes", "no"):
        return "bool"
    if re.match(r"^-?\d+\.?\d*\s*%?$", g):
        return "numeric"
    if re.match(r"^-?\d+\.?\d*\s*[MmBbKk]$", g):
        return "numeric"
    return "open"


async def judge_remote(client, model, prompt, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=16,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = (resp.choices[0].message.content or "").strip().lower()
            if t.startswith("yes"): return 1
            if t.startswith("no"): return 0
            return None
        except Exception:
            return None


async def score_pred(pred, gold, gtype, question, judge_client, judge_model, sem):
    if not pred.strip():
        return 0
    if gtype == "bool":
        return int(pred.strip().lower() == gold.strip().lower())
    if gtype == "numeric":
        return relaxed_correctness(pred, gold)
    # open-ended → judge
    prompt = CHARTQA_PRO_JUDGE.format(q=question, a=gold, r=pred)
    o = await judge_remote(judge_client, judge_model, prompt, sem)
    return o if o is not None else 0


async def rescore_sample(s, judge_client, judge_model, sem):
    gold = s.get("gold", "")
    gtype = gold_type(gold)
    question = s.get("question", "")

    if "step_records" not in s:
        return s, gtype

    new_srs = []
    for sr in s["step_records"]:
        with_preds = sr.get("sub_preds_with", [])
        without_preds = sr.get("sub_preds_without", [])

        with_tasks = [score_pred(p, gold, gtype, question, judge_client, judge_model, sem)
                      for p in with_preds]
        without_tasks = [score_pred(p, gold, gtype, question, judge_client, judge_model, sem)
                         for p in without_preds]
        with_outs, without_outs = await asyncio.gather(
            asyncio.gather(*with_tasks),
            asyncio.gather(*without_tasks),
        )

        with_valid = [o for o in with_outs if o is not None]
        without_valid = [o for o in without_outs if o is not None]
        mc_w = (sum(with_valid) / len(with_valid)) if with_valid else None
        mc_wo = (sum(without_valid) / len(without_valid)) if without_valid else None

        new_sr = {**sr}
        new_sr["sub_outcomes_with"] = with_outs
        new_sr["sub_outcomes_without"] = without_outs
        new_sr["mc_value"] = mc_w
        new_sr["mc_without_value"] = mc_wo
        new_sr["mc_var"] = statistics.pstdev(with_valid) if len(with_valid) >= 2 else 0.0
        new_sr["mc_without_var"] = statistics.pstdev(without_valid) if len(without_valid) >= 2 else 0.0
        new_sr["visual_dep"] = (mc_w - mc_wo) if (mc_w is not None and mc_wo is not None) else None
        new_srs.append(new_sr)

    s_new = {**s}
    s_new["step_records"] = new_srs
    s_new["_rescored"] = True
    s_new["_gold_type"] = gtype
    return s_new, gtype


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge_url", default="http://10.1.211.147:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"Loaded {len(samples)} samples from {args.input}")

    # Type distribution
    from collections import Counter
    types = Counter(gold_type(s.get("gold", "")) for s in samples)
    print(f"Gold types: {dict(types)}")

    judge_client = AsyncOpenAI(base_url=args.judge_url, api_key="dummy", timeout=600.0)
    sem = asyncio.Semaphore(args.concurrency)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cnt = [0]
    lock = asyncio.Lock()

    async def go(s):
        new_s, gt = await rescore_sample(s, judge_client, args.judge_model, sem)
        async with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(new_s, ensure_ascii=False) + "\n")
        cnt[0] += 1
        if cnt[0] % 5 == 0 or cnt[0] == len(samples):
            print(f"  {cnt[0]}/{len(samples)}")

    await asyncio.gather(*[go(s) for s in samples])


if __name__ == "__main__":
    asyncio.run(main())
