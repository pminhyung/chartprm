"""Preflight reward system: 4-axis perception verifier + MC rollout × multiplicative gating.

For each of 50 wrong samples:
1. Segment reasoning trace into steps (line-blocks).
2. Per step, extract chart-fact statements + classify into 4 axes (VALUE / LABEL / COUNT / STRUCTURE).
3. For each axis with claim, call VLM verifier with image + claim → YES/NO.
4. step_perception = mean(axis_scores).
5. MC rollout: sample K continuations from prefix(1..k) using policy → outcome judged by 397B.
   step_mc_value = fraction correct.
6. step_reward = step_perception × step_mc_value (multiplicative).
7. sample_reward = step-weighted aggregate (first 2 steps × 1.5, rest × 1.0).

Outputs:
- data/d2_hardbench/reports/preflight_reward.jsonl (per-sample step traces)
- console summary: per-axis distribution, perception×MC interaction
"""
from __future__ import annotations

import argparse, asyncio, base64, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

# ─── Step segmentation (line-blocks separated by sentence-end + reasoning markers) ───
STEP_MARKERS = [r"\bstep\s*\d+", r"\bfirst,", r"\bnext,", r"\bnow,", r"\bso,", r"\bfinally,",
                r"\bwait[,\s]", r"\bhmm[,\s]", r"\blet[''s]me", r"\blet[''s]"]


def split_steps(trace: str, max_steps: int = 8) -> list[str]:
    if not trace.strip(): return []
    # Primary: split on step markers
    text = trace.replace("\n\n", "\n")
    chunks = re.split(r"\n(?=" + "|".join(STEP_MARKERS) + ")", text, flags=re.I)
    if len(chunks) < 2:
        # Fallback: ~120-char chunks at sentence boundaries
        sents = re.split(r"(?<=[.?!])\s+", text)
        chunks, cur = [], ""
        for s in sents:
            if len(cur)+len(s) > 200 and cur:
                chunks.append(cur.strip()); cur = s
            else: cur += " " + s
        if cur.strip(): chunks.append(cur.strip())
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 15]
    return chunks[:max_steps]


# ─── Per-axis claim extraction (regex/heuristic from step text) ───
def extract_claims(step: str) -> dict[str, str]:
    """Return up to 1 claim per axis. Empty string if no claim detected."""
    claims = {"VALUE":"", "LABEL":"", "COUNT":"", "STRUCTURE":""}
    text = step

    # VALUE: numeric with unit or value-like ("X is N%", "approximately N", "value of N")
    m = re.search(r"((?:value|approximately|about|around|roughly|approx\.?|≈|~)\s*[\w\s\-,]{0,30}?-?\d+\.?\d*\s*(?:%|m|gb|usd|million|billion|k\b|million|p\b)?)", text, re.I)
    if not m:
        m = re.search(r"(\d+\.?\d*\s*(?:%|usd|million|billion|m\s|k\s|p\b))", text, re.I)
    if m: claims["VALUE"] = m.group(1).strip()[:150]

    # LABEL: legend/series/category name (after "is the", "called", "name is", or proper noun in quotes)
    m = re.search(r'((?:legend|series|category|method|model|line|bar|cluster)\s+(?:is|are|of|for|called|named|labeled)\s+["\']?[\w\-/\s]+["\']?)', text, re.I)
    if not m:
        # capitalized 2+ word noun phrase
        m = re.search(r"\b([A-Z][\w\-/]{2,}(?:\s+[A-Z][\w\-/]{2,})+)\b", text)
    if m: claims["LABEL"] = m.group(1).strip()[:150]

    # COUNT: "N items", "there are N", "X cheeses", "N bars"
    m = re.search(r"((?:there\s+are|there\s+is|i\s+see|i\s+count|count\s+is|total\s+of|number\s+of)\s+\d+\s+[\w\-/\s]{1,40})", text, re.I)
    if not m:
        m = re.search(r"(\b\d+\s+(?:items?|bars?|categories|series|methods?|movies?|clusters?|points?|cells?|countries|people))", text, re.I)
    if m: claims["COUNT"] = m.group(1).strip()[:150]

    # STRUCTURE: axis/encoding/color claims
    m = re.search(r"((?:x[-\s]?axis|y[-\s]?axis|z[-\s]?axis|horizontal\s+axis|vertical\s+axis|legend|color|shade|bar\s+(?:height|width|area)|top[-\s]?(?:left|right)|bottom[-\s]?(?:left|right)|leftmost|rightmost)[\s\w,'\-]{0,60})", text, re.I)
    if m: claims["STRUCTURE"] = m.group(1).strip()[:150]

    return claims


# ─── Single-call per-step 4-axis verifier (VLM judges all 4 axes from step text + chart) ───
STEP_4AXIS_PROMPT = """You are verifying one step of chart reasoning against the actual chart.

Step text from model:
\"\"\"
{step}
\"\"\"

Evaluate this step on 4 axes. For each axis, answer YES (the claim is correct), NO (claim is wrong), or NA (step makes no such claim).

Q1 VALUE: Are all numeric values / direct readings mentioned in the step approximately correct (≤5% off) compared to the chart?
Q2 LABEL: Are all named labels / legend / series / category / proper nouns mentioned correctly present in the chart?
Q3 COUNT: Are any count / "N items" / quantifier statements accurate?
Q4 STRUCTURE: Are statements about axes / color meaning / spatial encoding / chart structure correct?

Respond in EXACTLY this format (4 lines):
Q1: <YES/NO/NA>
Q2: <YES/NO/NA>
Q3: <YES/NO/NA>
Q4: <YES/NO/NA>
"""


def parse_4axis(text: str) -> dict[str, int | None]:
    out = {"VALUE":None,"LABEL":None,"COUNT":None,"STRUCTURE":None}
    AX = {"Q1":"VALUE","Q2":"LABEL","Q3":"COUNT","Q4":"STRUCTURE"}
    for ln in text.split("\n"):
        m = re.match(r"\s*Q([1-4])\s*[:\-]\s*(YES|NO|NA|N/A)\b", ln, re.I)
        if m:
            ax = AX[f"Q{m.group(1)}"]
            val = m.group(2).upper()
            if val == "YES": out[ax] = 1
            elif val == "NO": out[ax] = 0
            else: out[ax] = None  # NA → no claim of that type
    return out


def img_to_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def parse_yn(text: str) -> int | None:
    t = text.strip().lower()
    if t.startswith("yes") or " yes" in t[:20]: return 1
    if t.startswith("no") or " no" in t[:20]: return 0
    return None


async def vlm_verify_step(client: AsyncOpenAI, model: str, image_b64: str,
                          step_text: str, sem) -> dict[str, int | None]:
    """Single VLM call returns 4-axis verdict for one step."""
    async with sem:
        prompt = STEP_4AXIS_PROMPT.format(step=step_text[:1500])
        try:
            messages = [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                {"type":"text","text":prompt}]}]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=1024)
            content = resp.choices[0].message.content or ""
            return parse_4axis(content)
        except Exception:
            return {"VALUE":None,"LABEL":None,"COUNT":None,"STRUCTURE":None}


# ─── Outcome judge (re-use 397B remote for relaxed_text_match / charxiv_judge / chartmuseum_judge) ───
def relaxed_correctness(p: str, t: str) -> int:
    def _f(x):
        x = x.strip()
        try:
            if x.endswith("%"): return float(x.rstrip("%"))/100.0
            return float(x)
        except Exception: return None
    pf, tf = _f(p), _f(t)
    if pf is not None and tf:
        return 1 if abs(pf-tf)/abs(tf) <= 0.05 else 0
    return 1 if p.strip().lower() == t.strip().lower() else 0


CHARXIV_J = """You will be given a question, a ground truth answer, and a response. Determine if response is consistent with ground truth.
Q: {q}
GT: {a}
Resp: {r}
Reply with 'Verdict: Correct' or 'Verdict: Incorrect'."""
CHARTMUSEUM_J = """Is the model's answer correct vs ground truth?
Q: {q}
GT: {a}
Model: {r}
Reply only Yes or No."""


async def judge_remote(judge_client: AsyncOpenAI, model: str, prompt: str, sem) -> int | None:
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


# ─── MC rollout from prefix (K continuations) ───
async def mc_rollout(policy_client: AsyncOpenAI, model: str, image_b64: str,
                     question: str, prefix: str, K: int, sem) -> list[str]:
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
                            "chat_template_kwargs":{"enable_thinking":False}})
            return [c.message.content or "" for c in resp.choices]
        except Exception:
            return []


# ─── Orchestrator ───
async def process_sample(s, vlm_client, vlm_model, policy_client, policy_model,
                         judge_client, judge_model, vlm_sem, mc_sem, judge_sem,
                         do_mc: bool = True, K: int = 4):
    image_b64 = img_to_b64(s["image_path"])
    if not image_b64:
        return {**{k:v for k,v in s.items() if k!="trace"}, "error":"no_image"}
    steps = split_steps(s["trace"])
    step_records = []
    cum_prefix = ""
    for k, step_txt in enumerate(steps):
        # Single VLM call returns all 4 axes
        axis_results = await vlm_verify_step(vlm_client, vlm_model, image_b64,
                                              step_txt, vlm_sem)
        scored = [v for v in axis_results.values() if v is not None]
        step_perc = (sum(scored)/len(scored)) if scored else None
        claims = {}  # no separate extraction; VLM handles inline
        # MC rollout
        cum_prefix = (cum_prefix + "\n" + step_txt).strip()
        mc_value, mc_outcomes = None, []
        if do_mc and step_perc is not None and step_perc >= 0.5:
            rollouts = await mc_rollout(policy_client, policy_model, image_b64,
                                         s["question"], cum_prefix, K, mc_sem)
            # Judge each rollout
            judge_tasks = []
            for r in rollouts:
                # extract final answer
                m = re.search(r"<answer>(.*?)</answer>", r, re.S | re.I)
                pred = (m.group(1).strip() if m else r.strip().split("\n")[-1][:200])
                bench = s["bench"]
                if bench == "chartqa_pro":
                    mc_outcomes.append(relaxed_correctness(pred, s["gold"]))
                else:
                    tmpl = CHARXIV_J if bench == "charxiv_reasoning" else CHARTMUSEUM_J
                    prompt = tmpl.format(q=s["question"], a=s["gold"], r=pred)
                    judge_tasks.append(judge_remote(judge_client, judge_model, prompt, judge_sem))
            if judge_tasks:
                jr = await asyncio.gather(*judge_tasks)
                mc_outcomes.extend([x for x in jr if x is not None])
            mc_value = (sum(mc_outcomes)/len(mc_outcomes)) if mc_outcomes else None
        elif do_mc and step_perc is not None and step_perc < 0.5:
            mc_value = 0.0  # hard gating: skip MC, assume failure
        step_reward = None
        if step_perc is not None and mc_value is not None:
            step_reward = step_perc * mc_value
        step_records.append({
            "k": k, "step_text": step_txt[:300],
            "claims": {ax: c for ax, c in claims.items() if c},
            "axis_results": axis_results, "step_perception": step_perc,
            "mc_value": mc_value, "mc_outcomes": mc_outcomes, "step_reward": step_reward,
        })
    # sample reward: cascade-weighted aggregate (first 2 steps ×1.5)
    rs = [(1.5 if r["k"] < 2 else 1.0, r["step_reward"])
          for r in step_records if r["step_reward"] is not None]
    sample_reward = (sum(w*r for w,r in rs)/sum(w for w,_ in rs)) if rs else None
    return {**{k:v for k,v in s.items() if k!="trace"},
            "n_steps": len(steps), "step_records": step_records,
            "sample_reward": sample_reward}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_url", default="http://localhost:8500/v1")
    ap.add_argument("--vlm_model", default="qwen3vl_verifier")
    ap.add_argument("--policy_url", default="http://localhost:8501/v1")
    ap.add_argument("--policy_model", default="qwen3vl_policy")
    ap.add_argument("--judge_urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--input", default=str(BASE/"data/d2_hardbench/reports/preflight_50.jsonl"))
    ap.add_argument("--output", default=str(BASE/"data/d2_hardbench/reports/preflight_reward.jsonl"))
    ap.add_argument("--no_mc", action="store_true", help="skip MC rollout (verifier-only validation)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--vlm_concurrency", type=int, default=4)
    ap.add_argument("--mc_concurrency", type=int, default=2)
    ap.add_argument("--judge_concurrency", type=int, default=8)
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.input) if l.strip()]
    print(f"loaded {len(samples)} preflight samples")

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
    judge_clients = [AsyncOpenAI(base_url=u, api_key="dummy", timeout=300.0)
                     for u in args.judge_urls.split(",")]
    # Use first judge client for simplicity (could round-robin)
    judge_client = judge_clients[0]
    vlm_sem = asyncio.Semaphore(args.vlm_concurrency)
    mc_sem = asyncio.Semaphore(args.mc_concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)

    file_lock = asyncio.Lock()
    cnt_d = [0]
    async def go(s):
        rec = await process_sample(s, vlm_client, args.vlm_model,
                                    policy_client, args.policy_model,
                                    judge_client, args.judge_model,
                                    vlm_sem, mc_sem, judge_sem,
                                    do_mc=(not args.no_mc), K=args.K)
        async with file_lock:
            with open(args.output, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
        cnt_d[0] += 1
        if cnt_d[0] % 5 == 0:
            print(f"  processed {cnt_d[0]}/{len(todo)}")

    await asyncio.gather(*[go(s) for s in todo])
    print(f"done: {len(todo)} samples processed -> {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
