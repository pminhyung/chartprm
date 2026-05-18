"""Tier 1-A: re-measure perception with STRICT verifier prompt on existing 64 steps.

Keeps MC sub-rollout data from math_shepherd_dead.jsonl (policy unchanged).
Only re-runs the 4-axis VLM verifier with a sharpened prompt that:
  - Requires DIRECTLY READABLE statements for YES (no estimation/hedging)
  - Pushes hedged or imprecise claims to NO
  - Reserves NA only for steps that make ZERO claim of that type

Output: data/d2_hardbench/reports/math_shepherd_dead_sharpened.jsonl
"""
from __future__ import annotations
import argparse, asyncio, base64, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


STRICT_STEP_4AXIS_PROMPT = """You are STRICTLY verifying ONE step of chart reasoning against the actual chart.

Step text from model:
\"\"\"
{step}
\"\"\"

For each of 4 axes, answer YES, NO, or NA.

CRITICAL RULES (read first):
- YES requires the statement to be DIRECTLY READABLE from the chart, not estimated, inferred, hedged, or approximated.
- If the step uses hedge words like "approximately", "around", "roughly", "about", "I think", "seems", "appears to be", "looks like" AND the exact value is not directly readable → NO.
- If the numerical value differs from the chart by more than 5% → NO.
- If the step mentions a label / legend / series / category that is NOT exactly present in the chart → NO.
- NA ONLY if the step makes ZERO claim of that type (true silence). If the step makes a vague claim, prefer NO over NA.

Q1 VALUE: Every numeric value or direct reading mentioned in the step matches the chart within 5%, with NO hedging. Hedge without exact correctness → NO.
Q2 LABEL: Every named label / legend / series / category / proper noun is present in the chart EXACTLY as stated. Misspell or approximate label → NO.
Q3 COUNT: Every count / "N items" / quantifier statement is exact. "There are about 5" with actual 4 or 6 → NO.
Q4 STRUCTURE: Every statement about axes / color meaning / spatial encoding / chart structure is correct.

Respond in EXACTLY this format (4 lines, no extra text):
Q1: <YES/NO/NA>
Q2: <YES/NO/NA>
Q3: <YES/NO/NA>
Q4: <YES/NO/NA>
"""


def parse_4axis(text: str):
    out = {"VALUE":None, "LABEL":None, "COUNT":None, "STRUCTURE":None}
    AX = {"Q1":"VALUE","Q2":"LABEL","Q3":"COUNT","Q4":"STRUCTURE"}
    for ln in text.split("\n"):
        m = re.match(r"\s*Q([1-4])\s*[:\-]\s*(YES|NO|NA|N/A)\b", ln, re.I)
        if m:
            ax = AX[f"Q{m.group(1)}"]
            val = m.group(2).upper()
            if val == "YES": out[ax] = 1
            elif val == "NO": out[ax] = 0
            else: out[ax] = None
    return out


def img_to_b64(path):
    try:
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    except Exception: return None


async def strict_verify(client, model, image_b64, step_text, sem):
    async with sem:
        prompt = STRICT_STEP_4AXIS_PROMPT.format(step=step_text[:1500])
        try:
            messages = [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{image_b64}"}},
                {"type":"text","text":prompt}]}]
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=1024,
                extra_body={"chat_template_kwargs":{"enable_thinking":True}})
            content = resp.choices[0].message.content or ""
            return parse_4axis(content), content[:300]
        except Exception as e:
            return {"VALUE":None,"LABEL":None,"COUNT":None,"STRUCTURE":None}, f"<err:{str(e)[:80]}>"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm_url", default="http://localhost:8500/v1")
    ap.add_argument("--vlm_model", default="qwen3vl_verifier")
    ap.add_argument("--input", default=str(BASE/"data/d2_hardbench/reports/math_shepherd_dead.jsonl"))
    ap.add_argument("--output", default=str(BASE/"data/d2_hardbench/reports/math_shepherd_dead_sharpened.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input)]
    print(f"Loaded {len(records)} samples; re-measuring perception with STRICT prompt")

    client = AsyncOpenAI(base_url=args.vlm_url, api_key="dummy", timeout=600.0)
    sem = asyncio.Semaphore(args.concurrency)

    # Cache image b64 by path
    img_cache = {}

    async def process_rec(rec):
        if "error" in rec:
            return rec
        img = img_cache.get(rec["image_path"])
        if img is None:
            img = img_to_b64(rec["image_path"])
            img_cache[rec["image_path"]] = img
        if img is None:
            rec["error"] = "no_image_for_sharpen"
            return rec
        # Re-verify each step
        tasks = [strict_verify(client, args.vlm_model, img, sr["step_text"], sem)
                 for sr in rec["step_records"]]
        results = await asyncio.gather(*tasks)
        for sr, (axis_results, raw) in zip(rec["step_records"], results):
            scored = [v for v in axis_results.values() if isinstance(v, int)]
            sr["axis_results_strict"] = axis_results
            sr["step_perception_strict"] = (sum(scored)/len(scored)) if scored else None
            sr["strict_raw"] = raw
        return rec

    tasks = [process_rec(r) for r in records]
    done = await asyncio.gather(*tasks)

    with open(args.output, "w") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
