"""Hard-bench cf-cell automated reasoning error taxonomy.

For each cf sample (perception ≥0.5 AND outcome = 0) on 8 completed combos:
  Q + gold + pred (content+reasoning tail) → 397B classifier → 1 of 5 labels:
    TRUNCATION: model output preamble only, no final answer reached
    FORMAT_MISMATCH: answer present but format differs (extra trailing chars, wrong units)
    Q_INTENT: model misinterpreted question (e.g., answered when gold = "Unanswerable")
    CATEGORY_SELECT: model picked wrong group/cluster/name
    ARITHMETIC: chart values read correctly but math wrong
    OTHER: doesn't fit above

Outputs:
- data/d2_hardbench/reports/cf_taxonomy.jsonl (per-sample labels)
- data/d2_hardbench/reports/cf_taxonomy_summary.json (counts per combo × label)
"""
from __future__ import annotations

import argparse, asyncio, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
HB = BASE / "data/d2_hardbench"


TAXONOMY_PROMPT = """You are classifying a chart-reasoning failure case into ONE of 6 categories.

Question: {q}
Gold answer: {gold}
Model's full reasoning + content (tail 1500 chars):
{trace}

Categories (pick exactly ONE):
A. TRUNCATION — model output preamble only (e.g., "Got it, let's solve…", "Step 1: Examine…") and never reached a final answer.
B. FORMAT_MISMATCH — model produced an answer but format differs (extra trailing chars, wrong units, "$X" vs "X", "Yes." vs "Yes").
C. Q_INTENT — model misinterpreted the question (e.g., answered a value when gold = "Unanswerable", named a category when question asked for a percentage).
D. CATEGORY_SELECT — chart-reading was correct but model picked the WRONG group/cluster/series/method/name.
E. ARITHMETIC — chart values read correctly but the math (sum/diff/avg/ratio) is wrong.
F. OTHER — does not fit above (e.g., gibberish, meta-instruction, "**Example:**").

Respond with EXACTLY one letter (A/B/C/D/E/F) followed by ONE short reason line.
Format: "LABEL: <one_letter>\\nREASON: <short>"
"""


def parse_label(text: str) -> tuple[str | None, str]:
    m = re.search(r"LABEL:\s*([A-F])", text, re.IGNORECASE)
    if not m: return None, text[:200]
    letter = m.group(1).upper()
    rm = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
    reason = rm.group(1).strip()[:200] if rm else ""
    return letter, reason


LABEL_MAP = {
    "A": "TRUNCATION",
    "B": "FORMAT_MISMATCH",
    "C": "Q_INTENT",
    "D": "CATEGORY_SELECT",
    "E": "ARITHMETIC",
    "F": "OTHER",
}


def collect_cf_samples():
    """Return [(combo_key, sample_dict)] for all cf samples (perception ≥0.5, outcome 0)."""
    import numpy as np
    MODELS = ["qwen3vl_4b","qwen3vl_8b_thinking","chart_r1","chartgemma"]
    BENCHES = ["chartqa_pro","charxiv_reasoning","chartmuseum"]
    out = []
    for bench in BENCHES:
        for model in MODELS:
            perc_p = HB / f"perception/{model}/{bench}.jsonl"
            outc_p = HB / f"outcome/outcome_{model}_{bench}.jsonl"
            if not (perc_p.exists() and outc_p.exists()): continue
            outc = {json.loads(l)["id"]: json.loads(l) for l in open(outc_p) if l.strip()}
            for line in open(perc_p):
                if not line.strip(): continue
                r = json.loads(line)
                sid = r["id"]
                if not outc.get(sid) or outc[sid].get("outcome") != 0: continue
                scores = [s["mean_score"] for s in r.get("step_perception_scores",[])
                          if s and s.get("mean_score") is not None]
                if not scores: continue
                p_agg = float(np.mean(scores))
                if p_agg < 0.5: continue
                content = (r.get("content","") or "")
                reasoning = (r.get("reasoning_content","") or "")
                trace = (reasoning + "\n" + content)[-1500:]
                out.append({
                    "combo": f"{bench}__{model}",
                    "id": sid, "question": r["question"][:300],
                    "gold": outc[sid]["gold"][:200],
                    "trace": trace,
                    "p_agg": p_agg,
                })
    return out


async def classify_one(client, prompt, model, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=80,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"


def load_done(p):
    done = set()
    if p.exists():
        for line in open(p):
            try: done.add(json.loads(line)["id"]+"||"+json.loads(line)["combo"])
            except Exception: pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--concurrent_per_host", type=int, default=2)
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    args = ap.parse_args()

    samples = collect_cf_samples()
    print(f"collected {len(samples)} cf samples across 8 combos")

    out_p = HB / "reports/cf_taxonomy.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_p)
    todo = [s for s in samples if (s["id"]+"||"+s["combo"]) not in done]
    print(f"  done={len(done)} todo={len(todo)}")

    urls = args.urls.split(",")
    clients = [AsyncOpenAI(base_url=u, api_key="dummy", timeout=600.0) for u in urls]
    sem = asyncio.Semaphore(len(clients) * args.concurrent_per_host)
    cnt = {"i":0}
    def pick(): c = clients[cnt["i"] % len(clients)]; cnt["i"] += 1; return c
    file_lock = asyncio.Lock()

    async def go(s):
        prompt = TAXONOMY_PROMPT.format(q=s["question"], gold=s["gold"], trace=s["trace"])
        raw = await classify_one(pick(), prompt, args.judge_model, sem)
        letter, reason = parse_label(raw)
        rec = {**{k:v for k,v in s.items() if k!="trace"},
               "label_letter": letter, "label": LABEL_MAP.get(letter,"PARSE_FAIL"),
               "reason": reason, "raw": raw[:200]}
        async with file_lock:
            with open(out_p,"a") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n")

    done_n = [0]
    async def wrap(s):
        await go(s)
        done_n[0] += 1
        if done_n[0] % 30 == 0:
            print(f"  classified {done_n[0]}/{len(todo)}")

    await asyncio.gather(*[wrap(s) for s in todo])

    # summarize
    summary = {}
    for line in open(out_p):
        r = json.loads(line)
        summary.setdefault(r["combo"], {}).setdefault(r["label"], 0)
        summary[r["combo"]][r["label"]] += 1
    sum_p = HB / "reports/cf_taxonomy_summary.json"
    sum_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out_p}")
    print(f"-> {sum_p}")
    for combo, labs in summary.items():
        print(f"  {combo}: " + " ".join(f"{k}={v}" for k,v in sorted(labs.items())))


if __name__ == "__main__":
    asyncio.run(main())
