"""Root-cause analysis on ALL wrong samples (outcome=0) across 12 combos.

For each wrong sample, 397B identifies the FIRST step in the model's reasoning
where the error chain begins, classified into 10 actionable root-cause categories.

Goal: find the single dominant first-error type → target for process reward redesign.
"""
from __future__ import annotations

import argparse, asyncio, json, os, re
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
HB = BASE / "data/d2_hardbench"

MODELS = ["qwen3vl_4b","qwen3vl_8b_thinking","chart_r1","chartgemma"]
BENCHES = ["chartqa_pro","charxiv_reasoning","chartmuseum"]


ROOT_CAUSE_PROMPT = """You are diagnosing a chart-reasoning failure. Pinpoint where the error CHAIN STARTS — the FIRST step in the model's reasoning trace that introduces the deviation leading to the wrong final answer. Then classify the first-error TYPE.

Question: {q}
Gold answer: {gold}
Model prediction: {pred}
Model reasoning trace (full or tail 2500 chars):
---
{trace}
---

Categories — pick the ONE that best describes the FIRST ERROR (not subsequent cascading errors):
A. Q_TARGET_MISREAD — model didn't capture the question's specific TARGET from the start (e.g., which series/year/condition to focus on, what "highest among X" refers to).
B. SUBSET_FILTER_MISSED — didn't apply a filter/constraint mentioned in the question (e.g., "excluding X", "top 3 only", "before 2018"), so used wrong domain throughout.
C. CHART_VALUE_MISREAD — misread an actual value, label, or axis from the chart at an early step (perception failure on raw data).
D. WRONG_GROUP_ANCHORED — read chart correctly but selected the WRONG group/series/cluster/category to operate on, and built downstream reasoning on that wrong anchor.
E. REASONING_GAP — skipped a necessary intermediate sub-step (e.g., didn't break compound question into parts; missed a required computation).
F. ARITHMETIC — chart values and target read correctly, but math operation (sum/diff/avg/ratio) was wrong.
G. PREMATURE_CONCLUSION — jumped to a final answer before completing analysis; reasoning trace shows incomplete logic.
H. TRUNCATED — reasoning trace ends mid-thought, never reaches a final answer (out of tokens).
I. FORMAT_MISMATCH — model's final answer is semantically correct but format differs from gold (units, $, decimal places, "Yes." vs "Yes").
J. OTHER — none of the above.

Respond with exactly:
LABEL: <letter A-J>
QUOTE: <verbatim sentence(s) from the trace marking the first error, max 200 chars>
REASON: <one line explaining why this is the root cause>
"""


def parse_resp(text: str) -> tuple[str | None, str, str]:
    m = re.search(r"LABEL:\s*([A-J])", text, re.I)
    label = m.group(1).upper() if m else None
    qm = re.search(r"QUOTE:\s*(.+?)(?=\n[A-Z]+:|\Z)", text, re.S | re.I)
    quote = (qm.group(1).strip()[:250] if qm else "")
    rm = re.search(r"REASON:\s*(.+?)(?=\n[A-Z]+:|\Z)", text, re.S | re.I)
    reason = (rm.group(1).strip()[:250] if rm else "")
    return label, quote, reason


LABEL_MAP = {
    "A":"Q_TARGET_MISREAD","B":"SUBSET_FILTER_MISSED","C":"CHART_VALUE_MISREAD",
    "D":"WRONG_GROUP_ANCHORED","E":"REASONING_GAP","F":"ARITHMETIC",
    "G":"PREMATURE_CONCLUSION","H":"TRUNCATED","I":"FORMAT_MISMATCH","J":"OTHER",
}


def collect_wrong_samples():
    out = []
    for bench in BENCHES:
        for model in MODELS:
            inf_p = HB / f"inference/{model}/{bench}.jsonl"
            outc_p = HB / f"outcome/outcome_{model}_{bench}.jsonl"
            if not (inf_p.exists() and outc_p.exists()): continue
            outc = {json.loads(l)["id"]: json.loads(l) for l in open(outc_p) if l.strip()}
            inf_records = {}
            for l in open(inf_p):
                if not l.strip(): continue
                r = json.loads(l)
                sid = r.get("sample_id") or r.get("id")
                if sid: inf_records[sid] = r
            for sid, o_r in outc.items():
                if o_r.get("outcome") != 0: continue
                ir = inf_records.get(sid, {})
                content = (ir.get("content","") or "")
                reasoning = (ir.get("reasoning_content","") or "")
                trace = (reasoning + "\n\n" + content)[-2500:].strip()
                if not trace:
                    trace = "(empty trace)"
                out.append({
                    "combo": f"{bench}__{model}",
                    "id": sid,
                    "question": (ir.get("question","") or "")[:400],
                    "gold": str(o_r.get("gold",""))[:200],
                    "pred": str(o_r.get("prediction",""))[:300],
                    "trace": trace,
                })
    return out


async def classify_one(client, prompt, model, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0, max_tokens=200,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"


def load_done(p):
    done = set()
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
                done.add(r["id"]+"||"+r["combo"])
            except Exception: pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--concurrent_per_host", type=int, default=4)
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    args = ap.parse_args()

    samples = collect_wrong_samples()
    print(f"collected {len(samples)} wrong samples across 12 combos")

    out_p = HB / "reports/root_cause.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_p)
    todo = [s for s in samples if (s["id"]+"||"+s["combo"]) not in done]
    print(f"  done={len(done)}, todo={len(todo)}")

    urls = args.urls.split(",")
    clients = [AsyncOpenAI(base_url=u, api_key="dummy", timeout=600.0) for u in urls]
    sem = asyncio.Semaphore(len(clients) * args.concurrent_per_host)
    cnt = {"i":0}
    def pick(): c = clients[cnt["i"] % len(clients)]; cnt["i"] += 1; return c
    file_lock = asyncio.Lock()

    async def go(s):
        prompt = ROOT_CAUSE_PROMPT.format(q=s["question"], gold=s["gold"], pred=s["pred"], trace=s["trace"])
        raw = await classify_one(pick(), prompt, args.judge_model, sem)
        label, quote, reason = parse_resp(raw)
        rec = {**{k:v for k,v in s.items() if k!="trace"},
               "label_letter": label, "label": LABEL_MAP.get(label,"PARSE_FAIL"),
               "quote": quote, "reason": reason, "raw": raw[:250]}
        async with file_lock:
            with open(out_p, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")

    cnt_d = [0]
    async def wrap(s):
        await go(s)
        cnt_d[0] += 1
        if cnt_d[0] % 50 == 0:
            print(f"  classified {cnt_d[0]}/{len(todo)}")

    await asyncio.gather(*[wrap(s) for s in todo])

    # Summary
    from collections import Counter
    by_combo = {}
    total = Counter()
    for line in open(out_p):
        r = json.loads(line)
        by_combo.setdefault(r["combo"], Counter())[r["label"]] += 1
        total[r["label"]] += 1
    grand = sum(total.values())
    json.dump({"per_combo": {k: dict(v) for k,v in by_combo.items()},
               "aggregate": dict(total), "grand": grand},
              open(HB / "reports/root_cause_summary.json","w"), indent=2, ensure_ascii=False)
    print(f"\nN={grand} aggregate:")
    for L, v in sorted(total.items(), key=lambda x:-x[1]):
        print(f"  {L}: {v}  ({100*v/max(grand,1):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
