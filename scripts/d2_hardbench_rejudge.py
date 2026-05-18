"""Hard-bench cf-cell re-judge with extended answer extraction.

Problem detected: 98% of careful_flawed samples on hard bench lack `<answer>` tag.
The judge sees the FIRST line of content or full reasoning, both of which usually
contain only preamble ("Got it, let's solve..."), not the actual final answer.

Fix: extract candidate prediction from end-of-content+reasoning using patterns:
- last <answer>X</answer> tag (rare here, but kept for chartmuseum)
- regex on "(?:final answer|the answer is|answer:|=|so it is)\\s*[:\\s]*(.+)$"
- last sentence (5-30 words)
- last line if non-empty

Then re-judge via 397B remote multi-host (charxiv = CharXiv official prompt).
For chartqa_pro: re-apply relaxed_correctness on extracted pred.

Outputs:
- data/d2_hardbench/outcome_rejudge/<model>_<bench>.jsonl
- compare orig vs rejudge cf->grounded_correct flip rate
"""
from __future__ import annotations

import argparse, asyncio, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))


# ─── verbatim from d2_hardbench_outcome.py ───
def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> float:
    def _to_float(text: str):
        try:
            if text.endswith("%"): return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError: return None
    p, t = prediction.strip(), target.strip()
    pf, tf = _to_float(p), _to_float(t)
    if pf is not None and tf:
        return 1.0 if abs(pf - tf) / abs(tf) <= max_relative_change else 0.0
    return 1.0 if p.lower() == t.lower() else 0.0


CHARXIV_JUDGE = """You will be given a question, a ground truth answer, and a response from a model. Please determine if the response is consistent with the ground truth answer.

[Question]
{query}

[Ground Truth]
{answer}

[Response]
{response}

Please respond with "Correct" if the response is consistent with the ground truth answer (semantically equivalent, allowing minor variations in form), or "Incorrect" if it differs significantly. End your response with: Verdict: Correct OR Verdict: Incorrect"""

CHARTMUSEUM_JUDGE = """You are evaluating an answer to a chart question-answer task. Please compare the model's answer with the ground truth answer and decide if the model's answer is correct.

Question: {query}
Ground Truth Answer: {answer}
Model Answer: {response}

Is the model's answer correct? Reply with only "Yes" or "No"."""


def parse_charxiv(text: str) -> int:
    m = re.search(r"Verdict:\s*(\w+)", text, re.IGNORECASE)
    if not m: return -1
    return 1 if m.group(1).lower().startswith("correct") else 0


def parse_chartmuseum(text: str) -> int:
    t = text.strip().lower()
    if t.startswith("yes") or "yes" in t[:20]: return 1
    if t.startswith("no") or "no" in t[:20]: return 0
    return -1


# ─── Extended answer extraction ───
ANSWER_PATTERNS = [
    r"<answer>\s*(.*?)\s*</answer>",
    r"(?:final answer|the answer is|answer:|so the answer is|therefore,?\s*the answer is)\s*[:\s]*([^\n.]+)",
    r"(?:so it is|so we get|hence,?\s*it is|thus,?\s*it is)\s*[:\s]*([^\n.]+)",
    r"\\boxed\{([^}]+)\}",
]


def extract_candidate(content: str, reasoning: str) -> str:
    """Return best candidate final answer string for re-judge."""
    text = (reasoning + "\n" + content).strip() if reasoning else content.strip()
    if not text: return ""
    # 1. Pattern matches — prefer LAST hit (final answer at end of trace)
    for pat in ANSWER_PATTERNS:
        matches = list(re.finditer(pat, text, re.DOTALL | re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()
    # 2. Last non-empty line (often "X is the answer")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines: return ""
    # try last 3 lines, pick first non-meta
    for line in reversed(lines[-3:]):
        if 1 <= len(line) <= 200: return line
    return lines[-1][:200]


async def judge_remote(client: AsyncOpenAI, prompt: str, model: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=["chartqa_pro", "charxiv_reasoning", "chartmuseum"])
    ap.add_argument("--urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--concurrent_per_host", type=int, default=3)
    ap.add_argument("--judge_model", default="Qwen3.5-397B-A17B-FP8")
    args = ap.parse_args()

    perc_p = BASE / f"data/d2_hardbench/perception/{args.model}/{args.bench}.jsonl"
    out_p = BASE / f"data/d2_hardbench/outcome_rejudge/{args.model}_{args.bench}.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(perc_p) if l.strip()]
    done = load_done(out_p)
    todo = [r for r in recs if r["id"] not in done]
    print(f"[{args.model}/{args.bench}] total={len(recs)} done={len(done)} todo={len(todo)}")
    if not todo: return

    file_lock = asyncio.Lock()

    if args.bench == "chartqa_pro":
        # Deterministic
        for r in todo:
            content = r.get("content","") or ""
            reasoning = r.get("reasoning_content","") or ""
            pred_orig_firstline = (content.strip() or reasoning.strip()).split("\n")[0].strip()
            pred_ext = extract_candidate(content, reasoning)
            gold = r["gold_answer"]
            score_orig = int(relaxed_correctness(pred_orig_firstline, gold))
            score_ext = int(relaxed_correctness(pred_ext, gold))
            with open(out_p, "a") as f:
                f.write(json.dumps({
                    "id": r["id"], "bench": args.bench,
                    "scorer": "relaxed_correctness_ext",
                    "pred_first_line": pred_orig_firstline[:300],
                    "pred_extended": pred_ext[:300], "gold": gold,
                    "outcome_orig": score_orig, "outcome_ext": score_ext,
                }, ensure_ascii=False) + "\n")
        print(f"  deterministic done {len(todo)}")
        return

    # charxiv/chartmuseum: LLM judge via 397B remote
    urls = args.urls.split(",")
    clients = [AsyncOpenAI(base_url=u, api_key="dummy", timeout=900.0) for u in urls]
    sem = asyncio.Semaphore(len(clients) * args.concurrent_per_host)
    cnt = {"i": 0}
    def pick(): c = clients[cnt["i"] % len(clients)]; cnt["i"] += 1; return c

    prompt_tmpl = CHARXIV_JUDGE if args.bench == "charxiv_reasoning" else CHARTMUSEUM_JUDGE
    parser = parse_charxiv if args.bench == "charxiv_reasoning" else parse_chartmuseum
    scorer_tag = f"{args.bench}_397b_ext"

    async def judge_one(r):
        content = r.get("content","") or ""
        reasoning = r.get("reasoning_content","") or ""
        pred_ext = extract_candidate(content, reasoning)
        gold = r["gold_answer"]
        question = r["question"]
        prompt = prompt_tmpl.format(query=question, answer=gold, response=pred_ext or "(empty)")
        raw = await judge_remote(pick(), prompt, args.judge_model, sem)
        score_ext = parser(raw)
        rec = {"id": r["id"], "bench": args.bench, "scorer": scorer_tag,
               "pred_extended": pred_ext[:400], "gold": gold,
               "outcome_ext": score_ext if score_ext != -1 else None,
               "judge_raw_ext": raw[:200]}
        async with file_lock:
            with open(out_p, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    done_n = [0]
    async def wrap(r):
        await judge_one(r)
        done_n[0] += 1
        if done_n[0] % 20 == 0:
            print(f"  judged {done_n[0]}/{len(todo)}")

    await asyncio.gather(*[wrap(r) for r in todo])
    print(f"  charxiv re-judge done {len(todo)}")


if __name__ == "__main__":
    asyncio.run(main())
