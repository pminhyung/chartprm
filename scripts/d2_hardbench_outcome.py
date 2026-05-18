"""Hard-bench outcome scoring (per-bench standard).

- chartqa_pro      → score_standard.relaxed_correctness (LMMs-Eval verbatim, deterministic)
- charxiv_reasoning → CharXiv official LLM judge (Qwen3.5-397B remote)
- chartmuseum      → ChartMuseum official LLM judge (Qwen3.5-397B remote)

Multi-host async (4 remote 397B URLs), sample-level append, resume by id.

Reads perception jsonl (has content, reasoning_content, gold_answer, scoring tag).
Outputs: outcome_<model>_<bench>.jsonl with id, outcome (0/1), scorer.
"""
from __future__ import annotations

import argparse, asyncio, json, os, re, sys
from pathlib import Path
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))

# ───── Deterministic chartqa_pro scorer (verbatim from score_standard.py) ─────
def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> float:
    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None
    p, t = prediction.strip(), target.strip()
    pf, tf = _to_float(p), _to_float(t)
    if pf is not None and tf:
        return 1.0 if abs(pf - tf) / abs(tf) <= max_relative_change else 0.0
    return 1.0 if p.lower() == t.lower() else 0.0

# ───── CharXiv judge (verbatim from third_party/CharXiv/src/prompts.py) ─────
CHARXIV_JUDGE = """You will be given a question, a ground truth answer, and a response from a model. Please determine if the response is consistent with the ground truth answer.

[Question]
{query}

[Ground Truth]
{answer}

[Response]
{response}

Please respond with "Correct" if the response is consistent with the ground truth answer (semantically equivalent, allowing minor variations in form), or "Incorrect" if it differs significantly. End your response with: Verdict: Correct OR Verdict: Incorrect"""

# ───── ChartMuseum judge (verbatim from third_party/ChartMuseum) ─────
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


# ───── Extract final answer from content/reasoning ─────
def extract_final_answer(content: str, reasoning: str, bench: str) -> str:
    """For chartmuseum: <answer>X</answer> regex.
       For chartqa_pro: content stripped.
       For charxiv: content as-is."""
    text = content.strip() if content.strip() else reasoning.strip()
    if bench == "chartmuseum":
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if m: return m.group(1).strip()
        # fallback: last line
        return text.split("\n")[-1].strip() if text else ""
    # chartqa_pro: take first line or full content
    return text.split("\n")[0].strip() if "\n" in text else text


async def judge_one(client: AsyncOpenAI, model: str, prompt: str,
                    sem: asyncio.Semaphore) -> str:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"__ERROR__: {type(e).__name__}: {str(e)[:160]}"


def load_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done.add(json.loads(line)["id"])
                except Exception: pass
    return done


async def run(perception_path: Path, output_path: Path, bench: str,
              urls: list[str], model: str, concurrent_per_host: int):
    recs = [json.loads(l) for l in open(perception_path) if l.strip()]
    done = load_done(output_path)
    todo = [r for r in recs if r["id"] not in done]
    print(f"[outcome] bench={bench}  total={len(recs)} done={len(done)} todo={len(todo)}")
    if not todo: return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_lock = asyncio.Lock()

    # Deterministic for chartqa_pro
    if bench == "chartqa_pro":
        for r in todo:
            content = r.get("content", "") or ""
            reasoning = r.get("reasoning_content", "") or ""
            pred = extract_final_answer(content, reasoning, bench)
            gold = r["gold_answer"]
            score = float(relaxed_correctness(pred, gold))
            out_rec = {"id": r["id"], "bench": bench, "scorer": "relaxed_correctness",
                       "prediction": pred[:300], "gold": gold,
                       "outcome": int(score)}
            with open(output_path, "a") as f:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        print(f"  deterministic done {len(todo)}")
        return

    # LLM judge for charxiv / chartmuseum via remote 397B
    clients = [AsyncOpenAI(base_url=u, api_key="dummy", timeout=1800.0) for u in urls]
    sem = asyncio.Semaphore(len(clients) * concurrent_per_host)
    cnt = {"i": 0}
    def pick():
        c = clients[cnt["i"] % len(clients)]; cnt["i"] += 1; return c

    async def judge_sample(r):
        content = r.get("content", "") or ""
        reasoning = r.get("reasoning_content", "") or ""
        pred = extract_final_answer(content, reasoning, bench)
        gold = r["gold_answer"]
        if bench == "charxiv_reasoning":
            prompt = CHARXIV_JUDGE.format(query=r["question"], answer=gold, response=pred)
            raw = await judge_one(pick(), model, prompt, sem)
            verdict = parse_charxiv(raw)
        else:  # chartmuseum
            prompt = CHARTMUSEUM_JUDGE.format(query=r["question"], answer=gold, response=pred)
            raw = await judge_one(pick(), model, prompt, sem)
            verdict = parse_chartmuseum(raw)
        out_rec = {"id": r["id"], "bench": bench,
                   "scorer": f"{bench}_judge_397b",
                   "prediction": pred[:300], "gold": gold[:300],
                   "outcome": verdict if verdict != -1 else None,
                   "judge_raw": raw[:300]}
        async with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    tasks = [judge_sample(r) for r in todo]
    n_done = 0
    for f in asyncio.as_completed(tasks):
        await f; n_done += 1
        if n_done % 10 == 0 or n_done == len(todo):
            print(f"  judge: {n_done}/{len(todo)}", flush=True)
    # Stats
    rows = [json.loads(l) for l in open(output_path) if l.strip()]
    valid = [r for r in rows if r["outcome"] is not None]
    if valid:
        print(f"  outcome pass rate: {sum(r['outcome'] for r in valid)/len(valid)*100:.1f}% ({len(valid)} parsed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bench", required=True, choices=["chartqa_pro", "charxiv_reasoning", "chartmuseum"])
    ap.add_argument("--urls", default="http://10.1.211.147:8000/v1,http://10.1.211.148:8000/v1,http://10.1.211.169:8000/v1,http://10.1.211.170:8000/v1")
    ap.add_argument("--model", default="Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--concurrent_per_host", type=int, default=4)
    args = ap.parse_args()
    urls = [u.strip() for u in args.urls.split(",")]
    asyncio.run(run(Path(args.perception), Path(args.output), args.bench, urls, args.model, args.concurrent_per_host))


if __name__ == "__main__":
    sys.exit(main() or 0)
