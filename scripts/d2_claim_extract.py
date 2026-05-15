"""D2 claim extraction — regex (Tier 1) + LLM fallback (Tier 2) + oracle cross-check.

Guide §1.4.1 (extractor) + §3.5 (validation via LLM oracle Jaccard agreement).

Input:  data/d2_pilot/mc_results.jsonl   (or samples.jsonl if mc not yet done)
Output: data/d2_pilot/claims.jsonl
        data/d2_pilot/claim_validation.json   (Jaccard, recall_proxy)

LLM hosts: text-only mode on the 9B verifier hosts (ports 8100, 8101).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

BASE = Path(os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr"))
IN_JSONL_DEFAULT = BASE / "data/d2_pilot/mc_results.jsonl"
IN_JSONL_FALLBACK = BASE / "data/d2_pilot/samples.jsonl"
OUT_JSONL = BASE / "data/d2_pilot/claims.jsonl"
OUT_VALIDATE = BASE / "data/d2_pilot/claim_validation.json"

# ─── Tier 1: regex extractor (guide §1.4.1) ──────────────────────────────
VALUE_PATTERNS = [
    # "Q4 = 50", "Q4: 50", "Q4 is 50"
    re.compile(
        r"(?P<entity>[A-Z][\w\s\-]{1,30}?)\s*(?:[=:]|\bis\b|\bequals\b)\s*"
        r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|million|billion|M|B|K)?",
        re.IGNORECASE,
    ),
    # "the value of Q4 is 50" / "for Q4 is 50"
    re.compile(
        r"(?:value\s+of\s+|for\s+)(?P<entity>[\w\s\-]{1,30}?)\s+is\s+"
        r"(?P<value>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    # "Q4 (50)"
    re.compile(
        r"(?P<entity>[A-Z][\w\s\-]{1,30})\s*\((?P<value>-?\d+(?:\.\d+)?)\)",
    ),
]

CATEGORICAL_PATTERN = re.compile(
    r"(?P<comparator>highest|lowest|maximum|minimum|largest|smallest)\s+"
    r"(?:value|category|entity|region)?\s*(?:is|=)\s*"
    r"(?P<entity>[\w\s\-]{1,30})",
    re.IGNORECASE,
)


def regex_extract(step: str) -> list[dict]:
    claims: list[dict] = []
    seen = set()
    for pat in VALUE_PATTERNS:
        for m in pat.finditer(step):
            ent = m.group("entity").strip()
            try:
                val = float(m.group("value"))
            except (ValueError, IndexError):
                continue
            key = (ent.lower(), val)
            if key in seen:
                continue
            seen.add(key)
            try:
                unit = m.group("unit")
            except IndexError:
                unit = None
            claims.append({"type": "value", "entity": ent, "value": val, "unit": unit})
    for m in CATEGORICAL_PATTERN.finditer(step):
        ent = m.group("entity").strip()
        comp = m.group("comparator").lower()
        key = (ent.lower(), comp)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"type": "categorical", "entity": ent, "value": None, "comparator": comp})
    return claims


# ─── Tier 2: LLM fallback (only when regex returns 0 but step has numerics) ──
LLM_EXTRACT_PROMPT = """Extract all explicit numeric or categorical claims from this reasoning step.
Return ONLY a JSON list of {{"entity": str, "value": number_or_null, "type": "value" or "categorical"}}.
Empty list [] if no explicit claim.

Step:
{step}

JSON:"""


def parse_llm_claims(text: str) -> list[dict]:
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ent = str(it.get("entity", "")).strip()
        if not ent:
            continue
        t = it.get("type", "value")
        v = it.get("value")
        try:
            v = float(v) if v is not None else None
        except Exception:
            v = None
        out.append({"type": t if t in ("value", "categorical") else "value",
                    "entity": ent, "value": v, "unit": None})
    return out


async def llm_extract(client: AsyncOpenAI, model: str, step: str, sem: asyncio.Semaphore) -> list[dict]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": LLM_EXTRACT_PROMPT.format(step=step)}],
                temperature=0.0,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_llm_claims(resp.choices[0].message.content or "")
        except Exception:
            return []


# ─── Normalization for Jaccard ─────────────────────────────────────────────
def normalize_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def claims_to_set(claims: list[dict]) -> set[tuple[str, str]]:
    s = set()
    for c in claims:
        ent = normalize_text(str(c.get("entity") or ""))
        if not ent:
            continue
        v = c.get("value")
        v_s = "" if v is None else f"{float(v):.2f}"
        s.add((ent, v_s))
    return s


# ─── Driver ────────────────────────────────────────────────────────────────
async def run(in_path: Path, ports: list[int], model: str):
    records = [json.loads(l) for l in open(in_path) if l.strip()]
    print(f"[claim] input={len(records)} from {in_path}")

    # First pass: regex on every step
    for r in records:
        r["step_claims"] = [regex_extract(s) for s in r["steps"]]

    # Tier 2: LLM fallback for empty-but-numeric steps
    clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="dummy", timeout=1800.0) for p in ports]
    sem = asyncio.Semaphore(len(clients) * 5)
    cnt = {"i": 0}
    def pick():
        c = clients[cnt["i"] % len(clients)]
        cnt["i"] += 1
        return c

    fallback_tasks = []  # (record, step_idx, future)
    for r in records:
        for si, step in enumerate(r["steps"]):
            if r["step_claims"][si]:
                continue
            if not re.search(r"\d", step):
                continue
            fallback_tasks.append((r, si, llm_extract(pick(), model, step, sem)))
    print(f"[claim] regex empty + numeric → LLM fallback queue: {len(fallback_tasks)} steps")

    if fallback_tasks:
        results = await asyncio.gather(*[fut for _, _, fut in fallback_tasks])
        for (r, si, _), res in zip(fallback_tasks, results):
            r["step_claims"][si] = res

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[claim] wrote {len(records)} -> {OUT_JSONL}")

    # Stats
    n_steps = sum(len(r["steps"]) for r in records)
    n_claims = sum(len(c) for r in records for c in r["step_claims"])
    n_steps_with = sum(1 for r in records for c in r["step_claims"] if c)
    print(f"[claim] total steps: {n_steps}")
    print(f"[claim] total claims extracted: {n_claims} (avg {n_claims/n_steps:.2f}/step)")
    print(f"[claim] steps with ≥1 claim: {n_steps_with} ({n_steps_with/n_steps*100:.1f}%)")

    # Oracle Jaccard cross-check (guide §3.5)
    print("\n[claim] Running oracle Jaccard cross-check on 50 random steps...")
    rng = random.Random(42)
    all_steps = [(r["id"], si, r["steps"][si], r["step_claims"][si])
                 for r in records for si in range(len(r["steps"]))]
    sample_n = min(50, len(all_steps))
    sample = rng.sample(all_steps, sample_n)

    async def oracle_one(text):
        async with sem:
            try:
                resp = await pick().chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": LLM_EXTRACT_PROMPT.format(step=text)}],
                    temperature=0.0,
                    max_tokens=512,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return parse_llm_claims(resp.choices[0].message.content or "")
            except Exception:
                return None

    futures = [oracle_one(text) for _, _, text, _ in sample]
    oracle_results = await asyncio.gather(*futures)

    jaccards = []
    both_empty = 0
    only_one_empty = 0
    for (sid, si, text, regex_c), oc in zip(sample, oracle_results):
        if oc is None:
            continue
        r_set = claims_to_set(regex_c)
        o_set = claims_to_set(oc)
        if not r_set and not o_set:
            jaccards.append(1.0); both_empty += 1
        elif not r_set or not o_set:
            jaccards.append(0.0); only_one_empty += 1
        else:
            jaccards.append(len(r_set & o_set) / len(r_set | o_set))

    mean_j = float(np.mean(jaccards)) if jaccards else 0.0
    recall_proxy = sum(1 for a in jaccards if a >= 0.5) / max(len(jaccards), 1)
    pass_jaccard = mean_j >= 0.70
    pass_recall = recall_proxy >= 0.85

    print(f"  N evaluated: {len(jaccards)} (both empty: {both_empty}, only_one_empty: {only_one_empty})")
    print(f"  mean Jaccard:  {mean_j:.3f}  (target ≥0.70 → {'✓' if pass_jaccard else '✗'})")
    print(f"  recall proxy:  {recall_proxy*100:.1f}%  (target ≥85% → {'✓' if pass_recall else '✗'})")

    report = {
        "n_records": len(records),
        "n_steps": n_steps,
        "n_claims": n_claims,
        "n_steps_with_claim": n_steps_with,
        "oracle_n": len(jaccards),
        "oracle_both_empty": both_empty,
        "oracle_only_one_empty": only_one_empty,
        "mean_jaccard": mean_j,
        "recall_proxy": recall_proxy,
        "pass_jaccard": pass_jaccard,
        "pass_recall": pass_recall,
    }
    OUT_VALIDATE.write_text(json.dumps(report, indent=2))
    print(f"[claim] validation -> {OUT_VALIDATE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(IN_JSONL_DEFAULT))
    ap.add_argument("--ports", required=True, help="comma-separated 9B verifier ports (text-only)")
    ap.add_argument("--model", default="Qwen3.5-VL-9B")
    args = ap.parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[claim] {in_path} not found, falling back to {IN_JSONL_FALLBACK}")
        in_path = IN_JSONL_FALLBACK
    ports = [int(p) for p in args.ports.split(",")]
    asyncio.run(run(in_path, ports, args.model))


if __name__ == "__main__":
    sys.exit(main() or 0)
