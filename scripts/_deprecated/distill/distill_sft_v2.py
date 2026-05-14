"""SFT v8 Distill v2 — re-distill 28,594 rows through Qwen3.6-27B VLM teacher.

Per-source bench-format prompts (ChartQA / CharXiv / ChartMuseum) → teacher CoT
+ answer in the matching family target shape. Sample-level append-mode JSONL,
resume by sample_id, drop-on-mismatch.

See docs/sft_v8_distill_v2_spec.md for the full spec.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Project root for chartvr package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI

from chartvr.config import SAMPLING_PARAMS

# ── Family templates (verbatim from third_party publisher) ──

CHARTQA_POST = "\nAnswer the question with a single word."

CHARTMUSEUM_QA_PROMPT = """Please answer the question using the chart image.

Question: [QUESTION]

Please first generate your reasoning process and then provide the user with the answer. Use the following format:

<think>
... your thinking process here ...
</think>
<answer>
... your final answer (entity(s) or number) ...
</answer>"""

# CharXiv REASONING_RESP_INST (verbatim from third_party/CharXiv/src/constants.py)
CHARXIV_INST = {
    2: """{}
    * If there are options in the question, your final answer must conform to one of the options.
    * If there are additional instructions in the question, follow them accordingly.
    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only.
    """,
    4: """{}
    {}
    """,
}


def _charxiv_number_instruction(answer: str) -> str:
    """For inst_category=4: derive decimal-place instruction from gold answer."""
    a = str(answer).strip()
    if "." in a:
        try:
            float(a)
            decimals = len(a.split(".")[1])
            return f"* Your final answer must be a number with {decimals} decimal places."
        except ValueError:
            pass
    return "* Your final answer must be a number."


# ── Source → family mapping ──

CHARTQA_SOURCES = {"chartqa_train"}
CHARXIV_SOURCES = {"scientific", "scientific_ext"}
CHARTMUSEUM_SOURCES = {"plotly_complex", "kaggle_like", "synthetic",
                       "additional", "owid", "worldbank"}


def family_of(source: str) -> str:
    if source in CHARTQA_SOURCES:
        return "chartqa"
    if source in CHARXIV_SOURCES:
        return "charxiv"
    if source in CHARTMUSEUM_SOURCES:
        return "chartmuseum"
    raise ValueError(f"Unknown source: {source}")


def derive_charxiv_inst_cat(answer: str, answer_type: str) -> int:
    """numeric/digits-only → 4, else 2."""
    a = str(answer).strip()
    if answer_type == "numeric":
        return 4
    # digits + dot + percent + comma only?
    if a and re.fullmatch(r"-?[\d,.\s%]+", a):
        return 4
    return 2


MULTICHOICE_INSTRUCTION = "Answer with the value or name of the chosen option."


def render_user_text(question: str, source: str, answer: str, answer_type: str) -> tuple[str, dict]:
    """Return (user_text, extra_meta).

    Source → family → prompt template:
      - chartqa  : `{Q}\\nAnswer the question with a single word.`
      - charxiv  : CharXiv cat 2 (text/yesno/multichoice) or cat 4 (numeric)
      - chartmuseum : ChartMuseum QA_PROMPT (`<think>...</think><answer>X</answer>`)

    Multichoice add-on (letter-only instruction) is applied **only when the
    row is itself answer_type=='multichoice'**. Empirically this is the
    plotly_complex source only; defensive code path covers any future source.
    CharXiv cat 2's own template already handles options, so no extra hint
    needed there.
    """
    fam = family_of(source)
    is_mc = (answer_type == "multichoice")

    if fam == "chartqa":
        # ChartQA never carries multichoice in our corpus.
        return question + CHARTQA_POST, {}

    if fam == "charxiv":
        cat = derive_charxiv_inst_cat(answer, answer_type)
        if cat == 4:
            user = CHARXIV_INST[4].format(question, _charxiv_number_instruction(answer))
        else:
            user = CHARXIV_INST[2].format(question)
        return user, {"_charxiv_inst_cat": cat}

    if fam == "chartmuseum":
        # Multichoice rows: append a verbatim-option instruction inline so the
        # teacher emits the option as it appears on the chart (whether that's
        # a letter, a value, or a label). The ChartMuseum `<answer>...</answer>`
        # publisher template still wraps the response.
        q = f"{question}\n{MULTICHOICE_INSTRUCTION}" if is_mc else question
        meta = {"_is_multichoice": True} if is_mc else {}
        return CHARTMUSEUM_QA_PROMPT.replace("[QUESTION]", q), meta

    raise ValueError(fam)


# ── Relaxed match ──

YES_TOKENS = {"y", "yes", "true", "1", "ye"}
NO_TOKENS = {"n", "no", "false", "0"}


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" .!?,;:")


def _parse_num(s: str):
    s = s.replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


_LETTER_RE = re.compile(r"^[A-Ea-e]$")
_LETTER_EXTRACT_RE = re.compile(r"\b([A-Ea-e])\b")


def relaxed_match(pred: str, gold: str, is_multichoice: bool = False) -> bool:
    if not pred or not gold:
        return False
    p_norm = _norm_text(pred)
    g_norm = _norm_text(gold)
    # Multichoice: letter-only equality (case-insensitive). Tolerate phrases
    # like "B" or "Option B" by extracting a standalone letter.
    if is_multichoice and _LETTER_RE.match(g_norm):
        if _LETTER_RE.match(p_norm):
            return p_norm.upper() == g_norm.upper()
        m = _LETTER_EXTRACT_RE.search(p_norm)
        return bool(m and m.group(1).upper() == g_norm.upper())
    # Yes/No
    if g_norm in YES_TOKENS and p_norm in YES_TOKENS:
        return True
    if g_norm in NO_TOKENS and p_norm in NO_TOKENS:
        return True
    # Numeric
    pn = _parse_num(p_norm)
    gn = _parse_num(g_norm)
    if pn is not None and gn is not None:
        if abs(gn) < 1e-9:
            return abs(pn) < 1e-6
        return abs(pn - gn) / max(1.0, abs(gn)) <= 0.05
    # Text equality / contains
    if p_norm == g_norm:
        return True
    if g_norm and (g_norm in p_norm or p_norm in g_norm):
        return True
    return False


# ── Family target assembly ──

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_teacher_answer(content: str, family: str) -> str:
    """Pull the bare answer string from teacher post-think `content`.

    For ChartMuseum target shape, teacher is expected to emit `<answer>X</answer>`
    in content. For ChartQA/CharXiv, teacher should emit a bare phrase.
    """
    s = (content or "").strip()
    if not s:
        return ""
    if family == "chartmuseum":
        m = ANSWER_TAG_RE.search(s)
        if m:
            return m.group(1).strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in s.splitlines() if l.strip()]
        return lines[-1] if lines else ""
    # ChartQA / CharXiv
    # Strip any trailing punctuation, keep last non-empty line
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def assemble_assistant_text(reasoning: str, family: str, family_answer: str) -> str:
    rsn = (reasoning or "").strip()
    if family == "chartmuseum":
        return f"<think>\n{rsn}\n</think>\n<answer>{family_answer}</answer>"
    return f"<think>\n{rsn}\n</think>\n{family_answer}"


# ── Async distill core ──

async def call_teacher(client, model, sampling, msgs):
    """Issue a chat completion. We deliberately do NOT pass max_tokens — the
    server uses (max_model_len - prompt_tokens) automatically. Hard-coding a
    cap silently truncates long CoTs (16.7% length-cap rate observed at
    8192 in preflight)."""
    return await client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        presence_penalty=sampling.get("presence_penalty", 0.0),
        extra_body={
            "top_k": sampling["top_k"],
            "min_p": sampling["min_p"],
            "repetition_penalty": sampling.get("repetition_penalty", 1.0),
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )


def build_msgs(user_text: str, img_path: str) -> list:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": user_text},
    ]}]


async def distill_one(rec, idx, client, model, sampling):
    sample_id = f"{rec.get('source','?')}_{idx}"
    src = rec.get("source", "?")
    q = rec["question"]
    gold = str(rec.get("answer", "")).strip()
    ans_type = rec.get("answer_type", "") or ""
    img_path = rec.get("image_path", "")

    if not img_path or not os.path.exists(img_path):
        return None, dict(rec, sample_id=sample_id, error="image_missing")

    family = family_of(src)
    user_text, extra = render_user_text(q, src, gold, ans_type)
    is_mc = bool(extra.get("_is_multichoice")) or ans_type == "multichoice"
    msgs = build_msgs(user_text, img_path)

    t0 = time.time()
    try:
        r = await call_teacher(client, model, sampling, msgs)
    except Exception as e:
        return None, dict(rec, sample_id=sample_id, error=f"http:{type(e).__name__}:{str(e)[:200]}")
    dt = time.time() - t0

    msg = r.choices[0].message
    rsn = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or ""
    )
    ct = msg.content or ""
    fr = r.choices[0].finish_reason
    usage = r.usage

    teacher_pred = extract_teacher_answer(ct, family)
    teacher_match = relaxed_match(teacher_pred, gold, is_multichoice=is_mc)

    out_meta = {
        "_format_family": family,
        "_teacher_reasoning": rsn,
        "_teacher_content": ct,
        "_teacher_pred": teacher_pred,
        "_teacher_match": teacher_match,
        "_teacher_finish_reason": fr,
        "_teacher_reasoning_len": len(rsn),
        "_teacher_content_len": len(ct),
        "_teacher_completion_tok": usage.completion_tokens,
        "_teacher_dt": round(dt, 2),
        "user_text": user_text,
        **extra,
    }

    # Drop only on hard failures: empty teacher output. Length-cap and content
    # mismatches are kept — format conversion + scoring happens downstream.
    if not (rsn or ct):
        return None, dict(rec, sample_id=sample_id, _drop_reasons="empty_output", **out_meta)

    # Keep raw teacher output; format wrap is a downstream step.
    out = dict(rec)
    out["sample_id"] = sample_id
    out.update(out_meta)
    return out, None


# ── Orchestration ──

async def main_async(args):
    sampling = SAMPLING_PARAMS["27b"]["thinking"]
    print(f"[config] sampling={sampling}")
    print(f"[config] hosts={args.hosts.split(',')} conc/host={args.max_concurrent_per_host}")

    # Load input — either fresh source corpus or a failed JSONL for retry.
    rows = []
    if args.retry_failed:
        # In retry mode, build a sample_id -> original-rec lookup from --input
        # so we can recover full source fields even for HTTP-error rows that
        # were written with slim metadata only.
        src_lookup = {}
        with open(args.input) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sid = f"{d.get('source','?')}_{i}"
                src_lookup[sid] = (i, d)
        print(f"[retry] indexed {len(src_lookup)} input rows for sample_id lookup")

        with open(args.retry_failed) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sid = d.get("sample_id") or ""
                if sid in src_lookup:
                    idx, rec = src_lookup[sid]
                else:
                    # fallback: strip failure metadata, hope question/image are there
                    rec = {k: v for k, v in d.items()
                           if not k.startswith("_") and k not in {"sample_id", "error"}}
                    try:
                        idx = int(sid.rsplit("_", 1)[1])
                    except (IndexError, ValueError):
                        idx = -1
                rows.append((idx, rec))
        print(f"[retry] {len(rows)} rows from {args.retry_failed}")
    else:
        with open(args.input) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    rows.append((i, json.loads(line)))
        print(f"[input] {len(rows)} rows from {args.input}")

    # Resume — read existing output
    done = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        done.add(d.get("sample_id"))
                    except Exception:
                        pass
    print(f"[resume] {len(done)} already completed")

    # Filter source subset if requested
    if args.sources:
        wanted = set(args.sources.split(","))
        rows = [(i, r) for i, r in rows if r.get("source") in wanted]
        print(f"[filter] sources={wanted} → {len(rows)} rows")

    # Limit if --limit
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[limit] {len(rows)} rows")

    # Skip done
    todo = [(i, r) for i, r in rows if f"{r.get('source','?')}_{i}" not in done]
    print(f"[todo] {len(todo)} rows after resume")

    # ── Dynamic multi-host pool (queue + per-host workers + idle-probe) ──
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    host_states = {
        h: {
            "client": AsyncOpenAI(base_url=h, api_key="dummy", timeout=args.request_timeout),
            "alive": True,
            "calls": 0,
            "fails": 0,
        }
        for h in hosts
    }

    print(f"[probe] initial host check ({len(hosts)} hosts)")
    init_probes = await asyncio.gather(*[_probe_host(h, st) for h, st in host_states.items()])
    for h, alive in zip(hosts, init_probes):
        print(f"  {h}: {'ALIVE' if alive else 'DEAD (will idle-probe)'}")
    if not any(init_probes):
        print("[fatal] no hosts initially alive — workers will idle-probe but distill cannot progress until at least one comes up.")

    # Shared queue
    queue: asyncio.Queue = asyncio.Queue()
    for idx, rec in todo:
        queue.put_nowait((idx, rec))
    total_items = queue.qsize()

    failed_path = Path(args.failed)
    out_f = open(out_path, "a", buffering=1)  # line-buffered
    fail_f = open(failed_path, "a", buffering=1)
    out_lock = asyncio.Lock()
    fail_lock = asyncio.Lock()

    stats = Counter()
    t_start = time.time()

    async def write_kept(ok):
        async with out_lock:
            out_f.write(json.dumps(ok, ensure_ascii=False) + "\n")
        stats["kept"] += 1
        stats[f"kept_{ok.get('_format_family','?')}"] += 1

    async def write_failed(fail):
        async with fail_lock:
            fail_f.write(json.dumps(fail, ensure_ascii=False) + "\n")
        stats["dropped"] += 1
        err = fail.get("error", "")
        if err:
            stats[f"err_{err.split(':')[0]}"] += 1

    def _maybe_log_progress():
        total = stats["kept"] + stats["dropped"]
        if total > 0 and total % 10 == 0:
            elapsed = time.time() - t_start
            rate = total / max(elapsed, 1e-3)
            remain = total_items - total
            alive_hosts = sum(1 for s in host_states.values() if s["alive"])
            print(f"[{total:>5}/{total_items}] kept={stats['kept']} drop={stats['dropped']} "
                  f"alive={alive_hosts}/{len(hosts)} rate={rate:.2f}/s "
                  f"eta={remain/max(rate,1e-3)/60:.1f}min")

    async def worker(host_url: str):
        state = host_states[host_url]
        # Workers do NOT self-exit on empty queue — main coroutine cancels
        # them after queue.join() returns. This avoids races where a
        # requeued item lands in an empty queue after the last live worker
        # has already exited.
        while True:
            # Idle-probe loop when host is dead.
            if not state["alive"]:
                await asyncio.sleep(args.health_interval)
                await _probe_host(host_url, state)
                continue
            # Pull next item — bounded wait so cancellation can land.
            try:
                idx, rec = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                ok, fail = await distill_one(
                    rec, idx, state["client"], args.model_name, sampling
                )
                state["calls"] += 1
                if ok is not None:
                    await write_kept(ok)
                else:
                    err = fail.get("error", "")
                    if _is_connect_error(err):
                        # Host-level failure — requeue and mark dead.
                        print(f"[host] {host_url} → DEAD ({err.split(':',2)[1] if ':' in err else err})")
                        state["alive"] = False
                        state["fails"] += 1
                        await queue.put((idx, rec))
                    else:
                        await write_failed(fail)
            finally:
                queue.task_done()
                _maybe_log_progress()

    workers: list[asyncio.Task] = []
    for h in hosts:
        for _ in range(args.max_concurrent_per_host):
            workers.append(asyncio.create_task(worker(h)))

    try:
        await asyncio.wait_for(queue.join(), timeout=None)
    finally:
        # Workers exit naturally when queue is empty + their host's checks settle.
        for w in workers:
            if not w.done():
                w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.time() - t_start
    print(f"\n=== Distill complete ===")
    print(f"elapsed: {elapsed/60:.1f}min")
    print("[per-host stats]")
    for h, st in host_states.items():
        print(f"  {h:38s} alive={st['alive']} calls={st['calls']} fails={st['fails']}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28} {v}")

    out_f.close(); fail_f.close()


_CONNECT_ERROR_NAMES = (
    "APIConnectionError",
    "ConnectError",
    "ConnectionError",
    "ConnectionRefusedError",
    "RemoteProtocolError",
    "ReadError",
    "TimeoutException",  # only for ConnectTimeout subset; broad timeout still requeues
    # Host serves the wrong model (e.g., 147 swapped to GLM on schedule).
    # Treating 404 as host-level lets the queue route around the bad host
    # instead of burning 9 K requests against a dead-end endpoint at 30 ms each.
    "NotFoundError",
)


def _is_connect_error(err_str: str) -> bool:
    """Detect host-level (vs request-level) failures from distill_one's
    error string `http:<ExceptionClass>:<msg>`."""
    if not err_str.startswith("http:"):
        return False
    cls = err_str.split(":", 2)[1] if err_str.count(":") >= 2 else ""
    return any(name in cls for name in _CONNECT_ERROR_NAMES)


async def _probe_host(url: str, state: dict, timeout: float = 4.0) -> bool:
    """One-shot /v1/models ping. Updates state['alive'] in-place; logs on
    transition (alive→dead or dead→alive)."""
    was_alive = state["alive"]
    try:
        await asyncio.wait_for(state["client"].models.list(), timeout=timeout)
        if not was_alive:
            print(f"[host] {url} → BACK ALIVE")
        state["alive"] = True
        return True
    except Exception as e:
        if was_alive:
            print(f"[host] {url} → DEAD on probe ({type(e).__name__})")
        state["alive"] = False
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/sft_v8_improved_v2.jsonl")
    p.add_argument("--output", default="data/sft_v8_distill_v2.jsonl")
    p.add_argument("--failed", default="data/sft_v8_distill_v2_failed.jsonl")
    p.add_argument("--hosts", default="http://localhost:9101/v1,http://localhost:9102/v1,http://localhost:9103/v1")
    p.add_argument("--max_concurrent_per_host", type=int, default=6)
    p.add_argument("--model_name", default="Qwen3.6-27B",
                   help="Model name string sent in the OpenAI request body. "
                        "Use 'Qwen3.5-397B-A17B-FP8' for the external 397B teacher.")
    p.add_argument("--health_interval", type=float, default=1200.0,
                   help="Seconds a DEAD host is sidelined before the next probe. "
                        "Default 1200s (20 min) matches the upstream auto-restart cycle.")
    p.add_argument("--sources", default="", help="comma-separated source filter (default: all)")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--retry_failed", "--retry-failed", dest="retry_failed", default="",
                   help="Path to a failed JSONL (rerun rows from there)")
    p.add_argument("--request_timeout", type=float, default=900.0,
                   help="Per-request HTTP timeout (s). Default 900.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
