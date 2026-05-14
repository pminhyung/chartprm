"""Teacher distillation: 397B VLM generates high-quality SFT training samples.

Eval-identical prompt format (byte-for-byte match with eval_multi_server.py).
Filters by gold-answer match. Assembles full assistant text as
`<think>\\n{reasoning_content}\\n</think>\\n{content}</answer>` and stores it
in the `assistant_text` field so train_sft.py can use it verbatim (no
re-wrapping, preserving teacher's exact output format).

Usage:
    python scripts/teacher_distill.py \\
        --source data/distill_source.jsonl \\
        --output data/sft_hq.jsonl \\
        --hosts http://localhost:9200/v1,http://localhost:9201/v1 \\
        --max_concurrent_per_host 6
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
from collections import Counter

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chartvr.llm_client import MultiHostClient
from chartvr.prompts import EVAL_SYSTEM_PROMPT
from chartvr.extraction import extract_answer, relaxed_accuracy
from chartvr.config import SAMPLING_PARAMS


def encode_image(image_path: str) -> str:
    """Eval-identical image encoding (eval_multi_server.py L113-120)."""
    img = Image.open(image_path).convert("RGB")
    max_side = max(img.size)
    if max_side > 1024:
        scale = 1024 / max_side
        img = img.resize(
            (int(img.size[0] * scale), int(img.size[1] * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def dedup_key(sample: dict) -> str:
    img = sample.get("image_path", "")
    q = (sample.get("question") or "")[:80]
    return f"{img}|{q}"


def assemble_assistant_text(reasoning_content: str, content: str) -> str:
    """Reassemble full assistant text `<think>\\n{r}\\n</think>\\n{c}`.

    vLLM `--reasoning-parser qwen3` strips the `<think>/</think>` tags from
    reasoning_content. We re-add them here so training data carries the same
    raw format the model emits during eval. Also ensures the final `</answer>`
    closing tag is present (stop=["</answer>"] truncates it from content).
    """
    r = (reasoning_content or "").strip()
    c = (content or "").strip()
    text = f"<think>\n{r}\n</think>\n{c}"
    # Ensure `</answer>` closing tag is present
    if "<answer>" in text and "</answer>" not in text:
        text = text + "</answer>"
    return text


async def distill_one(client, sample, lock, out_f, written_keys, stats, sp):
    key = dedup_key(sample)
    if key in written_keys:
        return
    img_path = sample.get("image_path", "")
    if not img_path or not os.path.exists(img_path):
        async with lock:
            stats["no_image"] += 1
        return
    try:
        # PIL encoding is ~50-100ms per image — must be off-thread so the
        # event loop can keep pumping network I/O for other in-flight requests.
        img_b64 = await asyncio.to_thread(encode_image, img_path)
    except Exception:
        async with lock:
            stats["image_error"] += 1
        return

    # Eval-identical messages
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Look at this chart and answer the question.\n\nQuestion: {sample['question']}"},
        ]},
    ]

    try:
        resp = await client.chat(
            messages=messages,
            # max_tokens capped at 8192 to prevent runaway thinking chains
            # (we've observed single samples with 48k+ reasoning chars).
            # Still well under 9201's 65k max_model_len given ~2k prompt.
            max_tokens=8192,
            temperature=sp["temperature"],
            top_p=sp["top_p"],
            presence_penalty=sp.get("presence_penalty", 0.0),
            stop=["</answer>"],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": sp["top_k"],
                "min_p": sp["min_p"],
            },
        )
    except Exception as e:
        async with lock:
            stats["api_error"] += 1
            if stats["api_error"] <= 3:
                print(f"\n[api_error #{stats['api_error']}] {type(e).__name__}: {str(e)[:300]}", flush=True)
        return

    msg = resp.choices[0].message
    content = msg.content or ""
    raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
    reasoning_content = raw.get("reasoning_content", "") or raw.get("reasoning", "") or ""
    finish_reason = resp.choices[0].finish_reason or ""

    # Gate 1: non-empty reasoning
    if len(reasoning_content.strip()) < 5:
        async with lock:
            stats["empty_reasoning"] += 1
        return

    # Gate 1b: non-empty content (required for `<answer>` extraction)
    if not content.strip():
        async with lock:
            stats["empty_content"] += 1
        return

    # Gate 2: gold match
    pred = extract_answer(content)
    if not pred:
        async with lock:
            stats["no_pred"] += 1
        return

    gold = str(sample["answer"]).strip()
    if relaxed_accuracy(pred, gold) < 1.0:
        async with lock:
            stats["wrong_answer"] += 1
        return

    # Assemble full assistant text (teacher-native format, no re-wrapping downstream)
    assistant_text = assemble_assistant_text(reasoning_content, content)

    out = {
        "question": sample["question"],
        "answer": gold,
        "answer_type": sample.get("answer_type", ""),
        "assistant_text": assistant_text,
        "image_path": img_path,
        "csv_path": sample.get("csv_path", ""),
        "source": sample.get("source", ""),
        "chart_slug": sample.get("chart_slug", ""),
        "origin": sample.get("origin", ""),
        "teacher_pred": pred,
        "teacher_reasoning_len": len(reasoning_content),
        "teacher_content_len": len(content),
        "teacher_finish_reason": finish_reason,
        "distilled_from": "qwen3.5-397b-thinking",
    }
    async with lock:
        if key not in written_keys:
            written_keys.add(key)
            out_f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
            out_f.flush()
            stats["ok"] += 1


async def run_distill(args):
    with open(args.source) as f:
        samples = [json.loads(l) for l in f if l.strip()]

    # Optional filters
    if args.keep_sources:
        keep = set(args.keep_sources.split(","))
        samples = [s for s in samples if s.get("source", "") in keep]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    # Resume: load already-written keys
    written_keys: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    written_keys.add(dedup_key(rec))
                except Exception:
                    pass

    pending = [s for s in samples if dedup_key(s) not in written_keys]
    print(f"Source: {len(samples)} total, {len(written_keys)} done, {len(pending)} pending")

    hosts = args.hosts.split(",") if args.hosts else None
    client = MultiHostClient(
        hosts=hosts,
        max_concurrent_per_host=args.max_concurrent_per_host,
        dynamic=True,
        timeout=args.timeout,
    )
    sp = SAMPLING_PARAMS["397b"]["thinking"]
    print(f"Live hosts ({client.n_hosts}): {client._live_hosts}")
    print(f"Max concurrent: {client.max_concurrent}")
    print(f"Sampling: {sp}")
    print("max_tokens: omitted (vLLM auto = max_model_len - prompt_tokens)")

    lock = asyncio.Lock()
    stats: Counter = Counter()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a")

    # Worker-pool pattern: spawn N concurrent workers pulling from a queue.
    # Avoids creating 40K+ coroutines upfront (which would block the event
    # loop with per-coroutine startup work before any await can yield).
    queue: asyncio.Queue = asyncio.Queue()
    for s in pending:
        queue.put_nowait(s)

    pbar = tqdm(total=len(pending), desc="Distill", unit="sample")

    async def worker():
        while True:
            try:
                s = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await distill_one(client, s, lock, out_f, written_keys, stats, sp)
            except Exception:
                stats["task_exception"] += 1
            finally:
                queue.task_done()
                pbar.update(1)
                pbar.set_postfix(**{k: v for k, v in stats.items() if v > 0})

    # Number of workers = aggregate concurrency across all live hosts.
    # Client also has its own per-host semaphore, so this is just an upper
    # bound on how many distill_one invocations are alive at once.
    n_workers = max(client.max_concurrent, 1)
    workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
    await asyncio.gather(*workers)
    pbar.close()
    out_f.close()

    total = sum(stats.values())
    print(f"\nDistillation complete: {total} attempts")
    for k in sorted(stats.keys()):
        v = stats[k]
        pct = v * 100 // max(total, 1)
        print(f"  {k}: {v} ({pct}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hosts", default="",
                    help="Comma-separated host URLs. Empty = dynamic pool "
                         "(STABLE + DYNAMIC from config, runtime Qwen verified).")
    ap.add_argument("--max_concurrent_per_host", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="Per-request timeout in seconds (thinking mode can take "
                         "5+ min on long outputs). Default 900s.")
    ap.add_argument("--keep_sources", default="",
                    help="Comma-separated source filter (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit pending samples (for pilot runs)")
    args = ap.parse_args()
    asyncio.run(run_distill(args))


if __name__ == "__main__":
    main()
