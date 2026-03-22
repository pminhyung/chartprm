#!/usr/bin/env python3
"""
Benchmark inference script for ChartVCR.

Runs 5 vLLM server instances (2 GPUs each, ports 9101-9105) with
round-robin load balancing and 4 concurrent requests per server.

Usage:
    python scripts/run_benchmark_inference.py \
        --model-path models/qwen3vl-8b-thinking \
        --model-id qwen3vl-8b-thinking \
        --output-dir results/v4_redo/qwen3vl_zeroshot \
        --reasoning   # for Qwen3 thinking model
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
VLLM_PYTHON = "/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python"
LOG_DIR = Path("/tmp/vllm_logs_benchmark")

PORTS = [9101, 9102, 9103, 9104, 9105]
GPU_PAIRS = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]

N_INSTANCES = 5
GPUS_PER_INSTANCE = 2
CONCURRENCY_PER_SERVER = 4  # 4 × 5 = 20 total concurrent

MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85

SYSTEM_PROMPT = (
    "You are an expert chart analyst. After your reasoning, put your final answer "
    "inside <answer></answer> tags with ONLY the core value or keyword. "
    "Example: <answer>42</answer>"
)

EVAL_PROMPT_TEMPLATE = "Look at this chart and answer the question.\n\nQuestion: {question}"

BENCHMARKS = [
    {
        "name": "chartqa_human",
        "json_path": "data/chartqa/test/test_human.json",
        "image_dir": "data/chartqa/test/png",
    },
    {
        "name": "chartqa_augmented",
        "json_path": "data/chartqa/test/test_augmented.json",
        "image_dir": "data/chartqa/test/png",
    },
    {
        "name": "charxiv_reasoning",
        "json_path": "data/charxiv/val_reasoning.json",
        "image_dir": "data/charxiv/images",
    },
    {
        "name": "chartqa_pro",
        "json_path": "data/chartqa_pro/test.json",
        "image_dir": "data/chartqa_pro/images",
    },
]


# ─── Logging ──────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ─── Image Handling ───────────────────────────────────────────────────────────
def load_and_encode_image(image_path: str) -> str:
    """Load image, resize max side to 1024, return base64 PNG string."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        max_side = max(img.width, img.height)
        if max_side > 1024:
            scale = 1024 / max_side
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Answer Extraction ────────────────────────────────────────────────────────
def extract_answer(content: str) -> str:
    """Extract answer from content field (NOT reasoning).

    Supports <answer>...</answer> and [answer]...[/answer] tags.
    Uses LAST occurrence.
    """
    if not content:
        return ""

    # Try <answer>...</answer> tags (last occurrence)
    matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Try [answer]...[/answer] tags (last occurrence)
    matches = re.findall(r"\[answer\](.*?)\[/answer\]", content, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    return ""


# ─── Accuracy ─────────────────────────────────────────────────────────────────
def compute_accuracy(predicted: str, gold: str) -> float:
    """Relaxed accuracy: 5% tolerance for numeric, exact match for text."""
    pred = predicted.strip().lower()
    gold_str = str(gold).strip().lower()

    if pred == gold_str:
        return 1.0

    # Try numeric comparison with 5% tolerance
    try:
        pred_num = float(re.sub(r"[,%$]", "", pred))
        gold_num = float(re.sub(r"[,%$]", "", gold_str))
        if gold_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        if abs(pred_num - gold_num) / abs(gold_num) <= 0.05:
            return 1.0
    except (ValueError, ZeroDivisionError):
        pass

    return 0.0


# ─── vLLM Server Lifecycle ────────────────────────────────────────────────────
def start_vllm_servers(model_path: str, reasoning: bool) -> list[tuple]:
    """Start 5 vLLM server instances. Returns list of (port, proc) tuples."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []

    for i, (port, (gpu_a, gpu_b)) in enumerate(zip(PORTS, GPU_PAIRS)):
        gpu_str = f"{gpu_a},{gpu_b}"
        log_path = LOG_DIR / f"vllm_{port}.log"

        cmd = [
            VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--tensor-parallel-size", str(GPUS_PER_INSTANCE),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--dtype", "half",
            "--max-num-seqs", "16",
            "--trust-remote-code",
        ]
        if reasoning:
            cmd += ["--reasoning-parser", "deepseek_r1"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str

        log(f"  Starting instance {i}: GPU={gpu_str} port={port}")
        with open(log_path, "w") as lf:
            p = subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=lf,
                start_new_session=True,
            )
        procs.append((port, p))
        log(f"    PID={p.pid}")

    return procs


def wait_for_server(port: int, model_path: str, max_wait: int = 600) -> bool:
    """Wait until vLLM server is ready via /v1/models + warmup."""
    url_models = f"http://localhost:{port}/v1/models"
    url_chat = f"http://localhost:{port}/v1/chat/completions"

    deadline = time.time() + max_wait
    log(f"  Waiting for port {port} /v1/models...")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url_models, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("data"):
                    log(f"  Port {port}: models endpoint ready.")
                    break
        except Exception:
            pass
        time.sleep(3)
    else:
        log(f"ERROR: Port {port} not ready after {max_wait}s")
        return False

    # Warmup inference with text-only message (no image to keep it fast)
    payload = json.dumps({
        "model": model_path,
        "messages": [{"role": "user", "content": "1+1="}],
        "max_tokens": 4,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        url_chat,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    warmup_deadline = time.time() + 120
    log(f"  Port {port}: sending warmup inference...")
    while time.time() < warmup_deadline:
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
                if resp.get("choices"):
                    log(f"  Port {port}: warmup OK.")
                    return True
        except Exception as e:
            log(f"  Port {port}: warmup attempt: {e}")
        time.sleep(3)

    log(f"WARN: Port {port} warmup timed out")
    return False


def wait_for_all_servers(procs: list[tuple], model_path: str) -> list[tuple]:
    """Wait for all servers; return only healthy (port, proc) tuples."""
    healthy = []
    for port, p in procs:
        ok = wait_for_server(port, model_path)
        if ok:
            healthy.append((port, p))
        else:
            log(f"WARN: Port {port} failed — excluding from pool")
            try:
                pgid = os.getpgid(p.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
    return healthy


def stop_vllm_servers(procs: list[tuple]):
    """Stop all vLLM servers via PID-based kill (SIGTERM -> wait -> SIGKILL)."""
    if not procs:
        return
    log(f"Stopping {len(procs)} vLLM instances...")

    still_running = []
    for port, p in procs:
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)
            still_running.append((port, p, pgid))
            log(f"  SIGTERM -> port {port} (pgid {pgid})")
        except (ProcessLookupError, PermissionError):
            pass

    # Wait up to 60s for graceful shutdown
    deadline = time.time() + 60
    while still_running and time.time() < deadline:
        time.sleep(2)
        still_running = [(port, p, pgid) for port, p, pgid in still_running
                         if p.poll() is None]

    # SIGKILL survivors
    for port, p, pgid in still_running:
        try:
            os.killpg(pgid, signal.SIGKILL)
            log(f"  SIGKILL -> port {port} (pgid {pgid})")
        except (ProcessLookupError, PermissionError):
            pass

    # Clean orphaned children of OUR processes only
    for _, p in procs:
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(p.pid)], capture_output=True, text=True)
            for child_pid in result.stdout.strip().split("\n"):
                if child_pid.strip():
                    try:
                        os.kill(int(child_pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
        except Exception:
            pass

    time.sleep(2)
    log("Servers stopped.")


def verify_gpus_freed():
    """Print GPU memory status for GPUs 2-11 after stopping servers."""
    log("Verifying GPU memory freed...")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        parts = line.split(", ")
        if len(parts) == 2:
            idx, used = parts
            if int(idx) in range(2, 12):
                log(f"  GPU {idx}: {used} MiB used")


# ─── Inference ────────────────────────────────────────────────────────────────
def load_benchmark(benchmark: dict, project_dir: Path) -> list[dict]:
    """Load benchmark JSON and build per-sample dicts with absolute image paths."""
    json_path = project_dir / benchmark["json_path"]
    image_dir = project_dir / benchmark["image_dir"]

    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for i, item in enumerate(data):
        samples.append({
            "sample_id": i,
            "question": item["query"],
            "gold_answer": str(item["label"]),
            "image_path": str(image_dir / item["imgname"]),
        })
    return samples


def load_completed_set(output_path: Path) -> set[int]:
    """Load sample_ids already completed successfully (no error, non-empty answer)."""
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("error") is None and r.get("predicted_answer", "").strip():
                    completed.add(r["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_jsonl(path: Path, record: dict, written_ids: set):
    """Append a JSON record to file, skipping duplicates."""
    sid = record.get("sample_id")
    if sid in written_ids:
        return
    written_ids.add(sid)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()


class RoundRobinLB:
    """Round-robin load balancer with per-server semaphores."""

    def __init__(self, endpoints: list[str], concurrency: int):
        self.endpoints = endpoints
        self._counter = 0
        self._sems = {ep: asyncio.Semaphore(concurrency) for ep in endpoints}

    def next_endpoint(self) -> str:
        ep = self.endpoints[self._counter % len(self.endpoints)]
        self._counter += 1
        return ep


async def call_vllm(
    session,
    endpoint: str,
    model_path: str,
    image_b64: str,
    question: str,
    reasoning: bool,
) -> tuple[str, str, str | None]:
    """Call vLLM server via aiohttp. Returns (content, reasoning_content, error).

    For reasoning models: sets chat_template_kwargs enable_thinking=True so
    the server routes thinking tokens to reasoning_content and final answer
    to content (via deepseek_r1 reasoning parser).
    """
    import aiohttp

    url = f"{endpoint}/chat/completions"
    prompt = EVAL_PROMPT_TEMPLATE.format(question=question)

    payload = {
        "model": model_path,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    # Enable thinking mode for reasoning models: routing thinking trace to
    # reasoning_content and final answer to content
    if reasoning:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                return "", "", f"HTTP {resp.status}: {text[:200]}"
            data = await resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning") or msg.get("reasoning_content") or ""

            # Fallback: extract reasoning from <think> tags in content if parser
            # didn't separate them
            if not reasoning_content and "<think>" in content:
                m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if m:
                    reasoning_content = m.group(1).strip()
                    # Remove everything up to and including </think>
                    close_idx = content.rfind("</think>")
                    if close_idx >= 0:
                        content = content[close_idx + len("</think>"):].strip()

            return content, reasoning_content, None
    except Exception as e:
        return "", "", str(e)[:200]


async def process_sample(
    sample: dict,
    lb: RoundRobinLB,
    model_path: str,
    output_path: Path,
    session,
    pbar,
    lock: asyncio.Lock,
    stats: dict,
    reasoning: bool,
    written_ids: set = None,
):
    """Process a single sample: encode image, call API, compute accuracy, write result."""
    sample_id = sample["sample_id"]
    question = sample["question"]
    gold_answer = sample["gold_answer"]
    image_path = sample["image_path"]

    # Load and encode image (CPU-bound, offload to executor)
    try:
        loop = asyncio.get_event_loop()
        image_b64 = await loop.run_in_executor(
            None, load_and_encode_image, image_path
        )
    except Exception as e:
        record = {
            "sample_id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "image_path": image_path,
            "predicted_answer": "",
            "content": "",
            "reasoning_content": "",
            "accuracy": 0.0,
            "error": f"Image load error: {str(e)[:100]}",
        }
        async with lock:
            stats["error"] += 1
            append_jsonl(output_path, record, written_ids)
            pbar.update(1)
            pbar.set_postfix(ok=stats["ok"], err=stats["error"], refresh=False)
        return

    # Pick endpoint and acquire semaphore
    endpoint = lb.next_endpoint()
    async with lb._sems[endpoint]:
        content, reasoning_content, error = await call_vllm(
            session, endpoint, model_path, image_b64, question, reasoning
        )

    # Retry once on error with a different endpoint
    if error:
        endpoint2 = lb.next_endpoint()
        async with lb._sems[endpoint2]:
            content, reasoning_content, error = await call_vllm(
                session, endpoint2, model_path, image_b64, question, reasoning
            )

    if error:
        record = {
            "sample_id": sample_id,
            "question": question,
            "gold_answer": gold_answer,
            "image_path": image_path,
            "predicted_answer": "",
            "content": content,
            "reasoning_content": reasoning_content,
            "accuracy": 0.0,
            "error": error,
        }
        async with lock:
            stats["error"] += 1
            append_jsonl(output_path, record, written_ids)
            pbar.update(1)
            pbar.set_postfix(ok=stats["ok"], err=stats["error"], refresh=False)
        return

    predicted_answer = extract_answer(content)
    accuracy = compute_accuracy(predicted_answer, gold_answer)

    record = {
        "sample_id": sample_id,
        "question": question,
        "gold_answer": gold_answer,
        "image_path": image_path,
        "predicted_answer": predicted_answer,
        "content": content,
        "reasoning_content": reasoning_content,
        "accuracy": accuracy,
        "error": None,
    }

    async with lock:
        stats["ok"] += 1
        stats["accuracy_sum"] += accuracy
        append_jsonl(output_path, record, written_ids)
        pbar.update(1)
        pbar.set_postfix(
            ok=stats["ok"],
            err=stats["error"],
            acc=f"{stats['accuracy_sum'] / max(stats['ok'], 1):.3f}",
            refresh=False,
        )


async def run_benchmark_async(
    samples: list[dict],
    model_path: str,
    endpoints: list[str],
    output_path: Path,
    benchmark_name: str,
    reasoning: bool,
) -> dict:
    """Run inference on all pending samples. Returns accuracy summary dict."""
    import aiohttp

    # Resume: load completed sample_ids
    completed = load_completed_set(output_path)
    pending = [s for s in samples if s["sample_id"] not in completed]

    log(f"  {benchmark_name}: {len(samples)} total, {len(completed)} done, "
        f"{len(pending)} pending")

    if not pending:
        log(f"  {benchmark_name}: already complete, skipping.")
        return compute_accuracy_from_file(output_path)

    lb = RoundRobinLB(endpoints, CONCURRENCY_PER_SERVER)
    lock = asyncio.Lock()
    stats = {"ok": 0, "error": 0, "accuracy_sum": 0.0}
    written_ids = set(completed)  # Track all written IDs to prevent duplicates

    pbar = tqdm(
        total=len(samples),
        initial=len(completed),
        desc=benchmark_name,
        unit="req",
        ncols=100,
    )

    async with aiohttp.ClientSession(
        headers={"Authorization": "Bearer EMPTY"}
    ) as session:
        tasks = [
            process_sample(
                s, lb, model_path, output_path,
                session, pbar, lock, stats, reasoning,
                written_ids=written_ids,
            )
            for s in pending
        ]
        await asyncio.gather(*tasks)

    pbar.close()

    result = compute_accuracy_from_file(output_path)
    log(f"  {benchmark_name}: done. Accuracy={result['accuracy']:.4f} "
        f"({result['n_correct']:.0f}/{result['n_total']}), errors={stats['error']}")
    return result


def compute_accuracy_from_file(output_path: Path) -> dict:
    """Read output JSONL and compute overall accuracy from successful records (deduped)."""
    if not output_path.exists():
        return {"accuracy": 0.0, "n_correct": 0, "n_total": 0}
    seen_ids = set()
    n_total = 0
    n_correct = 0.0
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sid = r.get("sample_id")
                if sid in seen_ids:
                    continue  # skip duplicate
                seen_ids.add(sid)
                if r.get("error") is None:
                    n_total += 1
                    n_correct += float(r.get("accuracy", 0.0))
            except json.JSONDecodeError:
                continue
    acc = n_correct / n_total if n_total > 0 else 0.0
    return {"accuracy": acc, "n_correct": n_correct, "n_total": n_total}


def run_preflight(
    samples: list[dict],
    model_path: str,
    endpoints: list[str],
    output_dir: Path,
    reasoning: bool,
    n_samples: int = 20,
) -> bool:
    """Run preflight on first 20 samples with production concurrency. Returns pass/fail."""
    log(f"Running preflight on {n_samples} samples (production concurrency)...")
    preflight_samples = samples[:n_samples]
    preflight_out = output_dir / "_preflight_run.jsonl"
    preflight_out.unlink(missing_ok=True)

    asyncio.run(
        run_benchmark_async(
            preflight_samples, model_path, endpoints,
            preflight_out, "preflight", reasoning
        )
    )

    if not preflight_out.exists():
        log("PREFLIGHT FAILED: no output file")
        return False

    lines = []
    with open(preflight_out) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    n_lines = len(lines)
    if n_lines == 0:
        log("PREFLIGHT FAILED: 0 output lines")
        return False

    n_errors = sum(1 for r in lines if r.get("error"))
    n_answered = sum(1 for r in lines if r.get("predicted_answer", "").strip())

    error_rate = n_errors / n_lines
    answer_rate = n_answered / n_lines

    log(f"Preflight results: {n_lines}/{n_samples} lines, "
        f"error_rate={error_rate:.2f}, answer_rate={answer_rate:.2f}")

    if error_rate >= 0.10:
        log(f"PREFLIGHT FAILED: error rate {error_rate:.2f} >= 0.10")
        return False
    if answer_rate < 0.80:
        log(f"PREFLIGHT FAILED: answer rate {answer_rate:.2f} < 0.80")
        return False

    log("PREFLIGHT PASSED.")
    preflight_out.unlink(missing_ok=True)
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark inference across 4 chart benchmarks")
    parser.add_argument("--model-path", required=True,
                        help="Path to model (absolute or relative to project dir)")
    parser.add_argument("--model-id", required=True,
                        help="Short model ID for logging")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output JSONL files")
    parser.add_argument("--reasoning", action="store_true",
                        help=(
                            "Enable reasoning: starts server with "
                            "--enable-reasoning --reasoning-parser deepseek_r1, "
                            "sends chat_template_kwargs enable_thinking=True"
                        ))
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight run (not recommended)")
    parser.add_argument("--project-dir", default="/ex_disk2/mhpark/poc/chartvr",
                        help="Project root directory")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    if not Path(model_path).is_absolute():
        model_path = str(project_dir / model_path)

    log("=" * 60)
    log(f"  ChartVCR Benchmark Inference")
    log(f"  Model:     {args.model_id}")
    log(f"  Path:      {model_path}")
    log(f"  Output:    {output_dir}")
    log(f"  Reasoning: {args.reasoning}")
    log(f"  GPUs:      {[g for pair in GPU_PAIRS for g in pair]}")
    log("=" * 60)

    # Start servers
    log("\nStarting 5 vLLM instances...")
    procs = start_vllm_servers(model_path, args.reasoning)

    # Wait for healthy servers
    log("Waiting for servers to be ready...")
    healthy_procs = wait_for_all_servers(procs, model_path)

    if not healthy_procs:
        log("FATAL: No healthy servers. Aborting.")
        stop_vllm_servers(procs)
        sys.exit(1)

    endpoints = [f"http://localhost:{port}/v1" for port, _ in healthy_procs]
    log(f"{len(healthy_procs)}/{N_INSTANCES} servers healthy.")
    log(f"Endpoints: {endpoints}")

    try:
        # Load first benchmark for preflight
        first_bm = BENCHMARKS[0]
        first_samples = load_benchmark(first_bm, project_dir)

        # Preflight
        if not args.skip_preflight:
            ok = run_preflight(
                first_samples, model_path, endpoints, output_dir, args.reasoning
            )
            if not ok:
                log("FATAL: Preflight failed. Stopping servers.")
                stop_vllm_servers(healthy_procs)
                sys.exit(1)

        # Run all 4 benchmarks
        summary = {}
        for bm in BENCHMARKS:
            log(f"\n{'=' * 40}")
            log(f"  Benchmark: {bm['name']}")
            log(f"{'=' * 40}")

            samples = load_benchmark(bm, project_dir)
            output_path = output_dir / f"{bm['name']}.jsonl"

            result = asyncio.run(
                run_benchmark_async(
                    samples, model_path, endpoints,
                    output_path, bm["name"], args.reasoning
                )
            )
            summary[bm["name"]] = result

        # Print summary
        log("\n" + "=" * 60)
        log(f"  SUMMARY: {args.model_id}")
        log("=" * 60)
        for bm_name, res in summary.items():
            log(f"  {bm_name:30s}: {res['accuracy']:.4f}  "
                f"({res['n_correct']:.0f}/{res['n_total']})")
        log("=" * 60)

    finally:
        stop_vllm_servers(healthy_procs)
        verify_gpus_freed()


if __name__ == "__main__":
    main()
