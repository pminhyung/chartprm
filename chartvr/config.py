"""Central configuration for ChartVCR project."""
import os

BASE_DIR = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ── On-premise LLM servers (QA generation, verification, reward model) ──
# Legacy flat list (kept for backward compat; new code should use the
# get_live_qwen_hosts() helper below instead).
ONPREM_HOSTS = [
    "http://10.1.211.147:8000/v1",
    "http://10.1.211.148:8000/v1",
    "http://10.1.211.169:8000/v1",
    "http://10.1.211.170:8000/v1",
]
ONPREM_MODEL = "Qwen3.5-397B-A17B-FP8"
ONPREM_MAX_CONCURRENT_PER_HOST = 5

# Stable Qwen 397B hosts — always serving Qwen (still runtime-verified on use).
ONPREM_HOSTS_QWEN_STABLE = [
    "http://10.1.211.148:8000/v1",   # Qwen 397B fp8
    "http://localhost:9201/v1",       # Local Qwen 397B GPTQ-int4
]

# Dynamic hosts — rotate between Qwen and GLM by time-of-day.
# Fri 09:00 ~ Mon 09:00 is nominally Qwen, but always verify at runtime.
ONPREM_HOSTS_QWEN_DYNAMIC = [
    "http://10.1.211.147:8000/v1",
]


def get_live_qwen_hosts(include_dynamic: bool = True, verbose: bool = True) -> list[str]:
    """Return candidate on-prem hosts that are currently serving Qwen 397B.

    Performs a live /v1/models check on each candidate; hosts that are down,
    unreachable, or serving a different model (GLM) are excluded. This is the
    entry point any script should use when constructing a host pool.
    """
    from chartvr.host_check import filter_qwen_hosts
    candidates = list(ONPREM_HOSTS_QWEN_STABLE)
    if include_dynamic:
        candidates += ONPREM_HOSTS_QWEN_DYNAMIC
    return filter_qwen_hosts(candidates, verbose=verbose)

# ── Closed API (eval only — do NOT use without user permission) ──
CLOSED_API = {
    "gpt-4o": {"base_url": None, "max_concurrent": 12},
    "gemini-2.5-pro": {"base_url": None, "max_concurrent": 12},
    "claude-4-sonnet": {"base_url": None, "max_concurrent": 12},
}

# ── Policy models (local GPU) ──
POLICY_MODELS = {
    "4b": os.path.join(BASE_DIR, "models/qwen3.5-4b"),
    "9b": os.path.join(BASE_DIR, "models/qwen3.5-9b"),
}

# ── Data paths ──
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Defaults ──
DEFAULT_TIMEOUT = 120.0

# ── Qwen3.5 Recommended Sampling Parameters ──
# Source: Qwen3.5 official docs per model size

SAMPLING_PARAMS = {
    "4b": {
        "thinking_general": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
        "thinking_coding": {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        },
        "instruct_general": {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
        "instruct_reasoning": {
            "temperature": 1.0, "top_p": 1.0, "top_k": 40,
            "min_p": 0.0, "presence_penalty": 2.0, "repetition_penalty": 1.0,
        },
    },
    "9b": {
        "thinking_general": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
        "thinking_coding": {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        },
        "instruct_general": {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
        "instruct_reasoning": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
    },
    # 397B on-premise (QA generation, verification)
    # Matches Qwen3.5 official recommended sampling params exactly.
    "397b": {
        "thinking": {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.0,
        },
        "instruct": {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.0,
        },
    },
    # 27B VLM (judge + teacher distillation). User-supplied 2026-05-08.
    # 2026-05-08: thinking temp 1.0 → 0.6 (over-thinking length-cap mitigation).
    "27b": {
        "thinking": {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
        },
        "instruct": {
            "temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0,
        },
    },
}
