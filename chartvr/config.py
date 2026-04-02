"""Central configuration for ChartVCR project."""
import os

BASE_DIR = os.environ.get("CHARTVR_ROOT", "/ex_disk2/mhpark/poc/chartvr")

# ── On-premise LLM servers (QA generation, verification, reward model) ──
ONPREM_HOSTS = [
    "http://10.1.211.147:8000/v1",
    "http://10.1.211.148:8000/v1",
    "http://10.1.211.169:8000/v1",
    "http://10.1.211.170:8000/v1",
]
ONPREM_MODEL = "Qwen3.5-397B-A17B-FP8"
ONPREM_MAX_CONCURRENT_PER_HOST = 5

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
