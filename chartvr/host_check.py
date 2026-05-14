"""Runtime Qwen 397B host verification + dynamic pool management.

Some on-prem hosts (e.g. 10.1.211.147) rotate between Qwen and GLM by
time-of-day, so every caller must verify the host is actually serving Qwen
397B before issuing requests. This module provides the primitives; the
MultiHostClient uses them to build a dynamic, self-healing host pool.
"""
import sys

import httpx

from chartvr.config import ONPREM_MODEL


def verify_qwen_host(base_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Query {base_url}/models and check if it is serving Qwen 397B.

    Returns (is_qwen, info) where info is the model id on success or a
    short error string on failure.
    """
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        entries = data.get("data") or []
        if entries:
            mid = entries[0].get("id", "")
            return (mid == ONPREM_MODEL, mid or "empty id")
        return (False, "no data field")
    except Exception as e:
        return (False, f"{type(e).__name__}: {str(e)[:80]}")


def filter_qwen_hosts(candidates: list[str], verbose: bool = True) -> list[str]:
    """From candidate host URLs, return only those serving Qwen 397B right now."""
    kept: list[str] = []
    for h in candidates:
        ok, info = verify_qwen_host(h)
        if verbose:
            tag = "✓" if ok else "✗"
            print(f"  [{tag}] {h}  →  {info}", file=sys.stderr, flush=True)
        if ok:
            kept.append(h)
    return kept
