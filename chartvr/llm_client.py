"""Multi-host async LLM client with round-robin + failover + dynamic pool.

Key properties:
- **Dynamic host pool**: each host is runtime-verified for Qwen 397B.
  Hosts that are down / serving GLM / unreachable are dropped. A background
  revalidation loop re-adds hosts (e.g. 10.1.211.147) when they come back to
  Qwen, so long-running jobs self-heal without manual restarts.
- **Per-request failover**: if the chosen host fails, the request falls over
  to the next live host automatically (up to 2× live-pool size tries).
- **Sampling auto-injection**: Qwen3.5 recommended params are injected based
  on the `enable_thinking` flag in extra_body, with caller kwargs taking
  priority. `presence_penalty`, `top_p`, `temperature` go to the OpenAI
  top-level; `top_k`, `min_p`, `repetition_penalty` go to `extra_body`.
"""
import asyncio
import itertools
import sys

from openai import AsyncOpenAI

from chartvr.config import SAMPLING_PARAMS

# 397B on-premise defaults (used by MultiHostClient)
QWEN_THINKING_PARAMS = SAMPLING_PARAMS["397b"]["thinking"]
QWEN_INSTRUCT_PARAMS = SAMPLING_PARAMS["397b"]["instruct"]


class MultiHostClient:
    """Round-robin async LLM client with failover + dynamic host pool.

    Supports:
    - Multi-host on-premise (QA gen, reward model, teacher distillation):
      N hosts × M concurrent per host, with per-request failover.
    - Single-host closed API (eval): 1 host × 12 concurrent (via factory).

    Construction:
    - `MultiHostClient()` — dynamic pool from config.get_live_qwen_hosts()
      (STABLE + DYNAMIC). Background revalidation loop enabled.
    - `MultiHostClient(hosts=[...])` — explicit host list. Each host is still
      runtime-verified for Qwen; non-Qwen hosts are dropped at init.
      Revalidation loop still runs over the provided candidate list so the
      pool can recover from transient failures.
    - `MultiHostClient(hosts=[...], dynamic=False)` — legacy mode, no verify,
      no revalidation loop (not recommended for new code).
    """

    def __init__(self, hosts=None, max_concurrent_per_host=None, timeout=None,
                 model=None, api_key="dummy", dynamic=True,
                 revalidate_interval=120.0):
        from chartvr.config import (
            ONPREM_HOSTS_QWEN_STABLE, ONPREM_HOSTS_QWEN_DYNAMIC,
            ONPREM_MODEL, ONPREM_MAX_CONCURRENT_PER_HOST, DEFAULT_TIMEOUT,
        )
        self.model = model or ONPREM_MODEL
        self.max_concurrent_per_host = max_concurrent_per_host or ONPREM_MAX_CONCURRENT_PER_HOST
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.api_key = api_key
        self._dynamic = dynamic
        self._revalidate_interval = revalidate_interval
        self._revalidate_task = None

        if hosts is not None:
            self._candidates = list(hosts)
        else:
            self._candidates = list(ONPREM_HOSTS_QWEN_STABLE) + list(ONPREM_HOSTS_QWEN_DYNAMIC)

        if dynamic:
            from chartvr.host_check import filter_qwen_hosts
            live = filter_qwen_hosts(self._candidates, verbose=True)
            if not live:
                raise RuntimeError(
                    f"No live Qwen hosts at init. Candidates: {self._candidates}"
                )
        else:
            # Legacy mode: trust caller, no verify
            live = list(self._candidates)

        self._live_hosts: list[str] = []
        self._clients: list[AsyncOpenAI] = []
        self._sems: list[asyncio.Semaphore] = []
        self._rebuild_clients(live)

        # Round-robin counter (shared across all requests)
        self._rr_counter = 0
        self._pool_lock: asyncio.Lock | None = None  # lazy-init (needs running loop)

        # Legacy-compat attributes
        self.n_hosts = len(self._live_hosts)
        self.max_concurrent = len(self._live_hosts) * self.max_concurrent_per_host
        # Kept for legacy callers that read `.clients` / `.sem`
        self.clients = self._clients
        # Aggregate semaphore (legacy). New code uses per-host _sems.
        self.sem = asyncio.Semaphore(max(self.max_concurrent, 1))

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------
    def _rebuild_clients(self, live_hosts: list[str]) -> None:
        """Rebuild per-host client + semaphore lists from a live-hosts list.

        Called at init and whenever the revalidation loop detects a pool
        change. Callers should hold self._pool_lock when calling this in a
        running loop.
        """
        self._live_hosts = list(live_hosts)
        self._clients = [
            AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=self.timeout)
            for url in self._live_hosts
        ]
        self._sems = [
            asyncio.Semaphore(self.max_concurrent_per_host)
            for _ in self._live_hosts
        ]
        self.n_hosts = len(self._live_hosts)
        self.max_concurrent = self.n_hosts * self.max_concurrent_per_host
        self.clients = self._clients  # legacy mirror

    async def _revalidate_loop(self) -> None:
        """Background task: periodically re-verify all candidate hosts.

        When the Qwen-serving subset changes (e.g. 147 flips from GLM→Qwen,
        or a host crashes), rebuild the client pool accordingly. Runs until
        the owning client is garbage-collected (task cancelled).
        """
        from chartvr.host_check import filter_qwen_hosts
        while True:
            try:
                await asyncio.sleep(self._revalidate_interval)
                new_live = await asyncio.to_thread(
                    filter_qwen_hosts, self._candidates, False
                )
                if not new_live:
                    print(
                        "[host-pool] revalidation found NO live Qwen hosts — "
                        "keeping current pool until one recovers",
                        file=sys.stderr, flush=True,
                    )
                    continue
                async with self._pool_lock:  # type: ignore[union-attr]
                    if set(new_live) != set(self._live_hosts):
                        old = list(self._live_hosts)
                        print(
                            f"[host-pool] pool change: {old} → {new_live}",
                            file=sys.stderr, flush=True,
                        )
                        self._rebuild_clients(new_live)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(
                    f"[host-pool] revalidate error: {type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )

    def _ensure_background_tasks(self) -> None:
        """Lazily start the revalidation loop on first call inside a loop."""
        if not self._dynamic:
            return
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        if self._revalidate_task is None or self._revalidate_task.done():
            self._revalidate_task = asyncio.create_task(self._revalidate_loop())

    # ------------------------------------------------------------------
    # Classmethod factories (legacy compat)
    # ------------------------------------------------------------------
    @classmethod
    def from_closed_api(cls, provider, api_key):
        """Factory for closed API eval (single host, 12 concurrent, no verify)."""
        from chartvr.config import CLOSED_API
        cfg = CLOSED_API[provider]
        if not cfg["base_url"]:
            raise ValueError(
                f"base_url not configured for {provider} in chartvr.config.CLOSED_API"
            )
        return cls(
            hosts=[cfg["base_url"]],
            max_concurrent_per_host=cfg["max_concurrent"],
            model=provider,
            api_key=api_key,
            dynamic=False,
        )

    @classmethod
    def for_reward(cls, hosts=None):
        """Factory for reward model (on-premise, dynamic pool)."""
        return cls(hosts=hosts)

    # ------------------------------------------------------------------
    # Sampling param auto-injection
    # ------------------------------------------------------------------
    def _apply_sampling_defaults(self, kwargs: dict) -> dict:
        """Merge Qwen3.5 recommended sampling params into kwargs (non-destructive).

        Thinking vs instruct is selected by the `enable_thinking` flag in
        `extra_body.chat_template_kwargs`. Caller-provided values always win.
        """
        extra_body = dict(kwargs.get("extra_body") or {})
        enable_thinking = (
            extra_body.get("chat_template_kwargs", {}).get("enable_thinking", False)
        )
        defaults = QWEN_THINKING_PARAMS if enable_thinking else QWEN_INSTRUCT_PARAMS
        for k, v in defaults.items():
            if k in ("top_k", "min_p", "repetition_penalty"):
                extra_body.setdefault(k, v)
            else:
                kwargs.setdefault(k, v)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def chat(self, messages, **kwargs):
        """Send chat completion with per-request failover + sampling injection.

        - Picks a live host by round-robin.
        - On exception, falls over to the next live host automatically, up
          to `2 × n_live` total attempts.
        - Raises RuntimeError only if every attempt fails.
        """
        self._ensure_background_tasks()
        model = kwargs.pop("model", self.model)
        kwargs = self._apply_sampling_defaults(kwargs)

        # Snapshot live pool under the lock so we race safely with
        # _rebuild_clients() from the revalidation loop.
        if self._pool_lock is not None:
            async with self._pool_lock:
                clients_snapshot = list(self._clients)
                sems_snapshot = list(self._sems)
                hosts_snapshot = list(self._live_hosts)
        else:
            clients_snapshot = list(self._clients)
            sems_snapshot = list(self._sems)
            hosts_snapshot = list(self._live_hosts)

        n = len(clients_snapshot)
        if n == 0:
            raise RuntimeError("MultiHostClient: no live hosts available")

        max_tries = n * 2
        last_err: Exception | None = None
        for attempt in range(max_tries):
            self._rr_counter += 1
            idx = self._rr_counter % n
            client = clients_snapshot[idx]
            sem = sems_snapshot[idx]
            host = hosts_snapshot[idx]
            try:
                async with sem:
                    return await client.chat.completions.create(
                        model=model, messages=messages, **kwargs
                    )
            except Exception as e:  # broad: network, auth, server, model busy
                last_err = e
                if attempt < max_tries - 1:
                    # Short backoff only after a full sweep of hosts failed.
                    if attempt >= n - 1:
                        await asyncio.sleep(min(2.0 + 2.0 * (attempt - n + 1), 10.0))
                    # Optional: surface first few errors for visibility
                    if attempt < 3:
                        print(
                            f"[host-failover] {host} failed "
                            f"({type(e).__name__}: {str(e)[:120]}) — retrying",
                            file=sys.stderr, flush=True,
                        )
                    continue
        raise RuntimeError(
            f"MultiHostClient: all {max_tries} attempts failed across "
            f"{n} hosts. Last error: {type(last_err).__name__}: {last_err}"
        )
