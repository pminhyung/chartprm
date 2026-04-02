"""Multi-host async LLM client with round-robin load balancing."""
import asyncio
from openai import AsyncOpenAI


class MultiHostClient:
    """Round-robin async LLM client.

    Supports:
    - Multi-host on-premise (QA gen, reward model): N hosts × M concurrent
    - Single-host closed API (eval): 1 host × 12 concurrent
    """

    def __init__(self, hosts=None, max_concurrent_per_host=None, timeout=None,
                 model=None, api_key="dummy"):
        from chartvr.config import (ONPREM_HOSTS, ONPREM_MODEL,
                                     ONPREM_MAX_CONCURRENT_PER_HOST, DEFAULT_TIMEOUT)
        hosts = hosts or ONPREM_HOSTS
        self.model = model or ONPREM_MODEL
        max_concurrent_per_host = max_concurrent_per_host or ONPREM_MAX_CONCURRENT_PER_HOST
        timeout = timeout or DEFAULT_TIMEOUT

        self.clients = [
            AsyncOpenAI(base_url=url, api_key=api_key, timeout=timeout)
            for url in hosts
        ]
        self.sem = asyncio.Semaphore(len(hosts) * max_concurrent_per_host)
        self._counter = 0
        self.n_hosts = len(hosts)
        self.max_concurrent = len(hosts) * max_concurrent_per_host

    @classmethod
    def from_closed_api(cls, provider, api_key):
        """Factory for closed API eval (single host, 12 concurrent)."""
        from chartvr.config import CLOSED_API
        cfg = CLOSED_API[provider]
        kwargs = {"max_concurrent_per_host": cfg["max_concurrent"],
                  "model": provider, "api_key": api_key}
        if cfg["base_url"]:
            kwargs["hosts"] = [cfg["base_url"]]
        else:
            # Use provider's default SDK base_url
            kwargs["hosts"] = ["https://api.openai.com/v1"]  # placeholder
        return cls(**kwargs)

    @classmethod
    def for_reward(cls, hosts=None):
        """Factory for reward model (on-premise, same config as QA gen)."""
        return cls(hosts=hosts)

    def _next_client(self):
        self._counter += 1
        return self.clients[self._counter % len(self.clients)]

    async def chat(self, messages, **kwargs):
        """Send chat completion with round-robin + semaphore + retry."""
        client = self._next_client()
        model = kwargs.pop("model", self.model)
        async with self.sem:
            for attempt in range(3):
                try:
                    return await client.chat.completions.create(
                        model=model, messages=messages, **kwargs)
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise
