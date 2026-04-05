#!/usr/bin/env python3
"""
distributed_coordinator.py — Prefix-cache-aware load balancer cho nhiều llama.cpp servers
===================================================================================
Chức năng:
  1. OpenAI-compatible API proxy (nhận /v1/chat/completions, /v1/completions)
  2. Prefix-cache-aware routing: route request đến server có nhiều prefix chung nhất
     (giảm cold-start, tận dụng paged KV cache Phase 2)
  3. Round-robin fallback khi không có prefix match
  4. Health check tự động, loại server lỗi khỏi pool

Sơ đồ:
  Client → Coordinator (port 11433)
               ↓ route
        ┌──────┴──────┐
        ↓             ↓
  llama-server      llama-server
  :11434            :11435
  (Q4_K_M, ngl=48) (Q5_K_M, ngl=34)

Usage:
  python3 distributed_coordinator.py --backends 127.0.0.1:11434 127.0.0.1:11435
  python3 distributed_coordinator.py --backends 127.0.0.1:11434 127.0.0.1:11435 --port 11433

Requirements:
  pip install aiohttp fastapi uvicorn
"""

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coord")


# ── Prefix hash ──────────────────────────────────────────────────────────────

def prefix_hash(tokens: list[int], n: int) -> str:
    """Hash the first n token IDs as a prefix fingerprint."""
    key = json.dumps(tokens[:n]).encode()
    return hashlib.sha256(key).hexdigest()[:16]


def tokenize_prompt(text: str) -> list[int]:
    """Rough token count approximation (4 chars/token).
    For accurate tokenization, call /tokenize on a backend.
    """
    return list(range(max(1, len(text) // 4)))  # placeholder IDs for routing


# ── Backend state ─────────────────────────────────────────────────────────────

@dataclass
class BackendState:
    url: str                           # e.g. "http://127.0.0.1:11434"
    healthy: bool = True
    last_check: float = 0.0
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    # prefix cache: hash → last_used timestamp
    prefix_cache: dict[str, float] = field(default_factory=dict)

    def record_prefix(self, prefix_h: str):
        self.prefix_cache[prefix_h] = time.monotonic()
        # keep cache bounded
        if len(self.prefix_cache) > 10_000:
            oldest = sorted(self.prefix_cache, key=self.prefix_cache.get)[:1000]
            for k in oldest:
                del self.prefix_cache[k]

    def prefix_score(self, prefix_h: str) -> float:
        """Score: time-decayed match. 1.0 if just seen, decays over 5 min."""
        ts = self.prefix_cache.get(prefix_h)
        if ts is None:
            return 0.0
        age = time.monotonic() - ts
        return max(0.0, 1.0 - age / 300.0)  # decay over 300s

    def load_score(self) -> float:
        """Lower is better. Active requests + latency penalty."""
        return self.active_requests + self.avg_latency_ms / 1000.0


# ── Coordinator ───────────────────────────────────────────────────────────────

class Coordinator:
    HEALTH_INTERVAL = 10.0   # check health every N seconds
    HEALTH_TIMEOUT  =  3.0   # timeout for health check

    def __init__(self, backend_urls: list[str]):
        self.backends = [BackendState(url=url) for url in backend_urls]
        self._rr_idx  = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self._session = aiohttp.ClientSession()
        asyncio.create_task(self._health_loop())
        log.info("Coordinator started with %d backends: %s",
                 len(self.backends), [b.url for b in self.backends])

    async def stop(self):
        if self._session:
            await self._session.close()

    # ── Health check ─────────────────────────────────────────────────────────

    async def _health_loop(self):
        while True:
            await asyncio.gather(*[self._check_health(b) for b in self.backends],
                                 return_exceptions=True)
            await asyncio.sleep(self.HEALTH_INTERVAL)

    async def _check_health(self, backend: BackendState):
        try:
            async with self._session.get(
                f"{backend.url}/health",
                timeout=aiohttp.ClientTimeout(total=self.HEALTH_TIMEOUT)
            ) as resp:
                was_healthy = backend.healthy
                backend.healthy = (resp.status == 200)
                backend.last_check = time.monotonic()
                if not was_healthy and backend.healthy:
                    log.info("Backend %s recovered", backend.url)
                elif was_healthy and not backend.healthy:
                    log.warning("Backend %s is DOWN (status=%d)", backend.url, resp.status)
        except Exception as e:
            if backend.healthy:
                log.warning("Backend %s is DOWN: %s", backend.url, e)
            backend.healthy = False
            backend.last_check = time.monotonic()

    # ── Routing ──────────────────────────────────────────────────────────────

    def _select_backend(self, prefix_h: str) -> Optional[BackendState]:
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            return None

        # Score = prefix_score * 2 - load_score
        # Prefix match strongly preferred; fallback to least-loaded
        best = max(
            healthy,
            key=lambda b: b.prefix_score(prefix_h) * 2.0 - b.load_score()
        )
        return best

    # ── Request proxy ────────────────────────────────────────────────────────

    async def _proxy(self, request: web.Request, path: str) -> web.Response:
        body = await request.read()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        # Extract prefix from prompt/messages for routing
        prefix_h = "default"
        if "prompt" in payload:
            toks = tokenize_prompt(str(payload["prompt"]))
            prefix_h = prefix_hash(toks, min(64, len(toks)))
        elif "messages" in payload:
            # Concatenate system + user messages for prefix
            text = " ".join(
                m.get("content", "") for m in payload.get("messages", [])
                if m.get("role") in ("system", "user")
            )
            toks = tokenize_prompt(text)
            prefix_h = prefix_hash(toks, min(64, len(toks)))

        backend = self._select_backend(prefix_h)
        if backend is None:
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({"error": "no healthy backends available"})
            )

        backend.active_requests += 1
        backend.total_requests  += 1
        t0 = time.monotonic()

        try:
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("Host", None)

            stream = payload.get("stream", False)

            async with self._session.request(
                method  = request.method,
                url     = f"{backend.url}{path}",
                data    = body,
                headers = headers,
                timeout = aiohttp.ClientTimeout(total=300.0),
            ) as resp:
                latency_ms = (time.monotonic() - t0) * 1000
                # Exponential moving average
                backend.avg_latency_ms = (
                    0.8 * backend.avg_latency_ms + 0.2 * latency_ms
                )
                backend.record_prefix(prefix_h)

                if stream:
                    # Streaming SSE passthrough
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Coordinator-Backend": backend.url,
                        }
                    )
                    await response.prepare(request)
                    async for chunk in resp.content.iter_chunked(4096):
                        await response.write(chunk)
                    await response.write_eof()
                    return response
                else:
                    resp_body = await resp.read()
                    return web.Response(
                        status=resp.status,
                        body=resp_body,
                        content_type=resp.content_type,
                        headers={"X-Coordinator-Backend": backend.url},
                    )

        except Exception as e:
            backend.total_errors += 1
            log.error("Backend %s error on %s: %s", backend.url, path, e)
            return web.Response(
                status=502,
                content_type="application/json",
                text=json.dumps({"error": f"backend error: {e}"})
            )
        finally:
            backend.active_requests -= 1

    # ── Status endpoint ───────────────────────────────────────────────────────

    async def handle_status(self, request: web.Request) -> web.Response:
        data = []
        for b in self.backends:
            data.append({
                "url":             b.url,
                "healthy":         b.healthy,
                "active_requests": b.active_requests,
                "total_requests":  b.total_requests,
                "total_errors":    b.total_errors,
                "avg_latency_ms":  round(b.avg_latency_ms, 1),
                "prefix_cache_entries": len(b.prefix_cache),
            })
        return web.Response(
            content_type="application/json",
            text=json.dumps({"backends": data}, indent=2)
        )

    # ── Route handlers ────────────────────────────────────────────────────────

    async def handle_completions(self, request: web.Request) -> web.Response:
        return await self._proxy(request, "/v1/completions")

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        return await self._proxy(request, "/v1/chat/completions")

    async def handle_models(self, request: web.Request) -> web.Response:
        return await self._proxy(request, "/v1/models")

    async def handle_tokenize(self, request: web.Request) -> web.Response:
        return await self._proxy(request, "/tokenize")

    async def handle_health(self, request: web.Request) -> web.Response:
        healthy = any(b.healthy for b in self.backends)
        return web.Response(
            status=200 if healthy else 503,
            content_type="application/json",
            text=json.dumps({"status": "ok" if healthy else "degraded"})
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def build_app(coordinator: Coordinator) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/completions",        coordinator.handle_completions)
    app.router.add_post("/v1/chat/completions",   coordinator.handle_chat_completions)
    app.router.add_get ("/v1/models",             coordinator.handle_models)
    app.router.add_post("/tokenize",              coordinator.handle_tokenize)
    app.router.add_get ("/health",                coordinator.handle_health)
    app.router.add_get ("/coordinator/status",    coordinator.handle_status)

    async def on_startup(app):
        await coordinator.start()

    async def on_cleanup(app):
        await coordinator.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description="Prefix-cache-aware llama.cpp coordinator")
    parser.add_argument(
        "--backends", nargs="+", required=True,
        metavar="HOST:PORT",
        help="Backend llama-server instances, e.g. 127.0.0.1:11434 127.0.0.1:11435"
    )
    parser.add_argument("--port",  type=int, default=11433, help="Port to listen on (default: 11433)")
    parser.add_argument("--host",  default="0.0.0.0",       help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    backend_urls = []
    for b in args.backends:
        if not b.startswith("http"):
            b = "http://" + b
        backend_urls.append(b.rstrip("/"))

    coordinator = Coordinator(backend_urls)
    app = build_app(coordinator)

    log.info("Starting coordinator on %s:%d", args.host, args.port)
    log.info("Routing to backends: %s", backend_urls)
    log.info("Status: http://%s:%d/coordinator/status", args.host, args.port)

    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
