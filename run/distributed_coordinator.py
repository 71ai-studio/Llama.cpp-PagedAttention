#!/usr/bin/env python3
"""
distributed_coordinator.py — Prefix-cache-aware load balancer + Worker Registry
=================================================================================

Chức năng:
  1. OpenAI-compatible API proxy (nhận /v1/chat/completions, /v1/completions)
  2. Prefix-cache-aware routing: route request đến server có nhiều prefix chung nhất
  3. Round-robin fallback khi không có prefix match
  4. Health check tự động, loại server lỗi khỏi pool

  5. [Worker Registry] — Worker tự đăng ký, coordinator quản lý RPC list:
     POST /api/workers/register  — VM mới join cluster
     GET  /api/workers           — danh sách workers hiện tại
     DELETE /api/workers/{host}  — remove worker
     Khi workers thay đổi → ghi workers_registry.json → restart llama-server

Sơ đồ single VM → scale out:

  1 VM:                          + VM mới join:
  ┌──────────────────┐           ┌─────────────────────┐
  │ coordinator:11433│ ←─────── POST /api/workers/register
  │ llama-server:11434│          │ llama-rpc-server:50052│
  │ (no --rpc)       │ restart→  │ (auto-registered)    │
  │ (--rpc vm2:50052)│           └─────────────────────┘
  └──────────────────┘

Usage:
  # Single VM — coordinator tự quản lý llama-server qua PM2
  python3 distributed_coordinator.py --server-pm2-name q4km --port 11433

  # Multi llama-server backends (không dùng worker registry)
  python3 distributed_coordinator.py --backends 127.0.0.1:11434 127.0.0.1:11435

Requirements:
  pip install aiohttp
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import socket
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coord")

REGISTRY_FILE = Path(__file__).parent / "workers_registry.json"


# ── Prefix hash ──────────────────────────────────────────────────────────────

def prefix_hash(tokens: list, n: int) -> str:
    key = json.dumps(tokens[:n]).encode()
    return hashlib.sha256(key).hexdigest()[:16]


def tokenize_prompt(text: str) -> list:
    return list(range(max(1, len(text) // 4)))


# ── Backend state ─────────────────────────────────────────────────────────────

@dataclass
class BackendState:
    url: str
    healthy: bool = True
    last_check: float = 0.0
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    prefix_cache: dict = field(default_factory=dict)

    def record_prefix(self, prefix_h: str):
        self.prefix_cache[prefix_h] = time.monotonic()
        if len(self.prefix_cache) > 10_000:
            oldest = sorted(self.prefix_cache, key=self.prefix_cache.get)[:1000]
            for k in oldest:
                del self.prefix_cache[k]

    def prefix_score(self, prefix_h: str) -> float:
        ts = self.prefix_cache.get(prefix_h)
        if ts is None:
            return 0.0
        age = time.monotonic() - ts
        return max(0.0, 1.0 - age / 300.0)

    def load_score(self) -> float:
        return self.active_requests + self.avg_latency_ms / 1000.0


# ── Worker registry ───────────────────────────────────────────────────────────

@dataclass
class WorkerEntry:
    host: str
    rpc_port: int
    gpu: str = ""
    gpu_layers: int = 0
    registered_at: str = ""
    healthy: bool = False
    last_seen: float = 0.0

    def rpc_addr(self) -> str:
        return f"{self.host}:{self.rpc_port}"

    def to_dict(self) -> dict:
        return {
            "host":          self.host,
            "rpc_port":      self.rpc_port,
            "gpu":           self.gpu,
            "gpu_layers":    self.gpu_layers,
            "registered_at": self.registered_at,
            "healthy":       self.healthy,
        }


def _load_registry() -> list[WorkerEntry]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(REGISTRY_FILE.read_text())
        workers = []
        for w in data.get("workers", []):
            workers.append(WorkerEntry(
                host          = w["host"],
                rpc_port      = w["rpc_port"],
                gpu           = w.get("gpu", ""),
                gpu_layers    = w.get("gpu_layers", 0),
                registered_at = w.get("registered_at", ""),
            ))
        return workers
    except Exception as e:
        log.warning("Không đọc được registry: %s", e)
        return []


def _save_registry(workers: list[WorkerEntry]):
    data = {
        "_comment": "Auto-managed by coordinator.py",
        "workers":  [w.to_dict() for w in workers],
    }
    REGISTRY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── Coordinator ───────────────────────────────────────────────────────────────

class Coordinator:
    HEALTH_INTERVAL       = 10.0
    HEALTH_TIMEOUT        = 3.0
    WORKER_CHECK_INTERVAL = 15.0
    RESTART_DEBOUNCE      = 5.0   # đợi 5s sau khi worker thay đổi rồi mới restart

    def __init__(self, backend_urls: list[str],
                 server_pm2_name: Optional[str] = None,
                 server_start_cmd: Optional[str] = None,
                 model_file: Optional[str] = None):
        self.backends        = [BackendState(url=url) for url in backend_urls]
        self._rr_idx         = 0
        self._session: Optional[aiohttp.ClientSession] = None

        # Worker registry mode (quản lý llama-server dynamically)
        self.server_pm2_name  = server_pm2_name   # e.g. "q4km"
        self.server_start_cmd = server_start_cmd  # fallback nếu không dùng PM2
        # Model info (để worker mới biết cần tải model gì)
        self.model_file = model_file or os.environ.get("LLAMA_MODEL", "")
        self.workers: list[WorkerEntry] = _load_registry()
        self._workers_dirty   = False
        self._restart_pending = False
        self._restart_after   = 0.0

        log.info("Registry loaded: %d workers", len(self.workers))
        for w in self.workers:
            log.info("  - %s (gpu=%s)", w.rpc_addr(), w.gpu)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self._session = aiohttp.ClientSession()
        asyncio.create_task(self._health_loop())
        if self.server_pm2_name or self.server_start_cmd:
            asyncio.create_task(self._worker_health_loop())
            asyncio.create_task(self._restart_loop())
        log.info("Coordinator started with %d llama-server backends",
                 len(self.backends))

    async def stop(self):
        if self._session:
            await self._session.close()

    # ── llama-server health ───────────────────────────────────────────────────

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
                backend.healthy    = (resp.status == 200)
                backend.last_check = time.monotonic()
                if not was_healthy and backend.healthy:
                    log.info("Backend %s recovered", backend.url)
                elif was_healthy and not backend.healthy:
                    log.warning("Backend %s DOWN (status=%d)", backend.url, resp.status)
        except Exception as e:
            if backend.healthy:
                log.warning("Backend %s DOWN: %s", backend.url, e)
            backend.healthy = False

    # ── Worker health loop ────────────────────────────────────────────────────

    async def _worker_health_loop(self):
        """Check TCP reachability của mỗi RPC worker."""
        while True:
            changed = False
            for w in self.workers:
                prev    = w.healthy
                w.healthy = await asyncio.get_event_loop().run_in_executor(
                    None, _tcp_reachable, w.host, w.rpc_port
                )
                if w.healthy:
                    w.last_seen = time.monotonic()
                if prev != w.healthy:
                    state = "UP" if w.healthy else "DOWN"
                    log.info("Worker %s → %s", w.rpc_addr(), state)
                    changed = True
            if changed:
                _save_registry(self.workers)
                await self._schedule_restart("worker health changed")
            await asyncio.sleep(self.WORKER_CHECK_INTERVAL)

    # ── Restart loop ──────────────────────────────────────────────────────────

    async def _restart_loop(self):
        """Thực hiện restart llama-server sau debounce."""
        while True:
            await asyncio.sleep(1.0)
            if self._restart_pending and time.monotonic() >= self._restart_after:
                self._restart_pending = False
                await self._do_restart_server()

    async def _schedule_restart(self, reason: str):
        log.info("Scheduling llama-server restart in %.0fs (%s)",
                 self.RESTART_DEBOUNCE, reason)
        self._restart_pending = True
        self._restart_after   = time.monotonic() + self.RESTART_DEBOUNCE

    async def _do_restart_server(self):
        """Cập nhật LLAMA_RPC_WORKERS env và restart server qua PM2."""
        healthy_workers = [w for w in self.workers if w.healthy]
        rpc_list = ",".join(w.rpc_addr() for w in healthy_workers)

        log.info("Restarting server. RPC workers: [%s]", rpc_list or "none")

        # Ghi danh sách workers ra file để server.sh đọc
        _save_registry(self.workers)

        if self.server_pm2_name:
            # PM2: set env LLAMA_RPC_WORKERS rồi restart
            env_cmd = (
                f"pm2 restart {self.server_pm2_name} "
                f"--update-env"
            )
            # Cần set env trước khi restart — dùng pm2 set hoặc truyền qua os.environ
            os.environ["LLAMA_RPC_WORKERS"] = rpc_list
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        env_cmd, shell=True, capture_output=True, text=True
                    )
                )
                if result.returncode == 0:
                    log.info("PM2 restart %s OK", self.server_pm2_name)
                else:
                    log.error("PM2 restart failed: %s", result.stderr)
            except Exception as e:
                log.error("PM2 restart exception: %s", e)
        elif self.server_start_cmd:
            log.info("server_start_cmd mode — set LLAMA_RPC_WORKERS=%s", rpc_list)
            os.environ["LLAMA_RPC_WORKERS"] = rpc_list

    # ── Routing ──────────────────────────────────────────────────────────────

    def _select_backend(self, prefix_h: str) -> Optional[BackendState]:
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            return None
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

        prefix_h = "default"
        if "prompt" in payload:
            toks = tokenize_prompt(str(payload["prompt"]))
            prefix_h = prefix_hash(toks, min(64, len(toks)))
        elif "messages" in payload:
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
                backend.avg_latency_ms = 0.8 * backend.avg_latency_ms + 0.2 * latency_ms
                backend.record_prefix(prefix_h)

                if stream:
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

    # ── Worker Registry API ───────────────────────────────────────────────────

    async def handle_worker_register(self, request: web.Request) -> web.Response:
        """
        POST /api/workers/register
        Body: {"host": "192.168.1.11", "rpc_port": 50052, "gpu": "RTX 3090", "gpu_layers": 22}
        """
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400,
                content_type="application/json",
                text=json.dumps({"error": "invalid JSON"}))

        host     = body.get("host", "").strip()
        rpc_port = int(body.get("rpc_port", 50052))
        gpu      = body.get("gpu", "")
        gpu_layers = int(body.get("gpu_layers", 0))

        if not host:
            return web.Response(status=400,
                content_type="application/json",
                text=json.dumps({"error": "host required"}))

        # Upsert
        existing = next((w for w in self.workers if w.host == host), None)
        if existing:
            existing.rpc_port      = rpc_port
            existing.gpu           = gpu
            existing.gpu_layers    = gpu_layers
            existing.registered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            log.info("Worker updated: %s:%d (gpu=%s)", host, rpc_port, gpu)
            action = "updated"
        else:
            w = WorkerEntry(
                host          = host,
                rpc_port      = rpc_port,
                gpu           = gpu,
                gpu_layers    = gpu_layers,
                registered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self.workers.append(w)
            log.info("Worker registered: %s:%d (gpu=%s)", host, rpc_port, gpu)
            action = "registered"

        _save_registry(self.workers)

        # Check reachability ngay lập tức
        reachable = await asyncio.get_event_loop().run_in_executor(
            None, _tcp_reachable, host, rpc_port, 5.0
        )
        target = next((w for w in self.workers if w.host == host), None)
        if target:
            target.healthy = reachable

        if reachable:
            await self._schedule_restart(f"new worker {host}:{rpc_port}")
            msg = f"Worker {action} and healthy. Server restart scheduled."
        else:
            msg = f"Worker {action} but RPC port {rpc_port} not reachable yet. Will retry."

        return web.Response(
            status=200,
            content_type="application/json",
            text=json.dumps({
                "status":    "ok",
                "action":    action,
                "host":      host,
                "rpc_port":  rpc_port,
                "reachable": reachable,
                "message":   msg,
            })
        )

    async def handle_worker_list(self, request: web.Request) -> web.Response:
        """GET /api/workers"""
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "workers": [w.to_dict() for w in self.workers],
                "rpc_list": ",".join(
                    w.rpc_addr() for w in self.workers if w.healthy
                ),
            }, indent=2)
        )

    async def handle_worker_remove(self, request: web.Request) -> web.Response:
        """DELETE /api/workers/{host}"""
        host = request.match_info.get("host", "")
        before = len(self.workers)
        self.workers = [w for w in self.workers if w.host != host]
        removed = before - len(self.workers)

        if removed:
            _save_registry(self.workers)
            await self._schedule_restart(f"worker {host} removed")
            log.info("Worker removed: %s", host)

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "status":  "ok" if removed else "not_found",
                "removed": removed,
                "host":    host,
            })
        )

    # ── Status endpoint ───────────────────────────────────────────────────────

    async def handle_status(self, request: web.Request) -> web.Response:
        backends_data = [{
            "url":             b.url,
            "healthy":         b.healthy,
            "active_requests": b.active_requests,
            "total_requests":  b.total_requests,
            "total_errors":    b.total_errors,
            "avg_latency_ms":  round(b.avg_latency_ms, 1),
            "prefix_cache_entries": len(b.prefix_cache),
        } for b in self.backends]

        workers_data = [w.to_dict() for w in self.workers]
        healthy_workers = [w for w in self.workers if w.healthy]

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "backends": backends_data,
                "workers": {
                    "total":   len(self.workers),
                    "healthy": len(healthy_workers),
                    "rpc_list": ",".join(w.rpc_addr() for w in healthy_workers),
                    "list":    workers_data,
                },
                "restart_pending": self._restart_pending,
            }, indent=2)
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

    async def handle_info(self, request: web.Request) -> web.Response:
        """GET /api/info — thông tin cluster để worker mới biết cần tải model gì."""
        # Cố đọc model từ PM2 env nếu chưa có
        model = self.model_file
        if not model and self.server_pm2_name:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["pm2", "describe", self.server_pm2_name],
                        capture_output=True, text=True
                    )
                )
                for line in result.stdout.splitlines():
                    if "LLAMA_MODEL" in line:
                        parts = line.split()
                        idx = next((i for i, p in enumerate(parts) if "LLAMA_MODEL" in p), -1)
                        if idx >= 0 and idx + 1 < len(parts):
                            model = parts[idx + 1].strip("│|")
                            break
            except Exception:
                pass

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "model_file":       model or "",
                "rpc_port_default": 50052,
                "hf_repo":          "unsloth/Qwen3.5-27B-GGUF",
                "workers_total":    len(self.workers),
                "workers_healthy":  sum(1 for w in self.workers if w.healthy),
            })
        )


# ── App builder ───────────────────────────────────────────────────────────────

def build_app(coordinator: Coordinator) -> web.Application:
    app = web.Application()

    # OpenAI-compatible proxy
    app.router.add_post("/v1/completions",      coordinator.handle_completions)
    app.router.add_post("/v1/chat/completions", coordinator.handle_chat_completions)
    app.router.add_get ("/v1/models",           coordinator.handle_models)
    app.router.add_post("/tokenize",            coordinator.handle_tokenize)
    app.router.add_get ("/health",              coordinator.handle_health)
    app.router.add_get ("/coordinator/status",  coordinator.handle_status)

    # Worker registry
    app.router.add_post  ("/api/workers/register",    coordinator.handle_worker_register)
    app.router.add_get   ("/api/workers",             coordinator.handle_worker_list)
    app.router.add_delete("/api/workers/{host}",      coordinator.handle_worker_remove)
    app.router.add_get   ("/api/info",                coordinator.handle_info)

    async def on_startup(app):
        await coordinator.start()

    async def on_cleanup(app):
        await coordinator.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prefix-cache-aware llama.cpp coordinator + worker registry"
    )
    parser.add_argument(
        "--backends", nargs="*", default=[],
        metavar="HOST:PORT",
        help="Static llama-server backends, e.g. 127.0.0.1:11434"
    )
    parser.add_argument(
        "--server-pm2-name", default=None,
        metavar="NAME",
        help="PM2 app name của llama-server để restart khi worker thay đổi (e.g. q4km)"
    )
    parser.add_argument(
        "--port", type=int, default=11433,
        help="Port coordinator lắng nghe (default: 11433)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--model-file", default=None,
        metavar="FILENAME",
        help="GGUF filename đang chạy (e.g. Qwen3.5-27B-Q4_K_M.gguf) — "
             "expose qua /api/info để worker mới biết cần tải model gì. "
             "Default: đọc từ env LLAMA_MODEL."
    )
    args = parser.parse_args()

    backend_urls = []
    for b in (args.backends or []):
        if not b.startswith("http"):
            b = "http://" + b
        backend_urls.append(b.rstrip("/"))

    # Default: proxy tới local llama-server nếu không có backend
    if not backend_urls:
        backend_urls = ["http://127.0.0.1:11434"]
        log.info("Không có --backends → default http://127.0.0.1:11434")

    coordinator = Coordinator(
        backend_urls     = backend_urls,
        server_pm2_name  = args.server_pm2_name,
        model_file       = args.model_file,
    )
    app = build_app(coordinator)

    log.info("Coordinator: http://%s:%d", args.host, args.port)
    log.info("Backends:    %s", backend_urls)
    log.info("PM2 name:    %s", args.server_pm2_name or "(không quản lý restart)")
    log.info("")
    log.info("Worker registry API:")
    log.info("  POST   /api/workers/register   — VM mới join cluster")
    log.info("  GET    /api/workers            — list workers")
    log.info("  DELETE /api/workers/{host}     — remove worker")
    log.info("  GET    /coordinator/status     — full status")

    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
