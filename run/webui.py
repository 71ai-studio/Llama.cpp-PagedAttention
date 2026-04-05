#!/usr/bin/env python3
"""
webui.py — Local Web UI để quản lý llama.cpp cluster
=====================================================
Tính năng:
  - Dashboard: status PM2 apps, GPU/RAM stats realtime
  - Processes: start/stop/restart từng app, xem env vars
  - Workers: quản lý RPC workers (view/add/remove)
  - Logs: live tail PM2 logs qua SSE
  - Inference: chat UI test nhanh qua coordinator
  - Metrics: VRAM, requests/s, latency từ /metrics endpoint

Usage:
  python3 run/webui.py                    # port 7860
  python3 run/webui.py --port 8080
  python3 run/webui.py --coordinator http://localhost:11433
  python3 run/webui.py --llama-server http://localhost:11434
"""

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import psutil
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR      = Path(__file__).parent
REGISTRY_FILE   = SCRIPT_DIR / "workers_registry.json"
ECOSYSTEM_FILE  = SCRIPT_DIR / "ecosystem.config.js"

DEFAULT_PORT        = 7860
DEFAULT_COORDINATOR = "http://127.0.0.1:11433"
DEFAULT_LLAMA       = "http://127.0.0.1:11434"

app = FastAPI(title="llama.cpp WebUI")

# populated by main()
CFG: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_cmd(cmd: str | list, timeout: int = 10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd, shell=isinstance(cmd, str),
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def pm2_list() -> list[dict]:
    code, out, _ = run_cmd("pm2 jlist")
    if code != 0 or not out:
        return []
    try:
        data = json.loads(out)
        result = []
        for p in data:
            env = p.get("pm2_env", {})
            result.append({
                "name":    p.get("name", ""),
                "id":      p.get("pm_id", 0),
                "status":  env.get("status", "unknown"),
                "pid":     p.get("pid", 0),
                "uptime":  env.get("pm_uptime", 0),
                "restarts": env.get("restart_time", 0),
                "cpu":     p.get("monit", {}).get("cpu", 0),
                "memory":  p.get("monit", {}).get("memory", 0),
                "pm2_env": env,
            })
        return result
    except Exception:
        return []


def gpu_stats() -> list[dict]:
    """nvidia-smi hoặc fallback rỗng."""
    code, out, _ = run_cmd(
        "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,"
        "memory.used,memory.total,power.draw,power.limit "
        "--format=csv,noheader,nounits"
    )
    if code != 0 or not out:
        return []
    result = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            try:
                result.append({
                    "name":      parts[0],
                    "temp":      int(parts[1]),
                    "util":      int(parts[2]),
                    "vram_used": int(parts[3]),
                    "vram_total": int(parts[4]),
                    "power":     float(parts[5]) if parts[5] != "[N/A]" else 0,
                    "power_limit": float(parts[6]) if len(parts) > 6 and parts[6] != "[N/A]" else 0,
                })
            except (ValueError, IndexError):
                pass
    return result


def sys_stats() -> dict:
    mem = psutil.virtual_memory()
    return {
        "cpu_pct":    psutil.cpu_percent(interval=0.2),
        "ram_used":   mem.used // (1024 * 1024),
        "ram_total":  mem.total // (1024 * 1024),
        "ram_pct":    mem.percent,
        "ram_avail":  mem.available // (1024 * 1024),
    }


def load_workers() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(REGISTRY_FILE.read_text())
        return data.get("workers", [])
    except Exception:
        return []


def save_workers(workers: list[dict]):
    data = {
        "_comment": "Auto-managed by coordinator.py",
        "workers": workers,
    }
    REGISTRY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


async def http_get(url: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.json(content_type=None)
    except Exception:
        return None


async def http_post(url: str, body: dict, timeout: float = 60.0) -> Optional[dict]:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body,
                              timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.json(content_type=None)
    except Exception:
        return None


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    processes  = pm2_list()
    gpus       = gpu_stats()
    sys        = sys_stats()
    workers    = load_workers()
    coord_ok   = (await http_get(f"{CFG['coordinator']}/health")) is not None
    llama_ok   = (await http_get(f"{CFG['llama']}/health")) is not None
    return JSONResponse({
        "processes":    processes,
        "gpus":         gpus,
        "sys":          sys,
        "workers":      workers,
        "coordinator":  {"url": CFG["coordinator"], "healthy": coord_ok},
        "llama":        {"url": CFG["llama"], "healthy": llama_ok},
        "ts":           int(time.time()),
    })


@app.post("/api/process/{name}/restart")
async def api_restart(name: str):
    code, out, err = run_cmd(f"pm2 restart {name} --update-env")
    return JSONResponse({"ok": code == 0, "out": out, "err": err})


@app.post("/api/process/{name}/stop")
async def api_stop(name: str):
    code, out, err = run_cmd(f"pm2 stop {name}")
    return JSONResponse({"ok": code == 0, "out": out, "err": err})


@app.post("/api/process/{name}/start")
async def api_start(name: str):
    code, out, err = run_cmd(
        f"pm2 start {ECOSYSTEM_FILE} --only {name}"
    )
    return JSONResponse({"ok": code == 0, "out": out, "err": err})


@app.get("/api/process/{name}/env")
async def api_env(name: str):
    processes = pm2_list()
    p = next((x for x in processes if x["name"] == name), None)
    if not p:
        return JSONResponse({"error": "not found"}, status_code=404)
    env = p.get("pm2_env", {}).get("env", {})
    return JSONResponse({"name": name, "env": env})


@app.get("/api/workers")
async def api_workers():
    data = await http_get(f"{CFG['coordinator']}/api/workers")
    if data:
        return JSONResponse(data)
    # fallback: đọc file trực tiếp
    return JSONResponse({"workers": load_workers(), "rpc_list": ""})


@app.post("/api/workers")
async def api_workers_add(request: Request):
    body = await request.json()
    data = await http_post(f"{CFG['coordinator']}/api/workers/register", body)
    if data:
        return JSONResponse(data)
    return JSONResponse({"error": "coordinator không phản hồi"}, status_code=502)


@app.delete("/api/workers/{host}")
async def api_workers_remove(host: str):
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.delete(
                f"{CFG['coordinator']}/api/workers/{host}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                return JSONResponse(await r.json(content_type=None))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


@app.get("/api/metrics")
async def api_metrics():
    data = await http_get(f"{CFG['llama']}/metrics", timeout=5.0)
    coord = await http_get(f"{CFG['coordinator']}/coordinator/status", timeout=3.0)
    return JSONResponse({"llama_metrics": data, "coordinator": coord})


@app.post("/api/chat")
async def api_chat(request: Request):
    body  = await request.json()
    messages = body.get("messages", [])
    stream   = body.get("stream", False)
    model    = body.get("model", "local")

    payload = {"model": model, "messages": messages, "stream": stream,
               "temperature": body.get("temperature", 0.7),
               "max_tokens":  body.get("max_tokens", 512)}

    target = f"{CFG['coordinator']}/v1/chat/completions"

    if stream:
        async def gen() -> AsyncGenerator[bytes, None]:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as s:
                    async with s.post(target, json=payload,
                                      timeout=aiohttp.ClientTimeout(total=120)) as r:
                        async for chunk in r.content.iter_chunked(1024):
                            yield chunk
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        data = await http_post(target, payload, timeout=120)
        return JSONResponse(data or {"error": "no response"})


@app.get("/api/logs/{name}")
async def api_logs_sse(name: str, lines: int = 50):
    """SSE stream: tail PM2 logs realtime."""
    async def generate() -> AsyncGenerator[str, None]:
        # Gửi log cũ trước
        _, out, _ = run_cmd(f"pm2 logs {name} --lines {lines} --nostream --raw")
        for line in out.splitlines():
            yield f"data: {json.dumps({'line': line, 'ts': time.time()})}\n\n"

        # Follow log files
        log_dir = Path.home() / ".pm2" / "logs"
        out_log = log_dir / f"{name}-out.log"
        err_log = log_dir / f"{name}-error.log"

        # Dùng tail -f để stream
        cmd = ["tail", "-f", "-n", "0"]
        if out_log.exists():
            cmd.append(str(out_log))
        if err_log.exists():
            cmd.append(str(err_log))

        if len(cmd) <= 4:
            yield f"data: {json.dumps({'line': f'[log files not found for {name}]', 'ts': time.time()})}\n\n"
            return

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            while True:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=30.0
                )
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                yield f"data: {json.dumps({'line': text, 'ts': time.time()})}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'line': '[heartbeat]', 'ts': time.time()})}\n\n"
        except Exception:
            pass
        finally:
            proc.kill()

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llama.cpp WebUI</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config = { darkMode: 'class' }</script>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; }
  .tab-active { @apply border-b-2 border-indigo-500 text-indigo-400; }
  .badge-online  { background:#16a34a22; color:#4ade80; border:1px solid #16a34a55; }
  .badge-stopped { background:#71717a22; color:#a1a1aa; border:1px solid #71717a55; }
  .badge-erroring{ background:#dc262622; color:#f87171; border:1px solid #dc262655; }
  .badge-unknown { background:#ca8a0422; color:#fbbf24; border:1px solid #ca8a0455; }
  #log-box { font-family: 'Menlo', 'Consolas', monospace; font-size: 12px; }
  .log-err  { color: #f87171; }
  .log-warn { color: #fbbf24; }
  .log-info { color: #94a3b8; }
  .metric-bar { transition: width 0.4s ease; }
  .card { @apply bg-gray-800 rounded-xl border border-gray-700 p-4; }
  input, select { background:#1e293b; border:1px solid #334155; color:#e2e8f0;
    border-radius:6px; padding:6px 10px; outline:none; width:100%; }
  input:focus, select:focus { border-color:#6366f1; }
  .btn { cursor:pointer; border-radius:6px; padding:6px 14px;
    font-size:13px; font-weight:500; transition:opacity .15s; }
  .btn:hover { opacity:0.85; }
  .btn-primary { background:#4f46e5; color:#fff; }
  .btn-danger  { background:#dc2626; color:#fff; }
  .btn-gray    { background:#374151; color:#e5e7eb; }
  .btn-green   { background:#15803d; color:#fff; }
  .spinner { border:3px solid #334155; border-top-color:#6366f1;
    border-radius:50%; width:18px; height:18px; animation:spin .6s linear infinite; display:none; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .chat-msg-user { background:#312e81; border-radius:12px 12px 2px 12px; }
  .chat-msg-ai   { background:#1e293b; border-radius:12px 12px 12px 2px; border:1px solid #334155; }
</style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">

<!-- Header -->
<header class="bg-gray-800 border-b border-gray-700 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
  <div class="flex items-center gap-3">
    <span class="text-xl font-bold text-indigo-400">&#9650; llama.cpp</span>
    <span class="text-gray-500 text-sm">WebUI</span>
  </div>
  <div class="flex items-center gap-4">
    <span id="hdr-coord" class="text-xs px-2 py-1 rounded" style="background:#1e293b;border:1px solid #334155">
      coordinator —
    </span>
    <span id="hdr-llama" class="text-xs px-2 py-1 rounded" style="background:#1e293b;border:1px solid #334155">
      llama-server —
    </span>
    <button onclick="refreshAll()" class="btn btn-gray text-xs">&#8635; Refresh</button>
  </div>
</header>

<!-- Tabs -->
<nav class="bg-gray-800 border-b border-gray-700 px-6 flex gap-6">
  <button onclick="showTab('dashboard')" id="tab-dashboard"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Dashboard
  </button>
  <button onclick="showTab('processes')" id="tab-processes"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Processes
  </button>
  <button onclick="showTab('workers')" id="tab-workers"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Workers
  </button>
  <button onclick="showTab('logs')" id="tab-logs"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Logs
  </button>
  <button onclick="showTab('chat')" id="tab-chat"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Inference
  </button>
  <button onclick="showTab('metrics')" id="tab-metrics"
    class="tab-btn py-3 text-sm font-medium text-gray-400 hover:text-white border-b-2 border-transparent transition-colors">
    Metrics
  </button>
</nav>

<main class="p-6 max-w-7xl mx-auto">

<!-- ═══════════ DASHBOARD ═══════════ -->
<div id="page-dashboard" class="page space-y-6">

  <!-- GPU cards -->
  <div id="gpu-cards" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4"></div>

  <!-- System stats -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="sys-cards"></div>

  <!-- Process table -->
  <div class="card">
    <h2 class="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">PM2 Processes</h2>
    <div id="proc-table"></div>
  </div>
</div>

<!-- ═══════════ PROCESSES ═══════════ -->
<div id="page-processes" class="page hidden space-y-4">
  <div id="proc-cards" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
</div>

<!-- ═══════════ WORKERS ═══════════ -->
<div id="page-workers" class="page hidden space-y-4">
  <div class="card">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wide">RPC Workers</h2>
      <button onclick="document.getElementById('add-worker-form').classList.toggle('hidden')"
        class="btn btn-primary text-xs">+ Add Worker</button>
    </div>
    <div id="add-worker-form" class="hidden mb-4 p-4 bg-gray-700 rounded-lg grid grid-cols-2 md:grid-cols-4 gap-3">
      <div><label class="text-xs text-gray-400">Host / IP</label>
        <input id="w-host" placeholder="192.168.1.11"></div>
      <div><label class="text-xs text-gray-400">RPC Port</label>
        <input id="w-port" value="50052" type="number"></div>
      <div><label class="text-xs text-gray-400">GPU Name</label>
        <input id="w-gpu" placeholder="RTX 3090"></div>
      <div class="flex items-end">
        <button onclick="addWorker()" class="btn btn-green w-full">Register</button>
      </div>
    </div>
    <div id="workers-table"></div>
  </div>
</div>

<!-- ═══════════ LOGS ═══════════ -->
<div id="page-logs" class="page hidden">
  <div class="card">
    <div class="flex items-center gap-3 mb-3">
      <select id="log-app-select" onchange="startLogStream()" style="width:200px">
        <option value="q4km">q4km</option>
        <option value="coordinator">coordinator</option>
        <option value="q5km">q5km</option>
        <option value="q6k">q6k</option>
        <option value="q4km-flat">q4km-flat</option>
      </select>
      <button onclick="clearLogs()" class="btn btn-gray text-xs">Clear</button>
      <label class="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
        <input type="checkbox" id="log-autoscroll" checked class="w-3 h-3"> Auto-scroll
      </label>
      <label class="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
        <input type="checkbox" id="log-filter-err" class="w-3 h-3"> Errors only
      </label>
      <span id="log-status" class="text-xs text-gray-500 ml-auto"></span>
    </div>
    <div id="log-box" class="bg-gray-950 rounded-lg p-3 h-[60vh] overflow-y-auto text-xs leading-5"></div>
  </div>
</div>

<!-- ═══════════ INFERENCE / CHAT ═══════════ -->
<div id="page-chat" class="page hidden">
  <div class="card flex flex-col" style="height:80vh">
    <div class="flex items-center gap-3 mb-3 pb-3 border-b border-gray-700">
      <select id="chat-model" style="width:240px">
        <option value="local">local (default)</option>
      </select>
      <input id="chat-temp" type="number" value="0.7" step="0.1" min="0" max="2" style="width:80px">
      <span class="text-xs text-gray-500">temp</span>
      <input id="chat-max-tokens" type="number" value="512" step="64" min="64" style="width:90px">
      <span class="text-xs text-gray-500">max tokens</span>
      <button onclick="clearChat()" class="btn btn-gray text-xs ml-auto">Clear</button>
      <span id="chat-tps" class="text-xs text-green-400"></span>
    </div>
    <div id="chat-messages" class="flex-1 overflow-y-auto space-y-3 mb-3 pr-1"></div>
    <div class="flex gap-2">
      <textarea id="chat-input" rows="2"
        class="flex-1 resize-none rounded-lg border border-gray-600 bg-gray-950 text-sm p-2 text-gray-100"
        placeholder="Nhập tin nhắn... (Ctrl+Enter để gửi)"
        onkeydown="if(event.ctrlKey&&event.key==='Enter'){sendChat();}"></textarea>
      <button onclick="sendChat()" id="chat-send-btn" class="btn btn-primary self-end">Send</button>
    </div>
  </div>
</div>

<!-- ═══════════ METRICS ═══════════ -->
<div id="page-metrics" class="page hidden space-y-4">
  <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="metric-cards"></div>
  <div class="card">
    <h2 class="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">VRAM History</h2>
    <canvas id="vram-chart" height="120"></canvas>
  </div>
  <div class="card">
    <h2 class="text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">Raw /metrics (Prometheus)</h2>
    <pre id="raw-metrics" class="text-xs text-gray-400 overflow-x-auto max-h-64"></pre>
  </div>
</div>

</main>

<!-- Toast -->
<div id="toast" class="fixed bottom-6 right-6 hidden z-50
  bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 text-sm max-w-sm shadow-xl"></div>

<!-- Env Modal -->
<div id="env-modal" class="hidden fixed inset-0 bg-black/60 z-50 flex items-center justify-center">
  <div class="bg-gray-800 rounded-xl border border-gray-700 p-6 w-full max-w-xl max-h-[80vh] overflow-y-auto">
    <div class="flex justify-between mb-3">
      <h3 id="env-modal-title" class="font-semibold">Environment Variables</h3>
      <button onclick="document.getElementById('env-modal').classList.add('hidden')"
        class="text-gray-500 hover:text-white text-xl">&times;</button>
    </div>
    <div id="env-modal-body" class="text-sm font-mono space-y-1 text-gray-300"></div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let currentTab = 'dashboard';
let logEventSource = null;
let logLines = [];
let refreshInterval = null;
let vramHistory = [];
const MAX_VRAM_HISTORY = 60;
let chatMessages = [];

// ── Tab switching ──────────────────────────────────────────────────────────
function showTab(tab) {
  document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.remove('border-indigo-500','text-indigo-400');
    b.classList.add('border-transparent','text-gray-400');
  });
  document.getElementById('page-' + tab).classList.remove('hidden');
  const btn = document.getElementById('tab-' + tab);
  btn.classList.add('border-indigo-500','text-indigo-400');
  btn.classList.remove('border-transparent','text-gray-400');
  currentTab = tab;

  if (tab === 'logs') startLogStream();
  if (tab === 'metrics') loadMetrics();
  if (tab === 'workers') loadWorkers();
}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, type='info') {
  const el = document.getElementById('toast');
  el.className = 'fixed bottom-6 right-6 z-50 rounded-xl px-4 py-3 text-sm max-w-sm shadow-xl';
  el.style.background = type==='error'?'#7f1d1d':type==='ok'?'#14532d':'#1e293b';
  el.style.border = `1px solid ${type==='error'?'#b91c1c':type==='ok'?'#15803d':'#334155'}`;
  el.textContent = msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 3000);
}

// ── Helpers ────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b > 1e9) return (b/1e9).toFixed(1)+'GB';
  if (b > 1e6) return (b/1e6).toFixed(0)+'MB';
  return b+'B';
}
function fmtUptime(ms) {
  if (!ms) return '-';
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 60) return s+'s';
  if (s < 3600) return Math.floor(s/60)+'m';
  return Math.floor(s/3600)+'h ' + Math.floor((s%3600)/60)+'m';
}
function statusBadge(s) {
  const map = {online:'badge-online',stopped:'badge-stopped',
    erroring:'badge-erroring',launching:'badge-unknown'};
  const cls = map[s] || 'badge-unknown';
  return `<span class="${cls} px-2 py-0.5 rounded text-xs font-medium">${s}</span>`;
}

function bar(pct, color='#6366f1') {
  return `<div class="bg-gray-700 rounded-full h-2 w-full overflow-hidden">
    <div class="metric-bar h-2 rounded-full" style="width:${pct}%;background:${color}"></div>
  </div>`;
}

// ── Refresh all ────────────────────────────────────────────────────────────
async function refreshAll() {
  const data = await fetch('/api/status').then(r=>r.json()).catch(()=>null);
  if (!data) return;

  // Header badges
  const chdr = document.getElementById('hdr-coord');
  const lhdr = document.getElementById('hdr-llama');
  chdr.textContent = 'coordinator ' + (data.coordinator.healthy ? '●' : '○');
  chdr.style.color = data.coordinator.healthy ? '#4ade80' : '#f87171';
  lhdr.textContent = 'llama-server ' + (data.llama.healthy ? '●' : '○');
  lhdr.style.color = data.llama.healthy ? '#4ade80' : '#f87171';

  // GPU cards
  renderGpuCards(data.gpus);

  // Sys stats
  renderSysCards(data.sys);

  // Proc table (dashboard)
  renderProcTable(data.processes);

  // Proc cards (processes tab)
  renderProcCards(data.processes);

  // VRAM history for chart
  if (data.gpus.length > 0) {
    vramHistory.push({
      ts: data.ts,
      used: data.gpus[0].vram_used,
      total: data.gpus[0].vram_total,
    });
    if (vramHistory.length > MAX_VRAM_HISTORY) vramHistory.shift();
    if (currentTab === 'metrics') renderVramChart();
  }

  // Workers count badge
  const wh = data.workers.filter(w=>w.healthy).length;
  document.getElementById('tab-workers').textContent =
    `Workers${wh>0?' ('+wh+')':''}`;
}

// ── GPU cards ──────────────────────────────────────────────────────────────
function renderGpuCards(gpus) {
  const el = document.getElementById('gpu-cards');
  if (!gpus.length) {
    el.innerHTML = '<div class="card text-gray-500 text-sm">Không tìm thấy GPU (nvidia-smi không khả dụng)</div>';
    return;
  }
  el.innerHTML = gpus.map(g => {
    const vp = Math.round(g.vram_used/g.vram_total*100);
    const tc = g.temp > 80 ? '#f87171' : g.temp > 65 ? '#fbbf24' : '#4ade80';
    const uc = g.util > 90 ? '#f87171' : '#6366f1';
    const pw = g.power_limit > 0 ? Math.round(g.power/g.power_limit*100) : 0;
    return `<div class="card">
      <div class="flex justify-between items-start mb-3">
        <div>
          <div class="font-semibold text-sm">${g.name}</div>
          <div class="text-xs text-gray-500 mt-0.5">${g.vram_used} / ${g.vram_total} MiB</div>
        </div>
        <div class="text-right">
          <div class="text-2xl font-bold" style="color:${tc}">${g.temp}°C</div>
          ${g.power>0?`<div class="text-xs text-gray-500">${Math.round(g.power)}W${g.power_limit>0?' / '+Math.round(g.power_limit)+'W':''}</div>`:''}
        </div>
      </div>
      <div class="space-y-2">
        <div>
          <div class="flex justify-between text-xs text-gray-400 mb-1">
            <span>VRAM</span><span>${vp}%</span>
          </div>${bar(vp,'#6366f1')}
        </div>
        <div>
          <div class="flex justify-between text-xs text-gray-400 mb-1">
            <span>GPU Util</span><span>${g.util}%</span>
          </div>${bar(g.util, uc)}
        </div>
        ${pw>0?`<div>
          <div class="flex justify-between text-xs text-gray-400 mb-1">
            <span>Power</span><span>${pw}%</span>
          </div>${bar(pw,'#f59e0b')}
        </div>`:''}
      </div>
    </div>`;
  }).join('');
}

// ── Sys cards ──────────────────────────────────────────────────────────────
function renderSysCards(s) {
  document.getElementById('sys-cards').innerHTML = `
    <div class="card">
      <div class="text-xs text-gray-400 mb-1">CPU</div>
      <div class="text-2xl font-bold text-blue-400">${s.cpu_pct}%</div>
      ${bar(s.cpu_pct,'#3b82f6')}
    </div>
    <div class="card">
      <div class="text-xs text-gray-400 mb-1">RAM Used</div>
      <div class="text-2xl font-bold text-purple-400">${s.ram_pct}%</div>
      <div class="text-xs text-gray-500">${s.ram_used} / ${s.ram_total} MB</div>
      ${bar(s.ram_pct,'#a855f7')}
    </div>
    <div class="card">
      <div class="text-xs text-gray-400 mb-1">RAM Available</div>
      <div class="text-xl font-bold text-green-400">${s.ram_avail} MB</div>
    </div>
    <div class="card">
      <div class="text-xs text-gray-400 mb-1">RAM Total</div>
      <div class="text-xl font-bold text-gray-300">${s.ram_total} MB</div>
    </div>`;
}

// ── Process table (dashboard) ──────────────────────────────────────────────
function renderProcTable(procs) {
  if (!procs.length) {
    document.getElementById('proc-table').innerHTML =
      '<p class="text-gray-500 text-sm">PM2 không chạy hoặc chưa start app nào.</p>';
    return;
  }
  document.getElementById('proc-table').innerHTML = `
    <table class="w-full text-sm">
      <thead><tr class="text-xs text-gray-500 border-b border-gray-700">
        <th class="text-left pb-2">Name</th>
        <th class="text-left pb-2">Status</th>
        <th class="text-right pb-2">CPU</th>
        <th class="text-right pb-2">RAM</th>
        <th class="text-right pb-2">Restarts</th>
        <th class="text-right pb-2">Uptime</th>
        <th class="text-right pb-2">Actions</th>
      </tr></thead>
      <tbody>${procs.map(p=>`
        <tr class="border-b border-gray-700/50 hover:bg-gray-700/30">
          <td class="py-2 font-medium">${p.name}</td>
          <td class="py-2">${statusBadge(p.status)}</td>
          <td class="py-2 text-right text-gray-300">${p.cpu}%</td>
          <td class="py-2 text-right text-gray-300">${fmtBytes(p.memory)}</td>
          <td class="py-2 text-right ${p.restarts>3?'text-yellow-400':'text-gray-300'}">${p.restarts}</td>
          <td class="py-2 text-right text-gray-400">${fmtUptime(p.uptime)}</td>
          <td class="py-2 text-right">
            <button onclick="doRestart('${p.name}')" class="btn btn-primary text-xs mr-1">Restart</button>
            <button onclick="doStop('${p.name}')" class="btn btn-danger text-xs">Stop</button>
          </td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

// ── Process cards (processes tab) ─────────────────────────────────────────
function renderProcCards(procs) {
  if (!procs.length) {
    document.getElementById('proc-cards').innerHTML =
      '<div class="card text-gray-500">Không có PM2 process nào.</div>';
    return;
  }
  document.getElementById('proc-cards').innerHTML = procs.map(p => {
    const env = p.pm2_env?.env || {};
    const envItems = Object.entries(env)
      .filter(([k]) => k.startsWith('LLAMA_'))
      .map(([k,v]) => `<div class="flex gap-2"><span class="text-indigo-400">${k}</span><span class="text-gray-300">${v}</span></div>`)
      .join('');
    return `<div class="card">
      <div class="flex justify-between items-start mb-3">
        <div>
          <span class="font-semibold">${p.name}</span>
          <span class="ml-2 text-gray-500 text-xs">#${p.id}</span>
        </div>
        ${statusBadge(p.status)}
      </div>
      <div class="grid grid-cols-3 gap-2 text-center mb-3">
        <div class="bg-gray-700/50 rounded-lg p-2">
          <div class="text-lg font-bold text-blue-400">${p.cpu}%</div>
          <div class="text-xs text-gray-500">CPU</div>
        </div>
        <div class="bg-gray-700/50 rounded-lg p-2">
          <div class="text-lg font-bold text-purple-400">${fmtBytes(p.memory)}</div>
          <div class="text-xs text-gray-500">RAM</div>
        </div>
        <div class="bg-gray-700/50 rounded-lg p-2">
          <div class="text-lg font-bold ${p.restarts>3?'text-yellow-400':'text-gray-300'}">${p.restarts}</div>
          <div class="text-xs text-gray-500">Restarts</div>
        </div>
      </div>
      ${envItems ? `<div class="bg-gray-900/50 rounded-lg p-2 mb-3 text-xs font-mono space-y-0.5">${envItems}</div>` : ''}
      <div class="flex gap-2">
        <button onclick="doRestart('${p.name}')" class="btn btn-primary text-xs flex-1">&#8635; Restart</button>
        ${p.status==='online'
          ? `<button onclick="doStop('${p.name}')" class="btn btn-danger text-xs flex-1">&#9632; Stop</button>`
          : `<button onclick="doStart('${p.name}')" class="btn btn-green text-xs flex-1">&#9654; Start</button>`}
        <button onclick="viewLogs('${p.name}')" class="btn btn-gray text-xs">Logs</button>
      </div>
    </div>`;
  }).join('');
}

// ── Process actions ────────────────────────────────────────────────────────
async function doRestart(name) {
  toast('Restarting ' + name + '...');
  const r = await fetch(`/api/process/${name}/restart`, {method:'POST'}).then(r=>r.json());
  toast(r.ok ? name + ' restarted ✓' : 'Error: ' + r.err, r.ok?'ok':'error');
  setTimeout(refreshAll, 2000);
}
async function doStop(name) {
  if (!confirm('Stop ' + name + '?')) return;
  const r = await fetch(`/api/process/${name}/stop`, {method:'POST'}).then(r=>r.json());
  toast(r.ok ? name + ' stopped' : 'Error: ' + r.err, r.ok?'info':'error');
  setTimeout(refreshAll, 1000);
}
async function doStart(name) {
  toast('Starting ' + name + '...');
  const r = await fetch(`/api/process/${name}/start`, {method:'POST'}).then(r=>r.json());
  toast(r.ok ? name + ' started ✓' : 'Error: ' + r.err, r.ok?'ok':'error');
  setTimeout(refreshAll, 2000);
}

function viewLogs(name) {
  document.getElementById('log-app-select').value = name;
  showTab('logs');
}

// ── Workers ────────────────────────────────────────────────────────────────
async function loadWorkers() {
  const data = await fetch('/api/workers').then(r=>r.json()).catch(()=>null);
  const el = document.getElementById('workers-table');
  if (!data) { el.innerHTML = '<p class="text-gray-500 text-sm">Coordinator không phản hồi.</p>'; return; }
  const workers = data.workers || [];
  if (!workers.length) {
    el.innerHTML = '<p class="text-gray-500 text-sm">Chưa có worker nào. Single-VM mode.</p>';
    return;
  }
  el.innerHTML = `<table class="w-full text-sm">
    <thead><tr class="text-xs text-gray-500 border-b border-gray-700">
      <th class="text-left pb-2">Host</th>
      <th class="text-left pb-2">RPC Port</th>
      <th class="text-left pb-2">GPU</th>
      <th class="text-right pb-2">GPU Layers</th>
      <th class="text-left pb-2">Status</th>
      <th class="text-left pb-2">Registered</th>
      <th class="text-right pb-2">Actions</th>
    </tr></thead>
    <tbody>${workers.map(w=>`
      <tr class="border-b border-gray-700/50 hover:bg-gray-700/30">
        <td class="py-2 font-mono">${w.host}</td>
        <td class="py-2 font-mono">${w.rpc_port}</td>
        <td class="py-2 text-gray-300">${w.gpu||'-'}</td>
        <td class="py-2 text-right">${w.gpu_layers||0}</td>
        <td class="py-2">${w.healthy
          ? '<span class="badge-online px-2 py-0.5 rounded text-xs">healthy</span>'
          : '<span class="badge-erroring px-2 py-0.5 rounded text-xs">unreachable</span>'}</td>
        <td class="py-2 text-gray-400 text-xs">${w.registered_at||'-'}</td>
        <td class="py-2 text-right">
          <button onclick="removeWorker('${w.host}')"
            class="btn btn-danger text-xs">Remove</button>
        </td>
      </tr>`).join('')}
    </tbody>
  </table>`;
}

async function addWorker() {
  const host = document.getElementById('w-host').value.trim();
  const port = parseInt(document.getElementById('w-port').value);
  const gpu  = document.getElementById('w-gpu').value.trim();
  if (!host) { toast('Host không được để trống', 'error'); return; }
  const r = await fetch('/api/workers', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({host, rpc_port:port, gpu, gpu_layers:0})
  }).then(r=>r.json());
  toast(r.status==='ok' ? `Worker ${host} registered ✓` : (r.error||'Error'), r.status==='ok'?'ok':'error');
  setTimeout(loadWorkers, 500);
}

async function removeWorker(host) {
  if (!confirm(`Remove worker ${host}?`)) return;
  const r = await fetch(`/api/workers/${encodeURIComponent(host)}`, {method:'DELETE'}).then(r=>r.json());
  toast(r.removed ? `${host} removed` : 'Not found', r.removed?'ok':'error');
  setTimeout(loadWorkers, 500);
}

// ── Logs ───────────────────────────────────────────────────────────────────
function startLogStream() {
  if (logEventSource) logEventSource.close();
  const app = document.getElementById('log-app-select').value;
  document.getElementById('log-status').textContent = `Streaming ${app}...`;
  logEventSource = new EventSource(`/api/logs/${app}`);
  logEventSource.onmessage = e => {
    const d = JSON.parse(e.data);
    const filterErr = document.getElementById('log-filter-err').checked;
    const isErr = d.line.includes('ERROR') || d.line.includes('error') || d.line.includes('Error');
    const isWarn = d.line.includes('WARN') || d.line.includes('warn');
    if (filterErr && !isErr) return;

    logLines.push(d);
    if (logLines.length > 2000) logLines.shift();

    const el = document.getElementById('log-box');
    const div = document.createElement('div');
    div.className = isErr ? 'log-err' : isWarn ? 'log-warn' : 'log-info';
    const ts = new Date(d.ts*1000).toLocaleTimeString();
    div.textContent = `[${ts}] ${d.line}`;
    el.appendChild(div);
    if (el.children.length > 1000) el.removeChild(el.firstChild);
    if (document.getElementById('log-autoscroll').checked) el.scrollTop = el.scrollHeight;
  };
  logEventSource.onerror = () => {
    document.getElementById('log-status').textContent = 'Stream disconnected';
  };
}
function clearLogs() {
  logLines = [];
  document.getElementById('log-box').innerHTML = '';
}

// ── Chat ───────────────────────────────────────────────────────────────────
function appendMsg(role, content) {
  const el = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `p-3 text-sm ${role==='user'?'chat-msg-user ml-8':'chat-msg-ai mr-8'}`;
  if (role === 'user') {
    div.innerHTML = `<div class="text-xs text-indigo-300 mb-1">You</div><div class="whitespace-pre-wrap">${content}</div>`;
  } else {
    div.innerHTML = `<div class="text-xs text-gray-500 mb-1">Assistant</div><div class="whitespace-pre-wrap" id="ai-streaming-${Date.now()}">${content}</div>`;
  }
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  return div.querySelector('[id^=ai-streaming]') || div;
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  chatMessages.push({role:'user', content:text});
  appendMsg('user', text);

  const btn = document.getElementById('chat-send-btn');
  btn.disabled = true;
  btn.textContent = '...';

  const temp = parseFloat(document.getElementById('chat-temp').value);
  const maxTok = parseInt(document.getElementById('chat-max-tokens').value);

  const t0 = Date.now();
  let tokenCount = 0;
  const aiEl = appendMsg('assistant', '');

  try {
    const resp = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        messages: chatMessages,
        stream: true,
        temperature: temp,
        max_tokens: maxTok,
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let fullText = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') continue;
        try {
          const d = JSON.parse(raw);
          const delta = d.choices?.[0]?.delta?.content || '';
          if (delta) {
            fullText += delta;
            tokenCount++;
            aiEl.textContent = fullText;
            document.getElementById('chat-messages').scrollTop = 9999;
          }
        } catch {}
      }
    }
    const tps = tokenCount / ((Date.now()-t0)/1000);
    document.getElementById('chat-tps').textContent = `${tps.toFixed(1)} tok/s`;
    chatMessages.push({role:'assistant', content:fullText});
  } catch(e) {
    aiEl.textContent = 'Error: ' + e.message;
    aiEl.className += ' text-red-400';
  }

  btn.disabled = false;
  btn.textContent = 'Send';
}

function clearChat() {
  chatMessages = [];
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('chat-tps').textContent = '';
}

// ── Metrics ────────────────────────────────────────────────────────────────
async function loadMetrics() {
  const data = await fetch('/api/metrics').then(r=>r.json()).catch(()=>null);
  if (!data) return;

  const coord = data.coordinator;
  const mc = document.getElementById('metric-cards');
  if (coord) {
    const be = coord.backends?.[0];
    const wo = coord.workers;
    mc.innerHTML = `
      <div class="card">
        <div class="text-xs text-gray-400 mb-1">Active Requests</div>
        <div class="text-3xl font-bold text-blue-400">${be?.active_requests??'-'}</div>
        <div class="text-xs text-gray-500 mt-1">Total: ${be?.total_requests??0}</div>
      </div>
      <div class="card">
        <div class="text-xs text-gray-400 mb-1">Avg Latency</div>
        <div class="text-3xl font-bold text-purple-400">${be?.avg_latency_ms?.toFixed(0)??'-'}<span class="text-base text-gray-500">ms</span></div>
        <div class="text-xs text-gray-500 mt-1">Errors: ${be?.total_errors??0}</div>
      </div>
      <div class="card">
        <div class="text-xs text-gray-400 mb-1">Workers</div>
        <div class="text-3xl font-bold text-green-400">${wo?.healthy??0}<span class="text-base text-gray-500"> / ${wo?.total??0}</span></div>
        <div class="text-xs text-gray-500 mt-1">Prefix cache: ${be?.prefix_cache_entries??0} entries</div>
      </div>`;
  }

  // Render raw prometheus metrics
  if (data.llama_metrics) {
    document.getElementById('raw-metrics').textContent =
      typeof data.llama_metrics === 'string'
        ? data.llama_metrics
        : JSON.stringify(data.llama_metrics, null, 2);
  }

  renderVramChart();
}

function renderVramChart() {
  const canvas = document.getElementById('vram-chart');
  if (!canvas || !vramHistory.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 600;
  const H = canvas.height || 120;
  canvas.width = W;
  ctx.clearRect(0, 0, W, H);

  const total = vramHistory[0]?.total || 1;
  const pad = 30;
  const w = W - pad * 2;
  const h = H - pad;

  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    ctx.fillStyle = '#475569';
    ctx.font = '10px monospace';
    ctx.fillText(Math.round(total * (1 - i/4)) + 'M', 0, y + 4);
  }

  if (vramHistory.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 2;
  vramHistory.forEach((p, i) => {
    const x = pad + (i / (MAX_VRAM_HISTORY - 1)) * w;
    const y = pad + h - (p.used / total) * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill
  ctx.lineTo(pad + ((vramHistory.length-1)/(MAX_VRAM_HISTORY-1))*w, pad+h);
  ctx.lineTo(pad, pad+h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(99,102,241,0.15)';
  ctx.fill();
}

// ── Init ───────────────────────────────────────────────────────────────────
showTab('dashboard');
refreshAll();
refreshInterval = setInterval(() => {
  if (['dashboard','processes','metrics'].includes(currentTab)) refreshAll();
}, 5000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp WebUI — quản lý cluster qua trình duyệt"
    )
    parser.add_argument("--port",        type=int, default=DEFAULT_PORT,
                        help=f"Port cho WebUI (default: {DEFAULT_PORT})")
    parser.add_argument("--host",        default="127.0.0.1",
                        help="Host bind (default: 127.0.0.1 — chỉ local)")
    parser.add_argument("--coordinator", default=DEFAULT_COORDINATOR,
                        help=f"URL coordinator (default: {DEFAULT_COORDINATOR})")
    parser.add_argument("--llama-server", default=DEFAULT_LLAMA, dest="llama",
                        help=f"URL llama-server (default: {DEFAULT_LLAMA})")
    args = parser.parse_args()

    CFG["coordinator"] = args.coordinator.rstrip("/")
    CFG["llama"]       = args.llama.rstrip("/")

    print("=" * 60)
    print("  llama.cpp WebUI")
    print(f"  URL          : http://{args.host}:{args.port}")
    print(f"  Coordinator  : {CFG['coordinator']}")
    print(f"  llama-server : {CFG['llama']}")
    print("=" * 60)
    print(f"\n  Mở trình duyệt: http://localhost:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
