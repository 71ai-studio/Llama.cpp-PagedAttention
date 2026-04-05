#!/usr/bin/env python3
"""
cluster.py — Orchestrate multi-machine llama.cpp cluster
==========================================================
Gọi deploy.sh trên từng node qua SSH để pull/build/start.

Usage:
  python3 cluster.py start   [--config nodes.json] [--pull] [--rebuild]
  python3 cluster.py stop    [--config nodes.json]
  python3 cluster.py status  [--config nodes.json]
  python3 cluster.py logs    [--config nodes.json] [--node HOST]
  python3 cluster.py deploy  [--config nodes.json] [--pull] [--rebuild] [--node HOST]

Options:
  --config FILE   Path tới nodes.json (default: run/nodes.json)
  --node HOST     Chỉ áp dụng lệnh lên node có host này
  --pull          Truyền --pull vào deploy.sh
  --rebuild       Truyền --rebuild vào deploy.sh (build lại từ đầu)

Flow:
  start:
    1. Deploy workers song song (build + start rpc-server)
    2. Deploy main node sau (cần workers up trước)
    3. Health check tất cả nodes

  stop:
    1. Stop main trước (không nhận request mới)
    2. Stop workers

  status:
    - Kiểm tra health tất cả nodes (HTTP hoặc TCP)
    - In bảng tóm tắt
"""

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Node:
    role:       str
    host:       str
    ssh_user:   str
    gpu_layers: int
    port:       Optional[int]
    rpc_port:   Optional[int]
    platform:   str = ""
    gpu:        str = ""
    # runtime state
    last_status: str = "unknown"
    last_latency_ms: float = 0.0


@dataclass
class Cluster:
    name:         str
    model:        str
    context:      int
    kv_page_size: int
    n_parallel:   int
    nodes:        list[Node] = field(default_factory=list)

    @property
    def main_nodes(self):
        return [n for n in self.nodes if n.role == "main"]

    @property
    def worker_nodes(self):
        return [n for n in self.nodes if n.role == "worker"]


def load_cluster(config_path: str) -> Cluster:
    with open(config_path) as f:
        data = json.load(f)

    nodes = []
    for n in data.get("nodes", []):
        nodes.append(Node(
            role       = n.get("role",       "main"),
            host       = n.get("host",       "127.0.0.1"),
            ssh_user   = n.get("ssh_user",   "ubuntu"),
            gpu_layers = n.get("gpu_layers", 0),
            port       = n.get("port"),
            rpc_port   = n.get("rpc_port"),
            platform   = n.get("platform",   ""),
            gpu        = n.get("gpu",        ""),
        ))

    return Cluster(
        name         = data.get("cluster_name", "cluster"),
        model        = data.get("model",        "model.gguf"),
        context      = data.get("context",      4096),
        kv_page_size = data.get("kv_page_size", 16),
        n_parallel   = data.get("n_parallel",   4),
        nodes        = nodes,
    )


# ── SSH helpers ───────────────────────────────────────────────────────────────

REPO_DIR = Path(__file__).parent.parent.resolve()


def ssh_cmd(node: Node, remote_cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command on the remote node via SSH."""
    ssh = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{node.ssh_user}@{node.host}",
        remote_cmd,
    ]
    return subprocess.run(
        ssh,
        capture_output=capture,
        text=True,
    )


def is_local(node: Node) -> bool:
    """True nếu node là localhost."""
    import socket
    try:
        local_ips = {ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]}
        local_ips.add("127.0.0.1")
        return node.host in local_ips or node.host == "localhost"
    except Exception:
        return node.host in ("127.0.0.1", "localhost")


def run_on_node(node: Node, cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
    """Run command locally or via SSH depending on node.host."""
    if is_local(node):
        return subprocess.run(
            cmd, shell=True, capture_output=capture, text=True
        )
    else:
        return ssh_cmd(node, cmd, capture=capture)


# ── Deploy ────────────────────────────────────────────────────────────────────

def deploy_node(node: Node, config_path: str,
                pull: bool = False, rebuild: bool = False,
                verbose: bool = False) -> tuple[bool, str]:
    """Deploy và start một node. Returns (success, output)."""
    # Path của deploy.sh trên remote machine (assume cùng repo path)
    deploy_script = str(REPO_DIR / "deploy.sh")

    flags = [f"--config {shlex.quote(config_path)}",
             f"--self {node.host}"]
    if pull:    flags.append("--pull")
    if rebuild: flags.append("--rebuild")

    cmd = f"bash {shlex.quote(deploy_script)} {' '.join(flags)}"

    print(f"  [{node.host}] {cmd}")
    result = run_on_node(node, cmd, capture=not verbose)

    if result.returncode != 0:
        err = result.stderr if result.stderr else result.stdout
        return False, f"FAILED (exit {result.returncode}): {err[:300]}"

    return True, result.stdout[-300:] if result.stdout else "OK"


def deploy_nodes_parallel(nodes: list[Node], config_path: str,
                          pull: bool, rebuild: bool) -> dict[str, tuple[bool, str]]:
    """Deploy nhiều nodes song song."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = {
            pool.submit(deploy_node, n, config_path, pull, rebuild): n
            for n in nodes
        }
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                ok, msg = future.result()
            except Exception as e:
                ok, msg = False, str(e)
            results[node.host] = (ok, msg)
            status = "✓" if ok else "✗"
            print(f"  {status} {node.host} ({node.role})")
    return results


# ── Health check ──────────────────────────────────────────────────────────────

def check_http(host: str, port: int, path: str = "/health",
               timeout: float = 3.0) -> tuple[bool, float]:
    """HTTP health check. Returns (ok, latency_ms)."""
    import urllib.request
    import urllib.error
    url = f"http://{host}:{port}{path}"
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            latency = (time.monotonic() - t0) * 1000
            return resp.status == 200, latency
    except Exception:
        return False, 0.0


def check_tcp(host: str, port: int, timeout: float = 3.0) -> tuple[bool, float]:
    """TCP reachability check (for RPC workers)."""
    import socket
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, (time.monotonic() - t0) * 1000
    except Exception:
        return False, 0.0


def status_node(node: Node) -> tuple[bool, float, str]:
    """Check node health. Returns (ok, latency_ms, detail)."""
    if node.role == "main" and node.port:
        ok, ms = check_http(node.host, node.port)
        return ok, ms, f"HTTP :{node.port}"
    elif node.rpc_port:
        ok, ms = check_tcp(node.host, node.rpc_port)
        return ok, ms, f"TCP  :{node.rpc_port}"
    return False, 0.0, "no port"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_start(cluster: Cluster, config_path: str, pull: bool, rebuild: bool,
              node_filter: Optional[str]) -> int:
    nodes = cluster.nodes
    if node_filter:
        nodes = [n for n in nodes if n.host == node_filter]
    workers = [n for n in nodes if n.role == "worker"]
    mains   = [n for n in nodes if n.role == "main"]

    print(f"\n═ Cluster: {cluster.name} ═══════════════════════════════════")
    print(f"  Model: {cluster.model}  ctx={cluster.context}  kv_page={cluster.kv_page_size}")

    all_ok = True

    # 1) Workers song song
    if workers:
        print(f"\n[1/2] Deploying {len(workers)} worker(s) in parallel...")
        results = deploy_nodes_parallel(workers, config_path, pull, rebuild)
        for host, (ok, _) in results.items():
            if not ok: all_ok = False

        # Chờ workers up (tối đa 60s)
        print("\n  Chờ RPC workers sẵn sàng (tối đa 60s)...")
        deadline = time.monotonic() + 60
        for w in workers:
            if w.rpc_port:
                while time.monotonic() < deadline:
                    ok, _ = check_tcp(w.host, w.rpc_port, timeout=2.0)
                    if ok:
                        print(f"  ✓ {w.host}:{w.rpc_port} — up")
                        break
                    time.sleep(2)
                else:
                    print(f"  ✗ {w.host}:{w.rpc_port} — timeout")
                    all_ok = False

    # 2) Main node sau
    if mains:
        print(f"\n[2/2] Deploying {len(mains)} main node(s)...")
        results = deploy_nodes_parallel(mains, config_path, pull, rebuild)
        for host, (ok, _) in results.items():
            if not ok: all_ok = False

    # 3) Final status
    time.sleep(3)
    print("")
    cmd_status(cluster, node_filter=node_filter)
    return 0 if all_ok else 1


def cmd_stop(cluster: Cluster, node_filter: Optional[str]) -> int:
    nodes = cluster.nodes
    if node_filter:
        nodes = [n for n in nodes if n.host == node_filter]

    print(f"\n═ Stopping cluster: {cluster.name} ═══════════")

    # Stop main trước
    for node in [n for n in nodes if n.role == "main"]:
        print(f"  Stopping main {node.host}...")
        svc = f"llama-main-{node.host.replace('.', '-')}"
        run_on_node(node, f"systemctl stop {svc} 2>/dev/null || launchctl remove com.vuna.{svc} 2>/dev/null || pkill -f llama-server || true")

    # Rồi workers
    for node in [n for n in nodes if n.role == "worker"]:
        print(f"  Stopping worker {node.host}...")
        svc = f"llama-worker-{node.host.replace('.', '-')}"
        run_on_node(node, f"systemctl stop {svc} 2>/dev/null || launchctl remove com.vuna.{svc} 2>/dev/null || pkill -f llama-rpc-server || true")

    print("  Done.")
    return 0


def cmd_status(cluster: Cluster, node_filter: Optional[str] = None) -> int:
    nodes = cluster.nodes
    if node_filter:
        nodes = [n for n in nodes if n.host == node_filter]

    print(f"\n═ Status: {cluster.name} ═══════════════════════════════════")
    print(f"  {'Host':<16} {'Role':<8} {'GPU':<18} {'Check':<12} {'Latency':>9}  Status")
    print(f"  {'─'*16} {'─'*8} {'─'*18} {'─'*12} {'─'*9}  {'─'*10}")

    def check_one(node: Node):
        ok, ms, chk = status_node(node)
        return node, ok, ms, chk

    with concurrent.futures.ThreadPoolExecutor() as pool:
        for node, ok, ms, chk in pool.map(check_one, nodes):
            sym   = "✓" if ok else "✗"
            ms_s  = f"{ms:.0f}ms" if ok else "—"
            gpu_s = (node.gpu or f"{node.gpu_layers}L")[:18]
            status_s = "UP" if ok else "DOWN"
            print(f"  {node.host:<16} {node.role:<8} {gpu_s:<18} {chk:<12} {ms_s:>9}  {sym} {status_s}")

    print("")
    return 0


def cmd_logs(cluster: Cluster, node_filter: Optional[str]) -> int:
    nodes = cluster.nodes
    if node_filter:
        nodes = [n for n in nodes if n.host == node_filter]

    for node in nodes:
        print(f"\n─── {node.host} ({node.role}) ────────────────────────")
        svc = f"llama-{node.role}-{node.host.replace('.', '-')}"
        # Linux: journalctl; macOS: log file
        cmd = (f"journalctl -u {svc} -n 30 --no-pager 2>/dev/null || "
               f"tail -30 /tmp/{svc}.log 2>/dev/null || "
               f"echo '[no log found for {svc}]'")
        result = run_on_node(node, cmd, capture=True)
        print(result.stdout or result.stderr or "(empty)")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cluster manager cho distributed llama.cpp (RPC backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["start", "stop", "status", "logs", "deploy"])
    parser.add_argument("--config",  default=str(Path(__file__).parent / "nodes.json"),
                        help="Path tới nodes.json")
    parser.add_argument("--node",    metavar="HOST", help="Chỉ áp dụng cho node này")
    parser.add_argument("--pull",    action="store_true", help="Git pull trên mỗi node")
    parser.add_argument("--rebuild", action="store_true", help="Build lại từ đầu")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config không tìm thấy: {args.config}", file=sys.stderr)
        sys.exit(1)

    cluster = load_cluster(args.config)

    if args.command == "start":
        sys.exit(cmd_start(cluster, args.config, args.pull, args.rebuild, args.node))

    elif args.command == "deploy":
        sys.exit(cmd_start(cluster, args.config, args.pull, args.rebuild, args.node))

    elif args.command == "stop":
        sys.exit(cmd_stop(cluster, args.node))

    elif args.command == "status":
        sys.exit(cmd_status(cluster, args.node))

    elif args.command == "logs":
        sys.exit(cmd_logs(cluster, args.node))


if __name__ == "__main__":
    main()
