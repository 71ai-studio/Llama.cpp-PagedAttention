#!/usr/bin/env bash
# deploy.sh — Auto pull, build, và khởi động node (Ubuntu/macOS, main/worker)
# ============================================================================
# Mỗi máy trong cluster chạy script này 1 lần để setup và start service.
#
# Usage:
#   ./deploy.sh [OPTIONS]
#
# Options:
#   --config FILE    Path tới nodes.json  (default: run/nodes.json)
#   --self   IP      IP của máy này       (default: auto-detect)
#   --pull           Git pull trước khi build
#   --rebuild        Xoá build dir và build lại từ đầu
#   --no-service     Không install systemd/launchd service, chỉ run foreground
#   --dry-run        In lệnh nhưng không thực thi
#   --help
#
# Ví dụ:
#   # Máy main (192.168.1.10):
#   ./deploy.sh --config run/nodes.json --self 192.168.1.10 --pull
#
#   # Worker GPU (192.168.1.11):
#   ./deploy.sh --config run/nodes.json --self 192.168.1.11
#
#   # CPU-only worker:
#   ./deploy.sh --config run/nodes.json --self 192.168.1.12
#
# Cluster topology (run/nodes.json):
#   Main node:   llama-server + RPC client → kết nối các worker
#   Worker node: llama-rpc-server → cung cấp GPU backend qua mạng
# ============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$REPO_DIR/run/nodes.json"
SELF_IP=""
DO_PULL=false
DO_REBUILD=false
NO_SERVICE=false
DRY_RUN=false

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG_FILE="$2"; shift 2 ;;
        --self)      SELF_IP="$2";     shift 2 ;;
        --pull)      DO_PULL=true;     shift   ;;
        --rebuild)   DO_REBUILD=true;  shift   ;;
        --no-service) NO_SERVICE=true; shift   ;;
        --dry-run)   DRY_RUN=true;     shift   ;;
        --help|-h)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
run() {
    echo "  $ $*"
    $DRY_RUN && return 0
    "$@"
}

log()  { echo ""; echo "══ $* ══"; }
info() { echo "   $*"; }
warn() { echo "   ⚠  $*"; }
die()  { echo "   ✗  $*" >&2; exit 1; }

# ── Detect OS / Platform ─────────────────────────────────────────────────────
log "Platform detection"
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin)
        PLATFORM="macos"
        CMAKE_BACKEND="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
        info "macOS $ARCH — Metal backend"
        ;;
    Linux)
        if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ]; then
            PLATFORM="linux-cuda"
            export PATH="/usr/local/cuda/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'V\K[\d.]+' | head -1 || echo "?")
            CMAKE_BACKEND="-DGGML_CUDA=ON -DGGML_CUDA_FA=ON"
            info "Linux $ARCH — CUDA $CUDA_VER"
        else
            PLATFORM="linux-cpu"
            CMAKE_BACKEND=""
            warn "Linux $ARCH — CPU-only (CUDA không tìm thấy)"
        fi
        ;;
    *) die "Unsupported OS: $OS" ;;
esac

JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── Self IP ───────────────────────────────────────────────────────────────────
if [ -z "$SELF_IP" ]; then
    if [ "$OS" = "Darwin" ]; then
        SELF_IP=$(ipconfig getifaddr en0 2>/dev/null || \
                  ipconfig getifaddr en1 2>/dev/null || \
                  hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    else
        SELF_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    fi
    info "Auto-detected IP: $SELF_IP"
fi

# ── Read cluster config ───────────────────────────────────────────────────────
log "Reading cluster config"
[ -f "$CONFIG_FILE" ] || die "Config không tìm thấy: $CONFIG_FILE"
info "Config: $CONFIG_FILE"

# Dùng python3 để parse JSON (không cần jq)
read_json() {
    python3 -c "
import json, sys
data = json.load(open('$CONFIG_FILE'))
$1
"
}

NODE_COUNT=$(read_json "print(len(data.get('nodes', [])))")
info "Nodes trong cluster: $NODE_COUNT"

# Tìm config cho node hiện tại (match theo host)
NODE_JSON=$(python3 -c "
import json, sys
data = json.load(open('$CONFIG_FILE'))
self_ip = '$SELF_IP'
for n in data.get('nodes', []):
    if n.get('host') == self_ip or n.get('host') in ('localhost', '127.0.0.1'):
        print(json.dumps(n))
        sys.exit(0)
print('')
" 2>/dev/null || echo "")

if [ -z "$NODE_JSON" ]; then
    die "Không tìm thấy cấu hình cho IP $SELF_IP trong $CONFIG_FILE\nCác host: $(read_json "print(' '.join(n.get('host','') for n in data.get('nodes',[])) )")"
fi

NODE_ROLE=$(python3    -c "import json; d=json.loads('$NODE_JSON'); print(d.get('role','main'))")
NODE_GPU_LAYERS=$(python3 -c "import json; d=json.loads('$NODE_JSON'); print(d.get('gpu_layers', 0))")
NODE_PORT=$(python3    -c "import json; d=json.loads('$NODE_JSON'); print(d.get('port', 11434))")
NODE_RPC_PORT=$(python3 -c "import json; d=json.loads('$NODE_JSON'); print(d.get('rpc_port', 50052))")
MODEL_FILE=$(read_json  "print(data.get('model', 'model.gguf'))")
MODEL_PATH="$REPO_DIR/models/$MODEL_FILE"
CTX_SIZE=$(read_json    "print(data.get('context', 4096))")
KV_PAGE_SIZE=$(read_json "print(data.get('kv_page_size', 16))")
N_PARALLEL=$(read_json  "print(data.get('n_parallel', 4))")

info "Role     : $NODE_ROLE"
info "IP       : $SELF_IP"
info "GPU layers: $NODE_GPU_LAYERS"
[ "$NODE_ROLE" = "main"   ] && info "Port     : $NODE_PORT"
[ "$NODE_ROLE" = "worker" ] && info "RPC port : $NODE_RPC_PORT"

# ── RPC worker list (for main node) ──────────────────────────────────────────
if [ "$NODE_ROLE" = "main" ]; then
    RPC_LIST=$(python3 -c "
import json
data = json.load(open('$CONFIG_FILE'))
workers = [n for n in data.get('nodes', []) if n.get('role') == 'worker']
parts = ['{}:{}'.format(n['host'], n.get('rpc_port', 50052)) for n in workers]
print(','.join(parts))
")
    if [ -n "$RPC_LIST" ]; then
        info "RPC workers: $RPC_LIST"
    else
        info "RPC workers: none (single-machine mode)"
    fi
fi

# ── Source dir ───────────────────────────────────────────────────────────────
SRC_DIR="$REPO_DIR/llama-cpp-python/vendor/llama.cpp"
BUILD_DIR="$SRC_DIR/build"
BIN_DIR="$BUILD_DIR/bin"

# ── Git pull ─────────────────────────────────────────────────────────────────
if $DO_PULL; then
    log "Git pull"
    run git -C "$REPO_DIR" pull --ff-only
fi

# ── Build ─────────────────────────────────────────────────────────────────────
log "Build"

if $DO_REBUILD && [ -d "$BUILD_DIR" ]; then
    info "Xoá build dir cũ..."
    run rm -rf "$BUILD_DIR"
fi

run mkdir -p "$BUILD_DIR"
run cmake "$SRC_DIR" -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    $CMAKE_BACKEND \
    -DGGML_RPC=ON \
    -DLLAMA_BUILD_SERVER=ON

run cmake --build "$BUILD_DIR" -j"$JOBS"

# Verify binaries
[ -f "$BIN_DIR/llama-server" ]     || die "llama-server không build được"
[ -f "$BIN_DIR/llama-rpc-server" ] || die "llama-rpc-server không build được — kiểm tra GGML_RPC=ON"
info "Build OK → $BIN_DIR"

# ── Detect GPU layers (nếu không được set trong config) ──────────────────────
if [ "$NODE_GPU_LAYERS" -eq 0 ] && command -v python3 &>/dev/null && [ -f "$MODEL_PATH" ]; then
    NODE_GPU_LAYERS=$(python3 "$REPO_DIR/run/auto_ngl.py" "$MODEL_PATH" 1500 2>/dev/null || echo 0)
    info "Auto NGL: $NODE_GPU_LAYERS layers"
fi

# ── Build systemd / launchd service ──────────────────────────────────────────

SERVICE_NAME="llama-$(echo "$NODE_ROLE" | tr '[:upper:]' '[:lower:]')-${SELF_IP//./-}"

write_systemd_unit() {
    local unit_file="$1"
    local exec_cmd="$2"

    cat > "$unit_file" <<UNIT
[Unit]
Description=llama.cpp $NODE_ROLE node ($SELF_IP)
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$exec_cmd
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=LD_LIBRARY_PATH=$BIN_DIR

[Install]
WantedBy=multi-user.target
UNIT
    echo "$unit_file"
}

write_launchd_plist() {
    local plist_file="$1"
    shift
    local -a args=("$@")
    local args_xml=""
    for a in "${args[@]}"; do
        args_xml+="        <string>$a</string>\n"
    done
    cat > "$plist_file" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>             <string>com.vuna.$SERVICE_NAME</string>
    <key>ProgramArguments</key>
    <array>
$(printf '%s\n' "${args[@]}" | sed 's/^/        <string>/' | sed 's/$/<\/string>/')
    </array>
    <key>WorkingDirectory</key>  <string>$REPO_DIR</string>
    <key>RunAtLoad</key>         <true/>
    <key>KeepAlive</key>         <true/>
    <key>StandardOutPath</key>   <string>/tmp/$SERVICE_NAME.log</string>
    <key>StandardErrorPath</key> <string>/tmp/$SERVICE_NAME.err</string>
</dict>
</plist>
PLIST
    echo "$plist_file"
}

# ── Build command & start ─────────────────────────────────────────────────────
log "Configuring $NODE_ROLE service"

if [ "$NODE_ROLE" = "worker" ]; then
    # ── RPC Worker ────────────────────────────────────────────────────────────
    info "Mode: RPC worker — cung cấp GPU backend cho main node"
    info "Listening: 0.0.0.0:$NODE_RPC_PORT"

    CMD_ARGS=(
        "$BIN_DIR/llama-rpc-server"
        --host "0.0.0.0"
        --port "$NODE_RPC_PORT"
    )

    if ! $NO_SERVICE; then
        if [ "$OS" = "Linux" ]; then
            UNIT_FILE="/tmp/${SERVICE_NAME}.service"
            write_systemd_unit "$UNIT_FILE" "${CMD_ARGS[*]}" >/dev/null
            if [ -w /etc/systemd/system ]; then
                run cp "$UNIT_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
                run systemctl daemon-reload
                run systemctl enable  "$SERVICE_NAME"
                run systemctl restart "$SERVICE_NAME"
                info "Systemd service: $SERVICE_NAME"
                info "Xem log: journalctl -fu $SERVICE_NAME"
            else
                warn "Không có quyền write /etc/systemd/system — chạy foreground"
                NO_SERVICE=true
            fi
        elif [ "$OS" = "Darwin" ]; then
            PLIST_FILE="$HOME/Library/LaunchAgents/com.vuna.${SERVICE_NAME}.plist"
            write_launchd_plist "$PLIST_FILE" "${CMD_ARGS[@]}" >/dev/null
            run launchctl unload "$PLIST_FILE" 2>/dev/null || true
            run launchctl load   "$PLIST_FILE"
            info "LaunchAgent: $PLIST_FILE"
            info "Xem log: tail -f /tmp/$SERVICE_NAME.log"
        fi
    fi

    if $NO_SERVICE; then
        log "Starting RPC worker (foreground)"
        run "${CMD_ARGS[@]}"
    fi

else
    # ── Main server ───────────────────────────────────────────────────────────
    [ -f "$MODEL_PATH" ] || die "Model không tìm thấy: $MODEL_PATH\nChạy: ./run/download_models.sh"
    info "Mode: Main server — llama-server"
    info "Model: $MODEL_FILE"

    CMD_ARGS=(
        "$BIN_DIR/llama-server"
        --model        "$MODEL_PATH"
        --host         "0.0.0.0"
        --port         "$NODE_PORT"
        --ctx-size     "$CTX_SIZE"
        --n-gpu-layers "$NODE_GPU_LAYERS"
        --parallel     "$N_PARALLEL"
        --flash-attn
        --kv-page-size "$KV_PAGE_SIZE"
        --cont-batching
        --metrics
    )

    if [ -n "${RPC_LIST:-}" ]; then
        CMD_ARGS+=(--rpc "$RPC_LIST")
        info "RPC backends: $RPC_LIST"
    fi

    if ! $NO_SERVICE; then
        if [ "$OS" = "Linux" ]; then
            UNIT_FILE="/tmp/${SERVICE_NAME}.service"
            write_systemd_unit "$UNIT_FILE" "${CMD_ARGS[*]}" >/dev/null
            if [ -w /etc/systemd/system ]; then
                run cp "$UNIT_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
                run systemctl daemon-reload
                run systemctl enable  "$SERVICE_NAME"
                run systemctl restart "$SERVICE_NAME"
                info "Systemd service: $SERVICE_NAME"
                info "Xem log: journalctl -fu $SERVICE_NAME"
                info "API: http://$SELF_IP:$NODE_PORT/v1/chat/completions"
            else
                warn "Không có quyền write /etc/systemd/system — chạy foreground"
                NO_SERVICE=true
            fi
        elif [ "$OS" = "Darwin" ]; then
            PLIST_FILE="$HOME/Library/LaunchAgents/com.vuna.${SERVICE_NAME}.plist"
            write_launchd_plist "$PLIST_FILE" "${CMD_ARGS[@]}" >/dev/null
            run launchctl unload "$PLIST_FILE" 2>/dev/null || true
            run launchctl load   "$PLIST_FILE"
            info "LaunchAgent: $PLIST_FILE"
            info "Xem log: tail -f /tmp/$SERVICE_NAME.log"
            info "API: http://$SELF_IP:$NODE_PORT/v1/chat/completions"
        fi
    fi

    if $NO_SERVICE; then
        log "Starting main server (foreground)"
        run "${CMD_ARGS[@]}"
    fi
fi

log "Done"
echo ""
echo "   Cluster config : $CONFIG_FILE"
echo "   Role           : $NODE_ROLE"
echo "   Platform       : $PLATFORM"
echo ""
