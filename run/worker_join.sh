#!/usr/bin/env bash
# worker_join.sh — Chạy trên VM mới để join cluster tự động
#
# Workflow:
#   1. Query coordinator /api/info → lấy model_file cần tải
#   2. Kiểm tra model ở local; nếu chưa có → tải về
#   3. Detect GPU / tính NGL
#   4. Khởi động llama-rpc-server (compute backend, không cần model)
#   5. POST /api/workers/register → coordinator nhận → restart llama-server
#      với --rpc vm_này:PORT
#
# Lưu ý: llama-rpc-server là compute backend thuần tuý — KHÔNG load model.
# Model chỉ cần thiết nếu sau này VM này cũng chạy llama-server độc lập.
#
# Usage:
#   ./worker_join.sh --coordinator http://192.168.1.10:11433
#   ./worker_join.sh --coordinator http://vm1:11433 --rpc-port 50052
#   ./worker_join.sh --coordinator http://vm1:11433 --also-download-model
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
BIN_DIR="$SCRIPT_DIR/../llama-cpp-python/vendor/llama.cpp/build/bin"

# ── Parse arguments ───────────────────────────────────────────────────────
COORDINATOR_URL=""
RPC_PORT=50052
RPC_HOST=""          # public IP/hostname của VM này (auto-detect nếu để trống)
ALSO_DOWNLOAD=false  # --also-download-model: tải model về để chạy standalone

while [[ $# -gt 0 ]]; do
    case "$1" in
        --coordinator)    COORDINATOR_URL="$2";  shift 2 ;;
        --rpc-port)       RPC_PORT="$2";          shift 2 ;;
        --host)           RPC_HOST="$2";          shift 2 ;;
        --also-download-model) ALSO_DOWNLOAD=true; shift ;;
        -h|--help)
            echo "Usage: $0 --coordinator URL [--rpc-port PORT] [--host IP] [--also-download-model]"
            echo ""
            echo "  --coordinator URL         URL của coordinator VM chính (bắt buộc)"
            echo "  --rpc-port PORT           Port cho llama-rpc-server (default: 50052)"
            echo "  --host IP                 Public IP/hostname của VM này (auto-detect nếu bỏ)"
            echo "  --also-download-model     Tải model về để VM này có thể chạy standalone"
            exit 0
            ;;
        *) echo "[worker_join] Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$COORDINATOR_URL" ]; then
    echo "[worker_join] ERROR: --coordinator URL là bắt buộc"
    echo "  Ví dụ: $0 --coordinator http://192.168.1.10:11433"
    exit 1
fi

COORDINATOR_URL="${COORDINATOR_URL%/}"  # strip trailing slash

# ── Load hardware detection ───────────────────────────────────────────────
source "$SCRIPT_DIR/detect_hw.sh"

# ── Auto-detect public IP nếu chưa set ───────────────────────────────────
if [ -z "$RPC_HOST" ]; then
    # Ưu tiên: IP của interface có route ra ngoài
    RPC_HOST=$(python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
finally:
    s.close()
" 2>/dev/null || hostname -I | awk '{print $1}')
fi

echo "========================================================"
echo "  worker_join.sh"
echo "  Coordinator : $COORDINATOR_URL"
echo "  This VM IP  : $RPC_HOST"
echo "  RPC Port    : $RPC_PORT"
echo "  GPU type    : $GPU_TYPE"
echo "  VRAM        : ${VRAM_FREE_MB:-0} MB free"
echo "========================================================"
echo ""

# ── Query coordinator info ────────────────────────────────────────────────
echo "[worker_join] Lấy thông tin từ coordinator..."
INFO_JSON=$(curl -sf "${COORDINATOR_URL}/api/info" 2>/dev/null || echo "{}")
MODEL_FILE=$(echo "$INFO_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('model_file', ''))
" 2>/dev/null || echo "")
HF_REPO=$(echo "$INFO_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('hf_repo', 'unsloth/Qwen3.5-27B-GGUF'))
" 2>/dev/null || echo "unsloth/Qwen3.5-27B-GGUF")

if [ -n "$MODEL_FILE" ]; then
    echo "[worker_join] Coordinator đang chạy model: $MODEL_FILE"
else
    echo "[worker_join] WARN: Không lấy được model_file từ coordinator"
fi

# ── Kiểm tra / tải model (chỉ khi --also-download-model hoặc cần standalone) ─
if [ "$ALSO_DOWNLOAD" = true ] && [ -n "$MODEL_FILE" ]; then
    MODEL_PATH="$MODELS_DIR/$MODEL_FILE"
    mkdir -p "$MODELS_DIR"

    if [ -f "$MODEL_PATH" ]; then
        echo "[worker_join] Model đã tồn tại: $MODEL_PATH"
    else
        # Tìm ở các vị trí phổ biến trước khi tải
        FOUND=""
        for SEARCH_DIR in \
            "/home/*/llama.cpp/models" \
            "/opt/models" \
            "$HOME/models" \
            "$HOME/llama.cpp/models"
        do
            for F in $SEARCH_DIR/$MODEL_FILE 2>/dev/null; do
                if [ -f "$F" ]; then FOUND="$F"; break 2; fi
            done
        done

        if [ -n "$FOUND" ]; then
            echo "[worker_join] Tìm thấy model tại $FOUND — tạo symlink"
            ln -sf "$FOUND" "$MODEL_PATH"
        else
            echo "[worker_join] Tải $MODEL_FILE từ HuggingFace ($HF_REPO)..."
            BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"
            URL="${BASE_URL}/${MODEL_FILE}"

            if command -v wget &>/dev/null; then
                wget --continue --progress=bar:force -O "$MODEL_PATH" "$URL"
            elif command -v curl &>/dev/null; then
                curl -L --continue-at - --progress-bar -o "$MODEL_PATH" "$URL"
            else
                echo "[worker_join] ERROR: Cần wget hoặc curl để tải model"
                exit 1
            fi

            echo "[worker_join] Model đã tải: $MODEL_PATH"
        fi
    fi
fi

# ── Kiểm tra binary ───────────────────────────────────────────────────────
RPC_BIN="$BIN_DIR/llama-rpc-server"
if [ ! -f "$RPC_BIN" ]; then
    # Tìm trong PATH
    RPC_BIN=$(command -v llama-rpc-server 2>/dev/null || echo "")
fi
if [ -z "$RPC_BIN" ] || [ ! -f "$RPC_BIN" ]; then
    echo "[worker_join] ERROR: Không tìm thấy llama-rpc-server"
    echo "  Build bằng lệnh: cd $SCRIPT_DIR/.. && ./build.sh"
    exit 1
fi
echo "[worker_join] RPC binary: $RPC_BIN"

# ── Tính NGL cho rpc-server ───────────────────────────────────────────────
# llama-rpc-server dùng GPU để tăng tốc compute offload từ main server
RPC_NGL=0
if [ -n "${VRAM_FREE_MB:-}" ] && [ "$VRAM_FREE_MB" -gt 0 ] && [ -n "$MODEL_FILE" ]; then
    MODEL_PATH_FOR_NGL="$MODELS_DIR/$MODEL_FILE"
    if [ -f "$MODEL_PATH_FOR_NGL" ] && command -v python3 &>/dev/null; then
        # Reserve 1500MB cho buffer của rpc-server
        RPC_NGL=$(python3 "$SCRIPT_DIR/auto_ngl.py" "$MODEL_PATH_FOR_NGL" 1500 2>/dev/null || echo 0)
        echo "[worker_join] Auto NGL (rpc-server): $RPC_NGL"
    fi
fi
if [ "$RPC_NGL" -eq 0 ] && [ "$GPU_TYPE" != "none" ]; then
    # Không có model để tính → dùng NGL_AUTO từ detect_hw.sh
    RPC_NGL="${NGL_AUTO:-0}"
    echo "[worker_join] NGL fallback: $RPC_NGL (từ detect_hw)"
fi

# ── Khởi động llama-rpc-server ────────────────────────────────────────────
echo ""
echo "[worker_join] Khởi động llama-rpc-server trên ${RPC_HOST}:${RPC_PORT}..."

RPC_ARGS=(
    --host 0.0.0.0
    --port "$RPC_PORT"
)
if [ "$RPC_NGL" -gt 0 ]; then
    RPC_ARGS+=(--n-gpu-layers "$RPC_NGL")
fi

# Chạy background nếu không có PM2, ngược lại dùng PM2
if command -v pm2 &>/dev/null; then
    pm2 describe "llama-rpc-worker" &>/dev/null && pm2 delete "llama-rpc-worker" || true
    pm2 start "$RPC_BIN" \
        --name "llama-rpc-worker" \
        --interpreter none \
        -- "${RPC_ARGS[@]}"
    echo "[worker_join] llama-rpc-server chạy qua PM2 (name: llama-rpc-worker)"
else
    "$RPC_BIN" "${RPC_ARGS[@]}" &
    RPC_PID=$!
    echo "[worker_join] llama-rpc-server PID=$RPC_PID"
    echo "$RPC_PID" > /tmp/llama-rpc-worker.pid
fi

# Đợi port mở
echo "[worker_join] Đợi RPC server khởi động..."
for i in $(seq 1 20); do
    if python3 -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', $RPC_PORT), timeout=1)
    s.close(); sys.exit(0)
except: sys.exit(1)
" 2>/dev/null; then
        echo "[worker_join] RPC server sẵn sàng trên port $RPC_PORT"
        break
    fi
    echo "[worker_join] ($i/20) Chờ..."
    sleep 1
    if [ "$i" -eq 20 ]; then
        echo "[worker_join] WARN: Port $RPC_PORT vẫn chưa mở sau 20s"
    fi
done

# ── Detect GPU info để gửi lên ────────────────────────────────────────────
GPU_INFO="${GPU_TYPE}"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "$GPU_TYPE")
fi

# ── Đăng ký với coordinator ───────────────────────────────────────────────
echo ""
echo "[worker_join] Đăng ký với coordinator ${COORDINATOR_URL}..."

REGISTER_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'host':       '$RPC_HOST',
    'rpc_port':   $RPC_PORT,
    'gpu':        '$GPU_INFO',
    'gpu_layers': $RPC_NGL,
}))
")

RESPONSE=$(curl -sf \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$REGISTER_PAYLOAD" \
    "${COORDINATOR_URL}/api/workers/register" 2>&1) || {
    echo "[worker_join] ERROR: Không thể kết nối coordinator: $RESPONSE"
    exit 1
}

echo "[worker_join] Response: $RESPONSE"

STATUS=$(echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('status', 'error'))
" 2>/dev/null || echo "error")

if [ "$STATUS" = "ok" ]; then
    echo ""
    echo "========================================================"
    echo "  Worker joined successfully!"
    echo "  IP       : $RPC_HOST"
    echo "  RPC Port : $RPC_PORT"
    echo "  GPU      : $GPU_INFO"
    echo "  NGL      : $RPC_NGL"
    echo ""
    echo "  Coordinator sẽ restart llama-server trong ~5s với:"
    echo "    --rpc ${RPC_HOST}:${RPC_PORT}"
    echo "========================================================"
else
    echo "[worker_join] ERROR: Đăng ký thất bại: $RESPONSE"
    exit 1
fi
