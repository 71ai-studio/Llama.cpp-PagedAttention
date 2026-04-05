#!/usr/bin/env bash
# server.sh — khởi động llama-server với config hybrid GPU+CPU
# PM2 gọi script này trực tiếp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

# ── Load hardware detection ───────────────────────────────────────────────
source "$SCRIPT_DIR/detect_hw.sh"

# ── Đọc config từ biến môi trường (PM2 truyền vào) ───────────────────────
MODEL_FILE="${LLAMA_MODEL:-Qwen3.5-27B-Q4_K_M.gguf}"
SERVER_PORT="${LLAMA_PORT:-11434}"
SERVER_HOST="${LLAMA_HOST:-0.0.0.0}"
CONTEXT_SIZE="${LLAMA_CTX:-8192}"
N_PARALLEL="${LLAMA_PARALLEL:-4}"
KV_PAGE_SIZE="${LLAMA_KV_PAGE_SIZE:-16}"

# Auto-detect max NGL nếu không được override
if [ -n "${LLAMA_NGL:-}" ]; then
    NGL="$LLAMA_NGL"
else
    MODEL_PATH_TMP="$SCRIPT_DIR/../models/$MODEL_FILE"
    if [ -f "$MODEL_PATH_TMP" ] && command -v python3 &>/dev/null; then
        NGL=$(python3 "$SCRIPT_DIR/auto_ngl.py" "$MODEL_PATH_TMP" 1500 2>/dev/null || echo "$NGL_AUTO")
        echo "[server] Auto NGL: $NGL layers (calculated from VRAM)"
    else
        NGL="$NGL_AUTO"
    fi
fi

MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

# ── Kiểm tra model ────────────────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "[server] ERROR: Model không tồn tại: $MODEL_PATH"
    echo "[server] Chạy: ./download_models.sh"
    exit 1
fi

# ── Log thông tin khởi động ───────────────────────────────────────────────
echo "========================================================"
echo "  llama-server (paged KV edition)"
echo "  Model    : $MODEL_FILE"
echo "  Host     : $SERVER_HOST:$SERVER_PORT"
echo "  GPU type : $GPU_TYPE"
echo "  ngl      : $NGL / layers"
echo "  threads  : $N_THREADS"
echo "  ctx      : $CONTEXT_SIZE tokens"
echo "  parallel : $N_PARALLEL slots"
echo "  kv-page  : $KV_PAGE_SIZE"
echo "========================================================"

# ── Build args ────────────────────────────────────────────────────────────
ARGS=(
    --model        "$MODEL_PATH"
    --host         "$SERVER_HOST"
    --port         "$SERVER_PORT"
    --ctx-size     "$CONTEXT_SIZE"
    --n-gpu-layers "$NGL"
    --threads      "$N_THREADS"
    --parallel     "$N_PARALLEL"
    --flash-attn
    --kv-page-size "$KV_PAGE_SIZE"
    --cont-batching                  # continuous batching
    --metrics                        # expose /metrics endpoint
    --log-format    text
)

# Flash attn yêu cầu kv_unified
ARGS+=(--kv-unified)

# macOS Metal: thêm flag riêng
if [ "$GPU_TYPE" = "metal" ]; then
    ARGS+=(--metal)
fi

# CPU-only: tắt offload
if [ "$GPU_TYPE" = "none" ] || [ "$NGL" -eq 0 ]; then
    ARGS+=(--no-kv-offload)
fi

# ── Chạy server ───────────────────────────────────────────────────────────
echo "[server] Khởi động: $LLAMA_BIN ${ARGS[*]}"
echo ""
exec "$LLAMA_BIN" "${ARGS[@]}"
