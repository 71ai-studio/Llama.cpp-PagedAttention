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
# KV_RAM_LIMIT: tỉ lệ RAM tối đa dành cho paged KV cache (default: 80%)
KV_RAM_LIMIT="${LLAMA_KV_RAM_LIMIT:-0.80}"

# Auto-detect max NGL nếu không được override
if [ -n "${LLAMA_NGL:-}" ]; then
    NGL="$LLAMA_NGL"
else
    MODEL_PATH_TMP="$MODELS_DIR/$MODEL_FILE"
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

# ── Paged KV: giới hạn CONTEXT_SIZE để KV cache không vượt quá KV_RAM_LIMIT ──
# Paged KV cache bắt buộc dùng CPU RAM (offload_kqv=false) và pre-allocate
# toàn bộ n_ctx slots ngay lúc khởi động → phải kiểm soát n_ctx.
if [ "${KV_PAGE_SIZE:-0}" -gt 0 ] && command -v python3 &>/dev/null; then
    CAPPED_CTX=$(python3 "$SCRIPT_DIR/kv_ctx_limit.py" \
        "$MODEL_PATH" "$CONTEXT_SIZE" \
        --limit "$KV_RAM_LIMIT" \
        --ram-type available \
        --verbose \
        2>&1 | tee /dev/stderr | tail -1)

    # Chỉ dùng giá trị nếu là số hợp lệ
    if [[ "$CAPPED_CTX" =~ ^[0-9]+$ ]] && [ "$CAPPED_CTX" -gt 0 ]; then
        if [ "$CAPPED_CTX" -lt "$CONTEXT_SIZE" ]; then
            echo "[server] KV RAM limit (${KV_RAM_LIMIT}): ctx ${CONTEXT_SIZE} → ${CAPPED_CTX}"
        fi
        CONTEXT_SIZE="$CAPPED_CTX"
    else
        echo "[server] WARN: kv_ctx_limit trả về không hợp lệ ('$CAPPED_CTX') — dùng ctx gốc $CONTEXT_SIZE"
    fi
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
echo "  kv-limit : ${KV_RAM_LIMIT} × RAM available"
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

# Paged KV cache yêu cầu kv_unified; flat mode không cần nhưng không hại
if [ "${KV_PAGE_SIZE:-0}" -gt 0 ]; then
    ARGS+=(--kv-unified)
fi

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
