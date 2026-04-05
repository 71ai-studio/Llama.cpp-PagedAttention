#!/usr/bin/env bash
# detect_hw.sh — detect hardware và xuất biến môi trường
# Usage: source detect_hw.sh

# ── OS ────────────────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

# ── Binary path ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="$SCRIPT_DIR/../llama-cpp-python/vendor/llama.cpp/build/bin"

if [ "$OS" = "Darwin" ]; then
    # macOS: tìm binary trong build local hoặc Homebrew
    if [ -f "$BUILD_BIN/llama-server" ]; then
        LLAMA_BIN="$BUILD_BIN/llama-server"
    elif command -v llama-server &>/dev/null; then
        LLAMA_BIN="$(which llama-server)"
    else
        echo "[detect_hw] ERROR: llama-server không tìm thấy" >&2
        exit 1
    fi
else
    # Linux
    LLAMA_BIN="$BUILD_BIN/llama-server"
    if [ ! -f "$LLAMA_BIN" ]; then
        echo "[detect_hw] ERROR: llama-server không tìm thấy tại $LLAMA_BIN" >&2
        exit 1
    fi
fi

# ── GPU Detection ─────────────────────────────────────────────────────────
GPU_TYPE="none"
GPU_VRAM_MB=0
NGL_AUTO=0

if [ "$OS" = "Darwin" ]; then
    # macOS: Apple Silicon dùng Metal, unified memory
    if [ "$ARCH" = "arm64" ]; then
        GPU_TYPE="metal"
        # Lấy total unified memory (GB)
        TOTAL_RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        GPU_VRAM_MB=$(( TOTAL_RAM_GB * 1024 ))  # unified = all RAM
        NGL_AUTO=99  # Apple Silicon: tất cả layers lên GPU
        echo "[detect_hw] Apple Silicon: Metal, ${TOTAL_RAM_GB}GB unified memory, ngl=99"
    else
        GPU_TYPE="none"
        NGL_AUTO=0
        echo "[detect_hw] Intel Mac: CPU-only"
    fi

elif command -v nvidia-smi &>/dev/null; then
    # Linux + NVIDIA
    GPU_TYPE="cuda"
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_VRAM_MB="${GPU_VRAM_MB:-0}"

    # Tính ngl dựa trên VRAM available
    # Model Q4_K_M 27B: ~16.7GB, mỗi layer ~260MB
    # Để lại 1.5GB cho KV cache + overhead
    USABLE_MB=$(( GPU_VRAM_MB - 1500 ))
    NGL_AUTO=$(( USABLE_MB / 260 ))
    [ "$NGL_AUTO" -gt 64 ] && NGL_AUTO=64   # max layers cho 27B
    [ "$NGL_AUTO" -lt 0  ] && NGL_AUTO=0

    echo "[detect_hw] NVIDIA GPU: ${GPU_VRAM_MB}MB VRAM, ngl_auto=$NGL_AUTO"
else
    GPU_TYPE="none"
    NGL_AUTO=0
    echo "[detect_hw] CPU-only"
fi

# ── Threads ───────────────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
    N_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
else
    N_THREADS=$(nproc --ignore=2 2>/dev/null || echo 4)
fi

# ── Export ────────────────────────────────────────────────────────────────
export LLAMA_BIN
export GPU_TYPE
export GPU_VRAM_MB
export NGL_AUTO
export N_THREADS
export OS
export ARCH

echo "[detect_hw] llama-server : $LLAMA_BIN"
echo "[detect_hw] GPU type     : $GPU_TYPE"
echo "[detect_hw] ngl auto     : $NGL_AUTO"
echo "[detect_hw] threads      : $N_THREADS"
