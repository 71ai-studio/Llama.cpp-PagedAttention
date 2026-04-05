#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/llama-cpp-python/vendor/llama.cpp"
BUILD_DIR="$SRC_DIR/build"
LOG_FILE="$SCRIPT_DIR/build_$(date +%Y%m%d_%H%M%S).log"

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Source dir : $SRC_DIR"
echo "Build dir  : $BUILD_DIR"
echo "Log file   : $LOG_FILE"
echo "CUDA       : $(nvcc --version | head -1)"
echo "Started    : $(date)"
echo "------------------------------------------------------------"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ── Configure với CUDA ───────────────────────────────────────────────────────
echo "[1/2] CMake configure..."
cmake "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    2>&1 | tee "$LOG_FILE"

# ── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "[2/2] Build ($(nproc) cores)..."
cmake --build . -j"$(nproc)" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "------------------------------------------------------------"
echo "Finished : $(date)"
echo "Log saved: $LOG_FILE"

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "BUILD SUCCESS"
    echo ""
    echo "CUDA backends:"
    ls "$BUILD_DIR/bin/" | grep -i cuda || echo "(không thấy ggml-cuda — kiểm tra log)"
    ls "$BUILD_DIR/bin/" | grep -i ggml || true
else
    echo "BUILD FAILED (exit $EXIT_CODE)"
    exit "$EXIT_CODE"
fi
