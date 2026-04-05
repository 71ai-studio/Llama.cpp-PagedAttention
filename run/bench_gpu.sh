#!/usr/bin/env bash
# bench_gpu.sh — benchmark hybrid GPU+CPU (không chạy full CPU)
set -euo pipefail

BIN=/home/vuna/llama.cpp/llama-cpp-python/vendor/llama.cpp/build/bin
Q4=/home/vuna/llama.cpp/models/Qwen3.5-27B-Q4_K_M.gguf
Q6=/home/vuna/llama.cpp/models/Qwen_Qwen3.5-27B-Q6_K.gguf
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS=$SCRIPT_DIR/results

mkdir -p "$RESULTS"
cd "$BIN"
export LD_LIBRARY_PATH=.

# ── Auto-detect NGL tối ưu cho từng model ─────────────────────────────────
NGL_Q4=$(python3 "$SCRIPT_DIR/auto_ngl.py" "$Q4" 1500 2>/dev/null || echo 28)
NGL_Q6=$(python3 "$SCRIPT_DIR/auto_ngl.py" "$Q6" 1500 2>/dev/null || echo 28)

echo "Auto NGL: Q4_K_M=$NGL_Q4 layers, Q6_K=$NGL_Q6 layers"
echo ""

run() {
    local tag="$1"; shift
    echo "====== $tag ======"
    ./llama-bench "$@" 2>&1 | tee "$RESULTS/${tag}.log"
    echo ""
}

# Q4_K_M — max GPU offload, Flat vs Paged
run "q4km_ngl${NGL_Q4}_flat"    -m $Q4 -ngl $NGL_Q4 -fa 1                   -p 128 -n 32 -r 3
run "q4km_ngl${NGL_Q4}_paged16" -m $Q4 -ngl $NGL_Q4 -fa 1 --kv-page-size 16 -p 128 -n 32 -r 3
run "q4km_ngl${NGL_Q4}_paged32" -m $Q4 -ngl $NGL_Q4 -fa 1 --kv-page-size 32 -p 128 -n 32 -r 3

# Q4_K_M — ngl=28 (conservative, so sánh thêm)
run q4km_ngl28_flat    -m $Q4 -ngl 28 -fa 1                   -p 128 -n 32 -r 3
run q4km_ngl28_paged16 -m $Q4 -ngl 28 -fa 1 --kv-page-size 16 -p 128 -n 32 -r 3

# Q6_K — max GPU offload, Flat
run "q6k_ngl${NGL_Q6}_flat"     -m $Q6 -ngl $NGL_Q6 -fa 1                   -p 128 -n 32 -r 3

echo "====== DONE — kết quả tại $RESULTS ======"
ls -lh "$RESULTS"/*.log 2>/dev/null
