#!/usr/bin/env bash
# build_turbo.sh — Build llama-cpp-python with CUDA + TurboQuant support
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR/llama-cpp-python"

cd "$PKG_DIR"

echo "=== Building llama-cpp-python + TurboQuant (CUDA) ==="
echo "Package dir: $PKG_DIR"
echo ""

# Optionally clean CMake cache for a fresh build
if [[ "${1:-}" == "--clean" ]]; then
    echo "[clean] Removing CMake build cache..."
    rm -rf build/
fi

echo "[build] Running pip install with CUDA enabled..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install -e . --no-build-isolation 2>&1

echo ""
echo "=== Build complete ==="
echo "To verify turbo types are registered, run:"
echo "  python3 -c \"from llama_cpp import llama_cpp; print('TURBO4:', llama_cpp.GGML_TYPE_TURBO4)\""
