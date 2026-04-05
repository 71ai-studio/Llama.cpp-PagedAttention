#!/usr/bin/env bash
# build_macos.sh — Build llama.cpp với Metal backend (macOS arm64 + x86_64 universal binary)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llama-cpp-python/vendor/llama.cpp"
BUILD_DIR_ARM="$SRC_DIR/build_arm64"
BUILD_DIR_X86="$SRC_DIR/build_x86_64"
BUILD_DIR_UNIV="$SRC_DIR/build_universal"
LOG_FILE="$(dirname "${BASH_SOURCE[0]}")/build_macos_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================"
echo "  llama.cpp macOS Build — Metal + Universal Binary"
echo "  Source  : $SRC_DIR"
echo "  Log     : $LOG_FILE"
echo "  macOS   : $(sw_vers -productVersion 2>/dev/null || echo unknown)"
echo "  Xcode   : $(xcodebuild -version 2>/dev/null | head -1 || echo unknown)"
echo "======================================================================"

build_arch() {
    local arch="$1"
    local build_dir="$2"
    local cmake_arch_flag="$3"

    echo ""
    echo "── Building $arch ──────────────────────────────────────────────"
    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake "$SRC_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        2>&1 | tee -a "$LOG_FILE"

    cmake --build . -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" 2>&1 | tee -a "$LOG_FILE"
    echo "── $arch done ──────────────────────────────────────────────────"
}

# Build native only (for Universal, cross-compiling llama.cpp GGML Metal shaders
# requires same-arch build for shader compilation; lipo merges afterwards)
NATIVE_ARCH="$(uname -m)"

if [[ "${1:-}" == "--universal" ]]; then
    echo "[mode] Universal binary (arm64 + x86_64)"
    build_arch "arm64"   "$BUILD_DIR_ARM"  ""
    build_arch "x86_64"  "$BUILD_DIR_X86"  ""

    echo ""
    echo "── Merging universal binaries with lipo ────────────────────────"
    mkdir -p "$BUILD_DIR_UNIV/bin"

    for binary in llama-cli llama-server llama-bench llama-perplexity; do
        ARM_BIN="$BUILD_DIR_ARM/bin/$binary"
        X86_BIN="$BUILD_DIR_X86/bin/$binary"
        OUT_BIN="$BUILD_DIR_UNIV/bin/$binary"

        if [[ -f "$ARM_BIN" && -f "$X86_BIN" ]]; then
            lipo -create "$ARM_BIN" "$X86_BIN" -output "$OUT_BIN"
            echo "  lipo → $OUT_BIN ($(lipo -archs "$OUT_BIN"))"
        else
            echo "  SKIP $binary (missing arm64 or x86_64 build)"
        fi
    done

    FINAL_DIR="$BUILD_DIR_UNIV"
else
    echo "[mode] Native $NATIVE_ARCH only"
    BUILD_DIR="$SRC_DIR/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake "$SRC_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DLLAMA_BUILD_SERVER=ON \
        2>&1 | tee "$LOG_FILE"

    cmake --build . -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" 2>&1 | tee -a "$LOG_FILE"

    FINAL_DIR="$BUILD_DIR"
fi

echo ""
echo "======================================================================"
echo "  BUILD SUCCESS"
echo "  Binaries : $FINAL_DIR/bin/"
echo "  Log      : $LOG_FILE"
echo "======================================================================"
echo ""
echo "Binaries:"
ls "$FINAL_DIR/bin/" | grep -E "llama-(cli|server|bench|perplexity)" || true

echo ""
echo "Kiểm tra Metal backend:"
"$FINAL_DIR/bin/llama-cli" --version 2>&1 | head -3 || true
