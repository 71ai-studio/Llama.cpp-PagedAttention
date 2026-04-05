#!/usr/bin/env bash
# download_models.sh — tải Q4_K_M và Q5_K_M từ unsloth/Qwen3.5-27B-GGUF
# Chỉ tải nếu file chưa tồn tại
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

HF_REPO="unsloth/Qwen3.5-27B-GGUF"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

# Danh sách file cần tải: (filename, size_GB)
declare -A FILES=(
    ["Qwen3.5-27B-Q4_K_M.gguf"]="16.7"
    ["Qwen3.5-27B-Q5_K_M.gguf"]="19.6"
)

mkdir -p "$MODELS_DIR"

echo "======================================================================"
echo "  Download Qwen3.5-27B GGUF models"
echo "  Repo   : $HF_REPO"
echo "  Target : $MODELS_DIR"
echo "======================================================================"
echo ""

for FILENAME in "${!FILES[@]}"; do
    SIZE="${FILES[$FILENAME]}"
    DEST="$MODELS_DIR/$FILENAME"
    URL="$BASE_URL/$FILENAME"

    echo "----------------------------------------------------------------------"
    echo "File : $FILENAME  (~${SIZE}GB)"

    if [ -f "$DEST" ]; then
        ACTUAL_SIZE=$(du -h "$DEST" | cut -f1)
        echo "✓ Đã tồn tại ($ACTUAL_SIZE) — bỏ qua"
        echo ""
        continue
    fi

    echo "Bắt đầu tải: $URL"
    echo "Lưu vào    : $DEST"
    echo ""

    # Dùng wget với resume support
    if command -v wget &>/dev/null; then
        wget \
            --continue \
            --progress=bar:force \
            --show-progress \
            -O "$DEST" \
            "$URL"

    # Fallback: curl
    elif command -v curl &>/dev/null; then
        curl \
            -L \
            --continue-at - \
            --progress-bar \
            -o "$DEST" \
            "$URL"

    # Fallback: Python
    else
        python3 - <<PYEOF
import urllib.request, sys, os

url  = "$URL"
dest = "$DEST"

def progress(count, block_size, total_size):
    pct  = count * block_size * 100 // total_size
    done = count * block_size // (1024*1024)
    tot  = total_size // (1024*1024)
    sys.stdout.write(f"\r  {pct}%  {done}MB / {tot}MB")
    sys.stdout.flush()

print(f"Downloading {url}")
urllib.request.urlretrieve(url, dest, reporthook=progress)
print()
PYEOF
    fi

    if [ $? -eq 0 ]; then
        ACTUAL_SIZE=$(du -h "$DEST" | cut -f1)
        echo ""
        echo "✓ Tải xong: $DEST ($ACTUAL_SIZE)"
    else
        echo "✗ Tải thất bại: $FILENAME"
        rm -f "$DEST"   # xóa file incomplete
        exit 1
    fi
    echo ""
done

echo "======================================================================"
echo "  Hoàn tất! Models trong $MODELS_DIR:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (không có file .gguf)"
echo "======================================================================"
