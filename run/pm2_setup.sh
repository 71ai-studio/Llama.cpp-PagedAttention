#!/usr/bin/env bash
# pm2_setup.sh — cài đặt PM2, đăng ký startup, khởi động server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "  PM2 Setup — llama-server (paged KV)"
echo "======================================================================"

# ── Kiểm tra PM2 ──────────────────────────────────────────────────────────
if ! command -v pm2 &>/dev/null; then
    echo "[1/4] Cài đặt PM2..."
    if command -v npm &>/dev/null; then
        npm install -g pm2
    else
        echo "ERROR: cần Node.js và npm. Cài tại: https://nodejs.org"
        exit 1
    fi
else
    echo "[1/4] PM2 đã có: $(pm2 --version)"
fi

# ── Kiểm tra llama-server ─────────────────────────────────────────────────
BUILD_BIN="$SCRIPT_DIR/../llama-cpp-python/vendor/llama.cpp/build/bin"
if [ ! -f "$BUILD_BIN/llama-server" ]; then
    echo ""
    echo "[!] llama-server chưa được build."
    echo "    Chạy: /home/vuna/llama.cpp/build.sh"
    echo "    Sau đó chạy lại script này."
    exit 1
fi
echo "[2/4] llama-server: OK ($BUILD_BIN/llama-server)"

# ── chmod scripts ─────────────────────────────────────────────────────────
chmod +x "$SCRIPT_DIR/server.sh"
chmod +x "$SCRIPT_DIR/detect_hw.sh"
echo "[3/4] Permissions: OK"

# ── Start PM2 ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Khởi động PM2..."
echo ""

cd "$SCRIPT_DIR"

# Stop nếu đang chạy
pm2 stop ecosystem.config.js 2>/dev/null || true
pm2 delete ecosystem.config.js 2>/dev/null || true

# Start chỉ instance có model
echo "Kiểm tra models có sẵn..."
MODELS_DIR="$SCRIPT_DIR/../models"

START_ARGS=""
[ -f "$MODELS_DIR/Qwen3.5-27B-Q4_K_M.gguf"     ] && START_ARGS="$START_ARGS --only q4km,q4km-flat"   && echo "  ✓ Q4_K_M"
[ -f "$MODELS_DIR/Qwen3.5-27B-Q5_K_M.gguf"     ] && START_ARGS="$START_ARGS --only q5km"             && echo "  ✓ Q5_K_M"
[ -f "$MODELS_DIR/Qwen_Qwen3.5-27B-Q6_K.gguf"  ] && START_ARGS="$START_ARGS --only q6k"              && echo "  ✓ Q6_K"

if [ -z "$START_ARGS" ]; then
    echo ""
    echo "[!] Chưa có model nào. Chạy download trước:"
    echo "    $SCRIPT_DIR/download_models.sh"
    exit 1
fi

# PM2 không hỗ trợ --only nhiều instance cùng lúc, dùng vòng lặp
for INSTANCE in q4km q4km-flat q5km q6k; do
    MODEL_FILE=""
    case $INSTANCE in
        q4km|q4km-flat) MODEL_FILE="Qwen3.5-27B-Q4_K_M.gguf" ;;
        q5km)           MODEL_FILE="Qwen3.5-27B-Q5_K_M.gguf" ;;
        q6k)            MODEL_FILE="Qwen_Qwen3.5-27B-Q6_K.gguf" ;;
    esac

    if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
        pm2 start ecosystem.config.js --only "$INSTANCE"
    else
        echo "  ⚠️  Skip $INSTANCE (model chưa có)"
    fi
done

echo ""
pm2 save

# ── Auto-start khi reboot ─────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Đăng ký auto-start khi reboot:"
pm2 startup | tail -5
echo ""
echo "  Chạy lệnh 'sudo ...' ở trên để hoàn tất đăng ký."
echo "======================================================================"

echo ""
echo "PM2 status:"
pm2 list

echo ""
echo "======================================================================"
echo "  Endpoints:"
echo "    Q4_K_M  : http://localhost:11434/v1/chat/completions"
echo "    Q5_K_M  : http://localhost:11435/v1/chat/completions"
echo "    Q6_K    : http://localhost:11436/v1/chat/completions"
echo "    Flat KV : http://localhost:11437/v1/chat/completions"
echo ""
echo "  Lệnh PM2 thường dùng:"
echo "    pm2 list              # xem trạng thái"
echo "    pm2 logs q4km         # xem log"
echo "    pm2 monit             # dashboard realtime"
echo "    pm2 restart q4km      # restart 1 instance"
echo "    pm2 stop all          # dừng tất cả"
echo "======================================================================"
