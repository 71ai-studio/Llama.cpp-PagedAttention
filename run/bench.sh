#!/usr/bin/env bash
# bench.sh — chạy từng cấu hình trong configs.json, lưu kết quả ra results/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/../llama-cpp-python/vendor/llama.cpp/build/bin/llama-bench"
MODEL="$SCRIPT_DIR/../models/Qwen_Qwen3.5-27B-Q6_K.gguf"
CONFIGS="$SCRIPT_DIR/configs.json"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

# ── kiểm tra dependencies ──────────────────────────────────────────────────
if ! command -v jq &>/dev/null; then
    echo "ERROR: cần cài jq để đọc configs.json  (sudo apt install jq)"
    exit 1
fi
if [ ! -f "$BENCH" ]; then
    echo "ERROR: không tìm thấy llama-bench tại $BENCH"
    exit 1
fi
if [ ! -d "$(dirname "$MODEL")" ]; then
    echo "ERROR: thư mục models không tồn tại"
    exit 1
fi

# ── đọc danh sách cấu hình ─────────────────────────────────────────────────
N=$(jq 'length' "$CONFIGS")
echo "======================================================================"
echo "  llama-bench — so sánh KV cache configurations"
echo "  Model   : $MODEL"
echo "  Configs : $N cấu hình"
echo "  Results : $RESULTS_DIR"
echo "======================================================================"
echo ""

SUMMARY_FILE="$RESULTS_DIR/summary_$(date +%Y%m%d_%H%M%S).md"
{
echo "# Benchmark KV Cache — $(date)"
echo ""
echo "Model: \`$(basename "$MODEL")\`"
echo ""
} > "$SUMMARY_FILE"

for i in $(seq 0 $((N - 1))); do
    NAME=$(jq -r ".[$i].name"         "$CONFIGS")
    DESC=$(jq -r ".[$i].desc"         "$CONFIGS")
    MODEL_FILE=$(jq -r ".[$i].model"  "$CONFIGS")
    FA=$(jq -r ".[$i].fa"             "$CONFIGS")
    KVPS=$(jq -r ".[$i].kv_page_size" "$CONFIGS")
    NGL=$(jq -r ".[$i].ngl"           "$CONFIGS")
    NP=$(jq -r ".[$i].n_prompt"       "$CONFIGS")
    NG=$(jq -r ".[$i].n_gen"          "$CONFIGS")
    REPS=$(jq -r ".[$i].reps"         "$CONFIGS")

    MODEL_PATH="$SCRIPT_DIR/../models/$MODEL_FILE"
    OUT_MD="$RESULTS_DIR/${NAME}.md"
    LOG="$RESULTS_DIR/${NAME}.log"

    echo "----------------------------------------------------------------------"
    echo "[$((i+1))/$N] $NAME"
    echo "  Mô tả   : $DESC"
    echo "  Model   : $MODEL_FILE"
    echo "  Lệnh    : llama-bench -ngl $NGL -fa $FA --kv-page-size $KVPS -p $NP -n $NG -r $REPS"
    echo "  Log     : $LOG"

    # Kiểm tra model tồn tại
    if [ ! -f "$MODEL_PATH" ]; then
        echo "  ⚠️  SKIP: model không tồn tại — $MODEL_PATH"
        echo "  → Chạy download_models.sh để tải về"
        echo ""
        continue
    fi

    echo "----------------------------------------------------------------------"

    # Build args
    ARGS=(
        -m "$MODEL_PATH"
        -ngl "$NGL"
        -fa "$FA"
        --kv-page-size "$KVPS"
        -p "$NP"
        -n "$NG"
        -r "$REPS"
    )

    # Chạy và lưu log
    START_TS=$(date +%s)
    "$BENCH" "${ARGS[@]}" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [ "$EXIT_CODE" -eq 0 ]; then
        STATUS="OK"
        echo "  => Xong trong ${ELAPSED}s"
    else
        STATUS="FAILED (exit $EXIT_CODE)"
        echo "  => THẤT BẠI sau ${ELAPSED}s"
    fi

    # Ghi vào summary
    {
    echo "## $NAME — $DESC"
    echo ""
    echo "Thời gian chạy: ${ELAPSED}s | Trạng thái: $STATUS"
    echo ""
    echo '```'
    cat "$LOG"
    echo '```'
    echo ""
    } >> "$SUMMARY_FILE"

    echo ""
done

echo "======================================================================"
echo "  Hoàn tất! Summary: $SUMMARY_FILE"
echo "======================================================================"
