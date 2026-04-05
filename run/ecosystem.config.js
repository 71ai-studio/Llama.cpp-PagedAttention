// ecosystem.config.js — PM2 config cho llama-server + coordinator
// Usage:
//   pm2 start ecosystem.config.js                     # start tất cả
//   pm2 start ecosystem.config.js --only q4km         # chỉ llama-server
//   pm2 start ecosystem.config.js --only coordinator  # chỉ coordinator
//   pm2 stop all
//   pm2 restart q4km
//   pm2 logs coordinator
//   pm2 monit

const BASE_DIR = __dirname;
const MODELS_DIR = `${BASE_DIR}/../models`;
const SERVER_SCRIPT = `${BASE_DIR}/server.sh`;

// Cấu hình chung
const common = {
  script: SERVER_SCRIPT,
  interpreter: "/bin/bash",
  restart_delay: 5000,       // chờ 5s trước khi restart
  max_restarts: 5,            // tối đa 5 lần restart liên tiếp
  min_uptime: "30s",          // phải sống ít nhất 30s mới tính là stable
  watch: false,               // không watch file changes
  autorestart: true,
  log_date_format: "YYYY-MM-DD HH:mm:ss",
  merge_logs: true,
  env: {
    // Paged KV cache dùng CPU RAM — giới hạn tối đa 80% RAM available
    // Ghi đè bằng LLAMA_KV_RAM_LIMIT=0.5 nếu cần chạy nhiều instance
    LLAMA_KV_RAM_LIMIT: "0.80",
  },
};

module.exports = {
  apps: [

    // ── Q4_K_M — hybrid GPU+CPU (RTX 3060 12GB: ~48/64 layers)  ─────────
    {
      ...common,
      name: "q4km",
      env: {
        ...common.env,
        LLAMA_MODEL:        "Qwen3.5-27B-Q4_K_M.gguf",
        LLAMA_PORT:         "11434",
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "4",
        LLAMA_KV_PAGE_SIZE: "16",
        // LLAMA_NGL không set → auto_ngl.py tính tự động từ VRAM free
      },
    },

    // ── Q5_K_M — Chất lượng cao hơn, hybrid GPU+CPU ─────────────────────
    {
      ...common,
      name: "q5km",
      env: {
        ...common.env,
        LLAMA_MODEL:        "Qwen3.5-27B-Q5_K_M.gguf",
        LLAMA_PORT:         "11435",
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "2",
        LLAMA_KV_PAGE_SIZE: "16",
        // LLAMA_NGL không set → auto_ngl.py tính tự động
      },
    },

    // ── Q6_K — Model hiện có, hybrid ────────────────────────────────────
    {
      ...common,
      name: "q6k",
      env: {
        ...common.env,
        LLAMA_MODEL:        "Qwen_Qwen3.5-27B-Q6_K.gguf",
        LLAMA_PORT:         "11436",
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "4096",
        LLAMA_PARALLEL:     "2",
        LLAMA_KV_PAGE_SIZE: "16",
        // LLAMA_NGL không set → auto_ngl.py tính tự động
      },
    },

    // ── Q4_K_M (paged KV disabled) — dùng để benchmark so sánh ─────────
    {
      ...common,
      name: "q4km-flat",
      env: {
        ...common.env,
        LLAMA_MODEL:        "Qwen3.5-27B-Q4_K_M.gguf",
        LLAMA_PORT:         "11437",
        LLAMA_HOST:         "127.0.0.1",
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "4",
        LLAMA_KV_PAGE_SIZE: "0",
        // LLAMA_NGL không set → auto_ngl.py tính tự động
      },
    },

    // ── Coordinator — proxy + worker registry ────────────────────────────
    // Nhận requests tại :11433, proxy tới llama-server :11434
    // Worker mới join qua: POST http://this-vm:11433/api/workers/register
    // Khi có worker mới → tự restart q4km với --rpc worker:50052
    {
      ...common,
      name:        "coordinator",
      script:      `${BASE_DIR}/distributed_coordinator.py`,
      interpreter: "python3",
      args:        "--server-pm2-name q4km --port 11433 --model-file Qwen3.5-27B-Q4_K_M.gguf",
      env: {
        LLAMA_MODEL: "Qwen3.5-27B-Q4_K_M.gguf",
      },
      // coordinator không restart theo cơ chế llama-server
      max_restarts: 10,
      min_uptime:   "5s",
    },

  ],
};
