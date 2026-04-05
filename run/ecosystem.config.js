// ecosystem.config.js — PM2 config cho llama-server
// Usage:
//   pm2 start ecosystem.config.js              # start tất cả
//   pm2 start ecosystem.config.js --only q4km  # start 1 instance
//   pm2 stop all
//   pm2 restart q4km
//   pm2 logs q4km
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
};

module.exports = {
  apps: [

    // ── Q4_K_M — Khuyến nghị: full/partial GPU ──────────────────────────
    {
      ...common,
      name: "q4km",
      env: {
        LLAMA_MODEL:        "Qwen3.5-27B-Q4_K_M.gguf",
        LLAMA_PORT:         "11434",          // cùng port Ollama
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "4",
        LLAMA_KV_PAGE_SIZE: "16",
        LLAMA_NGL:          "99",             // Q4_K_M ~16.7GB: thử full GPU
      },
    },

    // ── Q5_K_M — Chất lượng cao hơn, hybrid GPU+CPU ─────────────────────
    {
      ...common,
      name: "q5km",
      env: {
        LLAMA_MODEL:        "Qwen3.5-27B-Q5_K_M.gguf",
        LLAMA_PORT:         "11435",
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "2",              // ít slot hơn vì model lớn
        LLAMA_KV_PAGE_SIZE: "16",
        LLAMA_NGL:          "28",             // ~19.6GB: hybrid
      },
    },

    // ── Q6_K — Model hiện có, hybrid ────────────────────────────────────
    {
      ...common,
      name: "q6k",
      env: {
        LLAMA_MODEL:        "Qwen_Qwen3.5-27B-Q6_K.gguf",
        LLAMA_PORT:         "11436",
        LLAMA_HOST:         "0.0.0.0",
        LLAMA_CTX:          "4096",
        LLAMA_PARALLEL:     "2",
        LLAMA_KV_PAGE_SIZE: "16",
        LLAMA_NGL:          "28",
      },
    },

    // ── Q4_K_M (paged KV disabled) — dùng để benchmark so sánh ─────────
    {
      ...common,
      name: "q4km-flat",
      env: {
        LLAMA_MODEL:        "Qwen3.5-27B-Q4_K_M.gguf",
        LLAMA_PORT:         "11437",
        LLAMA_HOST:         "127.0.0.1",      // internal only
        LLAMA_CTX:          "8192",
        LLAMA_PARALLEL:     "4",
        LLAMA_KV_PAGE_SIZE: "0",              // flat KV
        LLAMA_NGL:          "99",
      },
    },

  ],
};
