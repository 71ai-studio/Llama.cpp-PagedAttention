# llama.cpp — Paged KV Cache + Distributed Inference

**Base commit:** `f49e91787`
**Cập nhật:** 2026-04-05
**Thực hiện:** Vu Nguyen Anh
**Email:** vuna.aid@gmail.com

---

## Tổng quan

Dự án mở rộng llama.cpp với ba lớp cải tiến:

| Layer | Nội dung | Trạng thái |
|---|---|:-:|
| **Phase 1** | Paged KV Cache — block-aligned allocation, LRU free list | ✅ |
| **Phase 2** | Prefix caching, cross-sequence sharing, block-level eviction | ✅ |
| **Phase 3** | Python binding, macOS Metal, Homebrew, Distributed inference | ✅ |

---

## 1. Kiến trúc KV Cache

### 1.1 Vấn đề với Flat KV Cache

```
Flat KV Cache (llama.cpp gốc):
┌──────────────────────────────────────────────────────────┐
│  Cell 0 │ Cell 1 │ Cell 2 │ ... │  Cell N-1             │
└──────────────────────────────────────────────────────────┘
  seq_0      seq_0    seq_1            seq_2

Vấn đề:
  - Internal fragmentation: khoảng trống giữa sequences
  - Không thể reuse KV của prefix chung giữa requests
  - find_slot() linear scan O(N) để tìm slot trống
  - Eviction phải xoá cả sequence, không thể xoá từng block
```

### 1.2 Paged KV Cache (Phase 1)

```
Paged KV Cache (page_size = 16):
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Block 0     │  │  Block 1     │  │  Block 2     │
│  slot 0..15  │  │  slot 0..15  │  │  slot 0..15  │
│  seq_0       │  │  seq_0       │  │  seq_1       │
└──────────────┘  └──────────────┘  └──────────────┘
   phys=0            phys=1            phys=2

cell_idx = phys_block × page_size + slot
```

### 1.3 Prefix Caching + Block Sharing (Phase 2)

```
Hai sequences cùng system prompt → share physical blocks:

Seq A:  [Block 0] → [Block 1] → [Block 2_A]
                        ↑
Seq B:  [Block 0] → [Block 1] → [Block 2_B]
             shared (ref_cnt = 2)

Block 0, 1: hash(token_ids, positions) match → reuse
Block 2_A/B: diverge sau prefix → allocate riêng

Lợi ích:
  - Tiết kiệm VRAM: prefix 1024 tokens × 27B = ~400MB/sequence
  - 10 seq cùng prefix → 9 × 400MB = 3.6GB tiết kiệm
  - Block eviction: LRU active block, không phải cả sequence
```

### 1.4 Cấu trúc dữ liệu (Phase 1 + 2)

```cpp
struct paged_block_t {
    int32_t  ref_cnt      = 0;   // Phase 1: số sequences đang dùng
    int32_t  lru_prev     = -1;  // Phase 1: free-list doubly linked
    int32_t  lru_next     = -1;
    int32_t  alru_prev    = -1;  // Phase 2: active-LRU doubly linked
    int32_t  alru_next    = -1;
    uint64_t content_hash = 0;   // Phase 2: FNV-1a hash; 0 = uncommitted
};

// Phase 2 fields trong llama_kv_cache:
int32_t  paged_alru_head = -1;                          // active LRU head (MRU)
int32_t  paged_alru_tail = -1;                          // active LRU tail (LRU → evict)
unordered_map<uint64_t, int32_t>         paged_prefix_map;    // hash → block_id
vector<vector<pair<seq_id, int32_t>>>    paged_block_owners;  // block → [(seq,lb)]
```

---

## 2. Benchmark thực tế

### Hardware

```
GPU:   NVIDIA GeForce RTX 3060 — 12 GB VRAM (11909 MiB usable)
CPU:   Linux x86_64
Model: Qwen3.5-27B — Q4_K_M (15.58 GiB), Q6_K (21.49 GiB)
Build: f49e91787, CUDA compute 8.6, Flash Attention enabled
Test:  pp128 (prefill 128 tokens), tg32 (generate 32 tokens), r=3
```

### 2.1 Q4_K_M — Hiệu năng theo số GPU layers

> **Phát hiện quan trọng:** `auto_ngl.py` ban đầu tính sai NGL (49 → crash).
> Root cause: không trừ non-layer overhead ra khỏi usable VRAM khi tính.
> Fix: `ngl = floor((vram_free - reserve) / layer_mb) - 1` → ngl=48 (max thực tế).

```
Model: Qwen3.5-27B Q4_K_M (15.58 GiB) — Flat KV, Flash Attn
─────────────────────────────────────────────────────────────────
Config            ngl   VRAM used   pp128 (t/s)    tg32 (t/s)
─────────────────────────────────────────────────────────────────
Flat, ngl=28       28    ~6.0 GB      79.7 ± 1.5     1.78 ± 0.08
Flat, ngl=37       37    ~7.5 GB      99.0 ± 5.5     2.41 ± 0.05
Flat, ngl=48 ★     48   ~11.9 GB     136.5 ± 7.8     3.38 ± 0.24
Flat, ngl=49        —   OOM (crash)       —               —
─────────────────────────────────────────────────────────────────
★ ngl=48 = maximum an toàn (11.9/11.9 GB VRAM)
```

Gain từ ngl=28 → 48: **+71% prefill, +90% decode**

### 2.2 Q4_K_M — Flat vs Paged KV (ngl=48, max GPU)

```
Model: Qwen3.5-27B Q4_K_M — ngl=48, Flash Attn
───────────────────────────────────────────────────────────────────────
Config               kv_page   pp128 (t/s)    tg32 (t/s)   vs Flat
───────────────────────────────────────────────────────────────────────
Flat KV              —         136.5 ± 7.8     3.38 ± 0.24   baseline
Paged KV-16          16        127.3 ± 5.4     2.69 ± 0.15   −6.7%  / −20%
Paged KV-32          32        124.8 ± 5.5     2.54 ± 0.22   −8.6%  / −25%
───────────────────────────────────────────────────────────────────────
```

> **Lưu ý:** Overhead paged KV tại ngl=48 (VRAM gần đầy) cao hơn dự kiến do
> KV cache không được offload lên GPU (offload_kqv=false khi paged).
> Tại ngl=28 hoặc lower, overhead giảm đáng kể (KV pool không cạnh tranh VRAM).

### 2.3 Q4_K_M — So sánh đầy đủ (tất cả cấu hình)

```
Model: Qwen3.5-27B Q4_K_M — Flash Attn, 2026-04-05
────────────────────────────────────────────────────────────────────────────
Config                ngl   kv_page   pp128 (t/s)    tg32 (t/s)
────────────────────────────────────────────────────────────────────────────
Flat KV, ngl=28        28     —         79.7 ± 1.5     1.78 ± 0.08
Flat KV, ngl=37        37     —         99.0 ± 5.5     2.41 ± 0.05
Flat KV, ngl=48 ★      48     —        136.5 ± 7.8     3.38 ± 0.24  ← best speed
Paged KV-16, ngl=28    28    16         77.3 ± 4.0     1.61 ± 0.05
Paged KV-16, ngl=37    37    16         85.7 ± 1.7     2.04 ± 0.02
Paged KV-16, ngl=48    48    16        127.3 ± 5.4     2.69 ± 0.15
Paged KV-32, ngl=37    37    32         93.0 ± 5.8     2.10 ± 0.01
Paged KV-32, ngl=48    48    32        124.8 ± 5.5     2.54 ± 0.22  ← best paged
────────────────────────────────────────────────────────────────────────────
★ = recommended cho single-user / throughput
```

### 2.4 Q6_K — Ảnh hưởng của auto NGL fix

```
Model: Qwen3.5-27B Q6_K (21.49 GiB) — Flat KV, Flash Attn
────────────────────────────────────────────────────────────
Config          ngl (cũ)  ngl (fix)   pp128 (t/s)   tg32 (t/s)
────────────────────────────────────────────────────────────
Flat, ngl=24      24         24         55.3 ± 1.6    1.37 ± 0.02
Flat, ngl=34      —          34         66.4 ± 1.6    1.61 ± 0.05  +20% ↑
────────────────────────────────────────────────────────────
auto_ngl fix: Q6_K từ ngl=24 → ngl=34 (+20% throughput)
```

### 2.5 Khi nào dùng Paged KV

```
Use case                     Flat KV       Paged KV-16    Winner
───────────────────────────────────────────────────────────────────
Single user, max speed        136.5 t/s     127.3 t/s      Flat  ★
Multi-user (10+ sequences)    OOM/fragm.    OK             Paged ★
Same system prompt reuse      Không thể     −40% memory    Paged ★
Long context (>8K) eviction   Xoá cả seq   Per-block      Paged ★
Memory-constrained server     Waste ~30%    Tối ưu         Paged ★
```

---

## 3. Files đã thay đổi

### Phase 1 — Core KV Cache
| File | Thay đổi |
|---|---|
| `src/llama-kv-cache.h` | `paged_block_t`, paged pool fields, method declarations |
| `src/llama-kv-cache.cpp` | `paged_pool_init`, `find_slot_paged`, `paged_alloc_block`, `paged_free_block`, `paged_save_snap`, `paged_restore_snap`, cập nhật `seq_rm`, `clear`, `apply_ubatch`, `prepare` |
| `include/llama.h` | `kv_page_size` trong `llama_context_params` |
| `src/llama-cparams.h` | `kv_page_size` trong `llama_cparams` |
| `src/llama-context.cpp` | Auto-enforce flash_attn + kv_unified khi page_size > 0 |
| `common/arg.cpp` | `--kv-page-size N` flag |
| `tools/llama-bench/llama-bench.cpp` | `--kv-page-size` trong bench tool |

### Phase 2 — Prefix Caching + Eviction
| File | Thay đổi |
|---|---|
| `src/llama-kv-cache.h` | `alru_prev/next`, `content_hash` trong `paged_block_t`; `paged_alru_*`, `paged_prefix_map`, `paged_block_owners`; snapshot fields; 4 method declarations |
| `src/llama-kv-cache.cpp` | `paged_alru_push_head`, `paged_alru_remove`, `paged_evict_lru`, `paged_compute_hash`; rewrite `find_slot_paged` (4 phases); cập nhật `seq_rm`, `apply_ubatch`, `paged_save_snap`, `paged_restore_snap` |
| `src/llama-kv-cells.h` | `v_cells` → `mutable` (Phase 2 modifies cells in const `find_slot_paged`) |

### Phase 3 — Tooling + Distribution
| File | Mô tả |
|---|---|
| `llama_cpp/llama_cpp.py` | `kv_page_size: c_uint32` trong `LlamaContextParams._fields_` |
| `llama_cpp/llama.py` | `Llama(kv_page_size=0)` parameter, auto-enable flash_attn |
| `build_macos.sh` | Metal build + `--universal` (arm64 + x86_64 lipo) |
| `homebrew/llama-paged-kv.rb` | Homebrew formula |
| `run/distributed_coordinator.py` | Prefix-cache-aware load balancer (OpenAI-compatible) |
| `deploy.sh` | Auto pull/build/deploy trên Ubuntu/macOS, RPC worker/main |
| `run/nodes.json` | Cluster topology config |
| `run/cluster.py` | SSH orchestrator (start/stop/status/logs) |

### Tooling fixes
| File | Fix |
|---|---|
| `run/auto_ngl.py` | Bug: không trừ `non_layer_mb` → ngl quá cao → crash. Fix: `ngl = floor((vram_free - reserve - non_layer) / layer_mb) - 1` |

---

## 4. API Usage

### C API
```c
llama_context_params cparams = llama_context_default_params();
cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
cparams.kv_page_size    = 16;  // 0=flat, 8/16/32=paged
// kv_unified=true và offload_kqv=false được set tự động
```

### CLI
```bash
# Single node — paged KV
llama-server \
  --model models/Qwen3.5-27B-Q4_K_M.gguf \
  --flash-attn \
  --kv-page-size 16 \
  --n-gpu-layers 48 \
  --port 11434

# Benchmark
llama-bench -m model.gguf -fa 1 --kv-page-size 16 -ngl 48 -p 128 -n 32 -r 3
```

### Python
```python
from llama_cpp import Llama

# Phase 1: basic paged KV
llm = Llama("model.gguf", n_gpu_layers=48, kv_page_size=16)

# Phase 2: prefix caching tự động khi kv_page_size > 0
# Nhiều requests cùng system prompt → share physical KV blocks
results = [llm("Summarize: " + doc) for doc in documents]
```

---

## 5. Build

### Linux + CUDA (single node)
```bash
./build.sh
# Flags: -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DLLAMA_BUILD_SERVER=ON
```

### Linux + CUDA (cluster với RPC)
```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_RPC=ON" ./build.sh
```

### macOS Metal
```bash
./build_macos.sh              # native (arm64 hoặc x86_64)
./build_macos.sh --universal  # universal binary (arm64 + x86_64)
```

### llama-cpp-python (Python binding)
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install -e llama-cpp-python/ --no-build-isolation
```

---

## 6. Distributed Inference — Mô hình kết nối nhiều VM/PC/Mac

### 6.1 Kiến trúc tổng quan

```
                         Internet / LAN
         ┌───────────────────────────────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────┐                    ┌─────────────────────┐
│  Coordinator    │                    │  Client (OpenAI API) │
│  :11433         │◄───── requests ────│  curl / Python SDK  │
│  (load balance) │                    └─────────────────────┘
└────────┬────────┘
         │ prefix-cache-aware routing
    ┌────┴────────────────────────────┐
    ▼                                 ▼
┌──────────────────┐        ┌──────────────────┐
│  Main Node       │        │  Main Node       │
│  Ubuntu + CUDA   │        │  macOS M-series  │
│  llama-server    │        │  llama-server    │
│  :11434          │        │  :11435          │
│  ngl=48, Q4_K_M  │        │  ngl=99, Q4_K_M  │
└────────┬─────────┘        └────────┬─────────┘
         │ --rpc                     │ --rpc
    ┌────┴──────┐               ┌────┴──────┐
    ▼           ▼               ▼           ▼
┌────────┐ ┌────────┐     ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│     │Worker 3│ │Worker 4│
│RTX3090 │ │RTX3060 │     │M2 Pro  │ │CPU-only│
│:50052  │ │:50053  │     │:50054  │ │:50055  │
│22 layer│ │20 layer│     │99 layer│ │CPU GPU │
└────────┘ └────────┘     └────────┘ └────────┘
```

### 6.2 Mô hình kết nối thực tế (3 kịch bản)

#### Kịch bản A — Home lab (LAN, 2-4 máy)
```
Topology:
  PC chính (Ubuntu + RTX 3060)  → llama-server main + RPC client
  PC phụ  (Ubuntu + RTX 3090)  → llama-rpc-server worker
  MacBook (Apple M2 Pro 32GB)  → llama-rpc-server worker

Phân tải:
  - PC chính:  ngl=20 (20 layers lên GPU local)
  - PC phụ:    ngl=22 (22 layers qua RPC)
  - MacBook:   ngl=22 (22 layers qua RPC, Metal)
  - Tổng:      64/64 layers trên GPU → full offload 27B model

Bandwidth cần thiết:
  - RPC traffic: ~50-200 MB/s per worker (1Gbps LAN đủ)
  - Latency thêm: ~1-3ms per forward pass over LAN
```

#### Kịch bản B — Cloud VMs (AWS/GCP/Azure)
```
Topology:
  VM-1 (g4dn.xlarge, T4 16GB)  → main server :11434
  VM-2 (g4dn.xlarge, T4 16GB)  → RPC worker  :50052
  VM-3 (c5.4xlarge, CPU-only)  → RPC worker  :50053 (overflow)

nodes.json:
{
  "nodes": [
    {"role":"main",   "host":"10.0.1.10", "gpu_layers":30, "port":11434},
    {"role":"worker", "host":"10.0.1.11", "gpu_layers":34, "rpc_port":50052},
    {"role":"worker", "host":"10.0.1.12", "gpu_layers":0,  "rpc_port":50053}
  ]
}

Chi phí ước tính (AWS us-east-1):
  g4dn.xlarge × 2 = $0.526/h × 2 = $1.05/h
  c5.4xlarge × 1  = $0.68/h
  → ~$1.73/h cho 27B model full offload
```

#### Kịch bản C — MacOS Fleet (nhiều Mac M-series)
```
Topology:
  Mac Studio (M2 Ultra, 192GB) → main server :11434 (ngl=99, đủ cho 27B Q6_K)
  MacBook Pro (M3 Pro, 36GB)   → RPC worker  :50052 (ngl=99, 70B model shard)
  Mac Mini   (M2, 16GB)        → coordinator :11433 (load balance)

Đặc điểm Metal/macOS:
  - Unified memory: VRAM = RAM → không cần chia layer cẩn thận
  - M2 Ultra 192GB: chạy được 70B Q4_K_M full offload
  - RPC over Thunderbolt/LAN: ~10Gbps → latency < 0.5ms

build:
  ./build_macos.sh  # trên mỗi Mac
  # hoặc brew install ./homebrew/llama-paged-kv.rb
```

### 6.3 Setup cluster (script tự động)

```bash
# 1. Cấu hình nodes.json
vim run/nodes.json

# 2. Deploy toàn bộ cluster (SSH tới từng node)
python3 run/cluster.py start --config run/nodes.json --pull

# 3. Kiểm tra status
python3 run/cluster.py status
#
# ═ Status: llama-cluster ════════════════════════════════════
#   Host             Role     GPU                Check          Latency  Status
#   192.168.1.10     main     RTX 3060 12GB      HTTP :11434     45ms  ✓ UP
#   192.168.1.11     worker   RTX 3090 24GB      TCP  :50052      2ms  ✓ UP
#   192.168.1.13     worker   Apple M2 Pro 32GB  TCP  :50054      1ms  ✓ UP

# 4. Coordinator (load balance nhiều main nodes)
python3 run/distributed_coordinator.py \
  --backends 192.168.1.10:11434 192.168.1.13:11434 \
  --port 11433

# 5. Test API
curl http://localhost:11433/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

### 6.4 deploy.sh — tự động trên từng máy

```bash
# Trên mỗi node (Ubuntu hoặc macOS), script tự:
#   1. Detect platform (CUDA / Metal / CPU)
#   2. git pull (nếu --pull)
#   3. Build với GGML_RPC=ON + backend phù hợp
#   4. Đọc nodes.json, tìm config cho IP này
#   5. Khởi động đúng role (worker hoặc main)
#   6. Install systemd (Linux) / launchd (macOS) service

# Worker Ubuntu:
./deploy.sh --config run/nodes.json --self 192.168.1.11 --pull

# Main macOS:
./deploy.sh --config run/nodes.json --self 192.168.1.13

# Rebuild sau update:
./deploy.sh --config run/nodes.json --self 192.168.1.10 --pull --rebuild
```

### 6.5 Bandwidth và latency thực tế

```
Network       Bandwidth   RPC latency/token   Khuyến nghị
──────────────────────────────────────────────────────────
Gigabit LAN   ~125 MB/s   +1–3 ms            Tốt cho ≤3 workers
10G LAN       ~1.2 GB/s   <0.5 ms            Ideal cho fleet
WiFi 6        ~150 MB/s   +3–8 ms            Chấp nhận được
Internet WAN  ~10–50 MB/s +20–100 ms         Chỉ dùng khi cần
──────────────────────────────────────────────────────────
RPC overhead tổng = (n_workers × latency) per forward pass
27B model, 64 layers, 2 workers → +2–6ms/token overhead
```

---

## 7. Deployment đơn giản (single node, PM2)

```bash
# Tải models
./run/download_models.sh

# Setup và start
./run/pm2_setup.sh

# Quản lý
pm2 list
pm2 logs q4km
pm2 monit
pm2 restart q4km
```

| PM2 Name | Model | Port | KV Mode | ngl |
|---|---|:-:|:-:|:-:|
| `q4km` | Q4_K_M (15.6GB) | 11434 | paged-16 | 48 |
| `q4km-flat` | Q4_K_M | 11437 | flat | 48 |
| `q5km` | Q5_K_M (19.6GB) | 11435 | paged-16 | 34 |
| `q6k` | Q6_K (21.5GB) | 11436 | paged-16 | 34 |

---

## 8. Ràng buộc kỹ thuật

```
Khi kv_page_size > 0:
  flash_attn  = true    (bắt buộc, auto-enforced)
  kv_unified  = true    (bắt buộc, auto-enforced)
  offload_kqv = false   (KV pool ở CPU, không offload lên GPU)
  get_can_shift() = false  (không hỗ trợ context shift/RoPE scaling)
  v_trans     = false   (đảm bảo bởi flash_attn=true)

Overhead paged KV vs flat:
  find_slot_paged(): O(tokens) vs O(N) scan → nhanh hơn với N lớn
  alloc_block():     O(1) — lru_remove + lru_push
  evict_lru():       O(page_size × n_owners) — chỉ khi pool cạn
  prefix lookup:     O(1) — unordered_map hash

Phase 2 — khi nào prefix cache có hiệu quả:
  ✓ Nhiều requests cùng system prompt (RAG, chat với context cố định)
  ✓ Batch inference cùng template
  ✗ Mỗi request unique prefix → cache miss 100%, không lợi
  ✗ page_size lớn (64+) → khó match hash đủ tokens
  Khuyến nghị: page_size=16 cho cân bằng hit rate vs overhead
```

---

## 9. Roadmap

### Phase 1 ✅ — Paged KV Cache cơ bản
- [x] Block-aligned KV allocation
- [x] LRU free list management
- [x] Multi-sequence support
- [x] Snapshot/restore cho prepare()
- [x] `--kv-page-size` CLI flag
- [x] llama-bench integration
- [x] Unit tests (48/48 passed)

### Phase 2 ✅ — Prefix Caching + Eviction
- [x] Prefix caching — FNV-1a hash(token_id, pos) per block
- [x] Cross-sequence block sharing — ref_cnt + `cells.seq_add()`
- [x] Block-level eviction — active-LRU list, `paged_evict_lru()`

### Phase 3 ✅ — Tooling + Distribution
- [x] Python binding — `Llama(kv_page_size=16)` trong llama-cpp-python
- [x] macOS Metal build — `build_macos.sh` (native + universal binary)
- [x] Homebrew formula — `homebrew/llama-paged-kv.rb`
- [x] Distributed inference — RPC worker + coordinator + cluster manager

### Phase 4 — Kế hoạch
- [ ] CUDA kernel cho paged attention (hiện tại KV pool trên CPU)
- [ ] Quantized KV cache (TurboQuant+ GGML types) với prefix caching
- [ ] Async prefix prefetch (background fetch từ disk khi cache miss)
- [ ] WebSocket streaming qua coordinator

---

## 10. Tài liệu tham khảo

| Tài liệu | Nguồn |
|---|---|
| vLLM PagedAttention | Kwon et al., SOSP 2023 |
| llama.cpp base | github.com/ggml-org/llama.cpp (commit `f49e91787`) |
| Qwen3.5-27B | huggingface.co/Qwen/Qwen3.5-27B |
| GGUF Q4_K_M / Q6_K | huggingface.co/unsloth/Qwen3.5-27B-GGUF |
| llama.cpp RPC backend | github.com/ggml-org/llama.cpp/tree/master/tools/rpc |
