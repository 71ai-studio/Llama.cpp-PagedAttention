#!/usr/bin/env python3
"""
kv_ctx_limit.py — giới hạn n_ctx của paged KV cache theo % RAM hệ thống

Paged KV cache bắt buộc dùng CPU RAM (offload_kqv=false), và buffer được
pre-allocate cho TOÀN BỘ n_ctx_seq slots ngay lúc khởi động. Script này
tính n_ctx tối đa để KV cache không vượt quá giới hạn RAM cho phép.

Usage:
  python3 kv_ctx_limit.py <model.gguf> <requested_ctx> [OPTIONS]

Options:
  --limit FLOAT    Tỉ lệ RAM tối đa cho KV cache (default: 0.80 = 80%)
  --ram-type STR   'available' hoặc 'total' (default: available)
  --elem-bytes INT Bytes mỗi element (2=f16, 1=q8, default: 2)
  -v, --verbose    In thông tin chi tiết ra stderr

Output:
  Số nguyên — n_ctx thực sự sẽ dùng (có thể nhỏ hơn requested_ctx)

Examples:
  python3 kv_ctx_limit.py model.gguf 8192
  python3 kv_ctx_limit.py model.gguf 8192 --limit 0.5 --verbose
"""

import struct, os, sys, subprocess


# ─── GGUF metadata reader ─────────────────────────────────────────────────────

def read_gguf_kv_params(path):
    """
    Đọc các tham số cần cho KV cache từ GGUF header:
      - n_layers       : llama.block_count
      - n_head_kv      : llama.attention.head_count_kv
      - n_head         : llama.attention.head_count
      - n_embd         : llama.embedding_length
      - head_dim_k     : llama.attention.key_length   (optional, tính nếu thiếu)
      - head_dim_v     : llama.attention.value_length (optional, tính nếu thiếu)

    Returns dict hoặc None nếu không đọc được.
    """
    result = {}

    TARGET_KEYS = {
        'llama.block_count',
        'llama.attention.head_count_kv',
        'llama.attention.head_count',
        'llama.embedding_length',
        'llama.attention.key_length',
        'llama.attention.value_length',
    }

    SCALAR_FMT = {
        0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2),
        4: ('<I', 4), 5: ('<i', 4), 6: ('<f', 4), 7: ('<?', 1),
        10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8),
    }
    ARRAY_ELEM_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return None

            f.read(4)            # version
            f.read(8)            # n_tensors
            n_kv = struct.unpack('<Q', f.read(8))[0]

            def read_str():
                length = struct.unpack('<Q', f.read(8))[0]
                return f.read(length).decode('utf-8', errors='replace')

            for _ in range(min(n_kv, 1000)):
                key   = read_str()
                vtype = struct.unpack('<I', f.read(4))[0]

                if vtype == 8:    # string
                    val = read_str()
                elif vtype in SCALAR_FMT:
                    fmt, size = SCALAR_FMT[vtype]
                    val = struct.unpack(fmt, f.read(size))[0]
                elif vtype == 9:  # array — skip
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen  = struct.unpack('<Q', f.read(8))[0]
                    if atype in ARRAY_ELEM_SIZE:
                        f.read(ARRAY_ELEM_SIZE[atype] * alen)
                    elif atype == 8:
                        for _ in range(alen): read_str()
                    else:
                        break   # unknown array element type, abort
                    continue
                else:
                    break        # unknown value type, abort

                if key in TARGET_KEYS:
                    result[key] = val

                # Stop early once we have enough
                if all(k in result for k in (
                    'llama.block_count',
                    'llama.attention.head_count_kv',
                    'llama.embedding_length',
                    'llama.attention.head_count',
                )):
                    break

    except Exception as e:
        print(f"[kv_ctx_limit] Lỗi đọc GGUF: {e}", file=sys.stderr)
        return None

    return result if result else None


# ─── RAM detection ────────────────────────────────────────────────────────────

def get_ram_mb(ram_type='available'):
    """
    Trả về RAM (MB):
      'available' — RAM còn trống (sau khi tính buffer/cache), thường dùng
      'total'     — tổng RAM vật lý
    """
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if ram_type == 'available' and line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
                if ram_type == 'total' and line.startswith('MemTotal:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass

    # fallback: dùng `free`
    try:
        out = subprocess.check_output(['free', '-m'], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.split()
            if parts[0] == 'Mem:':
                if ram_type == 'available' and len(parts) >= 7:
                    return int(parts[6])
                elif ram_type == 'total':
                    return int(parts[1])
    except Exception:
        pass

    return 0


# ─── Core logic ───────────────────────────────────────────────────────────────

def compute_max_ctx(model_path, requested_ctx, limit=0.80,
                    ram_type='available', elem_bytes=2, verbose=False):
    """
    Tính n_ctx tối đa để KV cache paged không vượt quá `limit` × RAM.

    Công thức:
      kv_bytes_per_token = n_layers × 2 × n_kv_heads × head_dim × elem_bytes
      max_ctx = (ram_mb × 1024² × limit) / kv_bytes_per_token
    """
    params = read_gguf_kv_params(model_path) if os.path.isfile(model_path) else None

    # Lấy params từ GGUF, dùng fallback nếu thiếu
    if params:
        n_layers  = int(params.get('llama.block_count',               64))
        n_head_kv = int(params.get('llama.attention.head_count_kv',    8))
        n_head    = int(params.get('llama.attention.head_count',       32))
        n_embd    = int(params.get('llama.embedding_length',         4096))

        # head_dim: ưu tiên key_length nếu có, nếu không thì n_embd / n_head
        head_dim_k = int(params.get('llama.attention.key_length',
                                    n_embd // max(n_head, 1)))
        head_dim_v = int(params.get('llama.attention.value_length',
                                    n_embd // max(n_head, 1)))
    else:
        # Fallback an toàn cho 27B-class model
        n_layers, n_head_kv = 64, 8
        head_dim_k = head_dim_v = 128
        if verbose:
            print("[kv_ctx_limit] Không đọc được GGUF — dùng fallback 27B",
                  file=sys.stderr)

    # KV bytes mỗi token (K + V cho tất cả layers, cả 2 loại)
    kv_bytes_per_token = n_layers * (n_head_kv * head_dim_k + n_head_kv * head_dim_v) * elem_bytes

    # RAM limit
    ram_mb  = get_ram_mb(ram_type)
    limit_mb = ram_mb * limit
    limit_bytes = limit_mb * 1024 * 1024

    max_ctx = int(limit_bytes / kv_bytes_per_token) if kv_bytes_per_token > 0 else requested_ctx

    # Làm tròn xuống bội số 256 (llama.cpp pad n_ctx)
    max_ctx = (max_ctx // 256) * 256
    max_ctx = max(max_ctx, 256)   # tối thiểu 256

    final_ctx = min(requested_ctx, max_ctx)

    if verbose:
        kv_mb_requested = requested_ctx * kv_bytes_per_token / (1024 * 1024)
        kv_mb_final     = final_ctx     * kv_bytes_per_token / (1024 * 1024)
        print(f"[kv_ctx_limit] Model           : {os.path.basename(model_path)}",   file=sys.stderr)
        print(f"[kv_ctx_limit] n_layers        : {n_layers}",                       file=sys.stderr)
        print(f"[kv_ctx_limit] n_kv_heads      : {n_head_kv}",                      file=sys.stderr)
        print(f"[kv_ctx_limit] head_dim (K/V)  : {head_dim_k}/{head_dim_v}",        file=sys.stderr)
        print(f"[kv_ctx_limit] elem_bytes       : {elem_bytes}",                    file=sys.stderr)
        print(f"[kv_ctx_limit] KV bytes/token   : {kv_bytes_per_token:,}",          file=sys.stderr)
        print(f"[kv_ctx_limit] RAM ({ram_type:9s}): {ram_mb:,} MB",                 file=sys.stderr)
        print(f"[kv_ctx_limit] RAM limit ({int(limit*100)}%)  : {limit_mb:,.0f} MB",file=sys.stderr)
        print(f"[kv_ctx_limit] KV @ {requested_ctx} ctx  : {kv_mb_requested:,.0f} MB",file=sys.stderr)
        print(f"[kv_ctx_limit] max_ctx          : {max_ctx}",                       file=sys.stderr)
        print(f"[kv_ctx_limit] → final_ctx      : {final_ctx}"
              + (" (capped)" if final_ctx < requested_ctx else " (OK)"),             file=sys.stderr)
        if final_ctx < requested_ctx:
            print(f"[kv_ctx_limit] KV @ {final_ctx} ctx   : {kv_mb_final:,.0f} MB", file=sys.stderr)

    return final_ctx


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Giới hạn n_ctx paged KV cache theo % RAM'
    )
    parser.add_argument('model',         help='Đường dẫn model .gguf')
    parser.add_argument('requested_ctx', type=int, help='n_ctx yêu cầu')
    parser.add_argument('--limit',       type=float, default=0.80,
                        help='Tỉ lệ RAM tối đa (default: 0.80)')
    parser.add_argument('--ram-type',    default='available',
                        choices=['available', 'total'],
                        help='Loại RAM: available (default) hoặc total')
    parser.add_argument('--elem-bytes',  type=int, default=2,
                        help='Bytes mỗi element KV: 2=f16, 1=q8 (default: 2)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    ctx = compute_max_ctx(
        model_path   = args.model,
        requested_ctx= args.requested_ctx,
        limit        = args.limit,
        ram_type     = args.ram_type,
        elem_bytes   = args.elem_bytes,
        verbose      = args.verbose,
    )
    print(ctx)
