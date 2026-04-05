#!/usr/bin/env python3
"""
auto_ngl.py — tính số GPU layers tối đa có thể offload dựa trên VRAM và model
Usage: python3 auto_ngl.py <model.gguf> [vram_reserve_mb=1500]
Output: số nguyên (ngl) in ra stdout
"""

import struct, os, sys, subprocess

def read_block_count(path):
    """Đọc n_layers từ GGUF metadata."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            return None
        f.read(4)   # version
        f.read(8)   # n_tensors
        n_kv = struct.unpack('<Q', f.read(8))[0]

        def read_str():
            l = struct.unpack('<Q', f.read(8))[0]
            return f.read(l).decode('utf-8', errors='replace')

        for _ in range(min(n_kv, 500)):
            try:
                key = read_str()
                vtype = struct.unpack('<I', f.read(4))[0]

                if vtype == 8:    val = read_str()
                elif vtype == 4:  val = struct.unpack('<I', f.read(4))[0]
                elif vtype == 5:  val = struct.unpack('<i', f.read(4))[0]
                elif vtype == 10: val = struct.unpack('<Q', f.read(8))[0]
                elif vtype == 6:  val = struct.unpack('<f', f.read(4))[0]
                elif vtype == 7:  val = struct.unpack('<?', f.read(1))[0]
                elif vtype == 0:  val = struct.unpack('<B', f.read(1))[0]
                elif vtype == 1:  val = struct.unpack('<b', f.read(1))[0]
                elif vtype == 2:  val = struct.unpack('<H', f.read(2))[0]
                elif vtype == 3:  val = struct.unpack('<h', f.read(2))[0]
                elif vtype == 11: val = struct.unpack('<q', f.read(8))[0]
                elif vtype == 12: val = struct.unpack('<d', f.read(8))[0]
                elif vtype == 9:  # array — skip
                    atype = struct.unpack('<I', f.read(4))[0]
                    alen  = struct.unpack('<Q', f.read(8))[0]
                    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
                    if atype in sizes:
                        f.read(sizes[atype] * alen)
                    elif atype == 8:
                        for _ in range(alen): read_str()
                    else: break
                    continue
                else:
                    break

                if key.endswith('.block_count'):
                    return int(val)

            except Exception:
                break
    return None


def get_vram_total_mb():
    """Lấy tổng VRAM từ nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, text=True
        )
        return int(out.strip().split('\n')[0].strip())
    except Exception:
        return 0


def get_vram_free_mb():
    """Lấy VRAM còn trống (sau khi trừ OS và process khác)."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, text=True
        )
        return int(out.strip().split('\n')[0].strip())
    except Exception:
        return 0


def calc_ngl(model_path, vram_reserve_mb=1500, verbose=False):
    """
    Tính ngl tối đa.

    Logic:
      - model_size_mb     : kích thước file (xấp xỉ tổng weights)
      - non_layer_mb      : embedding + lm_head + misc (~15% model size)
      - layer_mb          : (model_size - non_layer) / n_layers
      - usable_vram_mb    : vram_free - reserve
      - ngl               : floor(usable_vram / layer_mb)
    """
    if not os.path.exists(model_path):
        if verbose: print(f"[auto_ngl] File không tồn tại: {model_path}", file=sys.stderr)
        return 0

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    n_layers      = read_block_count(model_path)

    if n_layers is None or n_layers <= 0:
        if verbose: print("[auto_ngl] Không đọc được block_count", file=sys.stderr)
        return 0

    # Non-layer overhead: embedding table + lm_head + misc ≈ 15% model size
    non_layer_mb = model_size_mb * 0.15
    layer_mb     = (model_size_mb - non_layer_mb) / n_layers

    vram_free_mb  = get_vram_free_mb()
    vram_total_mb = get_vram_total_mb()

    # Dùng free VRAM trừ reserve; trừ thêm 1 layer để tránh OOM khi
    # VRAM thực tế per-layer cao hơn ước tính (scratch buffers, output tensors)
    usable_mb = max(0, vram_free_mb - vram_reserve_mb)
    ngl       = max(0, int(usable_mb / layer_mb) - 1)
    ngl       = min(ngl, n_layers)
    ngl       = max(ngl, 0)

    if verbose:
        print(f"[auto_ngl] Model         : {os.path.basename(model_path)}", file=sys.stderr)
        print(f"[auto_ngl] Model size    : {model_size_mb:.0f} MB", file=sys.stderr)
        print(f"[auto_ngl] n_layers      : {n_layers}", file=sys.stderr)
        print(f"[auto_ngl] per-layer     : {layer_mb:.0f} MB", file=sys.stderr)
        print(f"[auto_ngl] VRAM total    : {vram_total_mb} MB", file=sys.stderr)
        print(f"[auto_ngl] VRAM free     : {vram_free_mb} MB", file=sys.stderr)
        print(f"[auto_ngl] VRAM usable   : {usable_mb} MB (after {vram_reserve_mb}MB reserve)", file=sys.stderr)
        print(f"[auto_ngl] → ngl         : {ngl} / {n_layers}", file=sys.stderr)

    return ngl


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.gguf> [vram_reserve_mb]", file=sys.stderr)
        sys.exit(1)

    model_path    = sys.argv[1]
    args_rest     = [a for a in sys.argv[2:] if not a.startswith('-')]
    reserve_mb    = int(args_rest[0]) if args_rest else 1500
    verbose       = '--verbose' in sys.argv or '-v' in sys.argv

    ngl = calc_ngl(model_path, reserve_mb, verbose=True)
    print(ngl)
