/**
 * ggml-turbo.c — TurboQuant KV cache quantization implementation
 *
 * Algorithm (per vector of TURBO_BLOCK_SIZE elements):
 *
 *   Quantize:
 *     1. Extract L2 norm, normalize x to unit norm
 *     2. Apply WHT rotation: x' = D2 · H · D1 · x  (D = diagonal random signs)
 *     3. PolarQuant: find nearest centroid per coordinate → b-1 bit index
 *     4. Reconstruct PolarQuant approximation x̂', inverse rotate → x̂
 *     5. Compute residual r = x - x̂
 *     6. QJL: compute sign(P · r) for random projection P → 1-bit signs
 *     7. Store: pq_indices, qjl_signs, vec_norm (f16), res_norm (f16)
 *
 *   Dequantize:
 *     1. Reconstruct PolarQuant: look up centroids, inverse rotate, rescale by vec_norm
 *     2. Reconstruct QJL: x̃_qjl = sqrt(π/2) / d · res_norm · P^T · qjl_signs
 *     3. Return x̂ + x̃_qjl
 */

#include "ggml-turbo.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// ─── Codebooks ────────────────────────────────────────────────────────────────
// Baseline centroids for d=128 (sigma = 1/sqrt(128) = 0.08839).
// For d != 128, codebook_scale(d) = sqrt(128/d) corrects the centroid values
// so they match N(0, 1/sqrt(d)) at runtime (e.g. d=64 → scale=sqrt(2)).

// 1-bit: c = sqrt(2/(π·128)) = 0.07064
const float TURBO_CODEBOOK_1BIT[2] = { -0.07064f, 0.07064f };

// 2-bit: paper formula scaled by 1/sqrt(128)
const float TURBO_CODEBOOK_2BIT[4] = {
    -0.13363f,  // -1.51 / sqrt(128)
    -0.04003f,  // -0.453 / sqrt(128)
     0.04003f,
     0.13363f,
};

// 3-bit: Lloyd's algorithm on N(0, 1/128), 8 centroids
// Values obtained from running turboquant/codebook.py _lloyds_gaussian(8, 0.08839)
const float TURBO_CODEBOOK_3BIT[8] = {
    -0.19182f,
    -0.12549f,
    -0.07564f,
    -0.02566f,
     0.02566f,
     0.07564f,
     0.12549f,
     0.19182f,
};

// Codebook scale factor for non-128 dimensions: multiply centroids by sqrt(128/d)
static inline float codebook_scale(int d) {
    return sqrtf(128.0f / (float)d);
}

// ─── Float16 helpers ──────────────────────────────────────────────────────────

static inline uint16_t f32_to_f16(float x) {
    return GGML_FP32_TO_FP16(x);
}

static inline float f16_to_f32(uint16_t x) {
    return GGML_FP16_TO_FP32(x);
}

// ─── Fast Walsh-Hadamard Transform ────────────────────────────────────────────

void fast_wht(float * x, int n) {
    // Cooley-Tukey-style butterfly, O(n log n)
    // n must be a power of 2
    assert(n > 0 && (n & (n - 1)) == 0);

    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
        h <<= 1;
    }

    // Normalize by 1/sqrt(n)
    float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sqrt_n;
    }
}

// ─── Deterministic sign generation (xorshift32) ───────────────────────────────

static uint32_t xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Generate WHT rotation sign vectors for given seed and dimension
void turbo_wht_signs(uint32_t seed, float * signs1, float * signs2, int dim) {
    // Pad to next power of 2
    int padded = 1;
    while (padded < dim) padded <<= 1;

    uint32_t state1 = seed ^ 0xDEADBEEF;
    uint32_t state2 = seed ^ 0xCAFEBABE;

    for (int i = 0; i < padded; i++) {
        signs1[i] = (xorshift32(&state1) & 1) ? 1.0f : -1.0f;
        signs2[i] = (xorshift32(&state2) & 1) ? 1.0f : -1.0f;
    }
}

// ─── Apply WHT rotation to a vector ──────────────────────────────────────────
// Computes: y = D2 · H · D1 · x   (D = diagonal of signs)
// x and work must be float arrays of size padded_dim.

static void apply_rotation(
    const float * x, float * work,
    const float * signs1, const float * signs2,
    int dim, int padded_dim
) {
    // Copy and zero-pad
    memcpy(work, x, dim * sizeof(float));
    memset(work + dim, 0, (padded_dim - dim) * sizeof(float));

    // D1
    for (int i = 0; i < padded_dim; i++) {
        work[i] *= signs1[i];
    }
    // H
    fast_wht(work, padded_dim);
    // D2
    for (int i = 0; i < padded_dim; i++) {
        work[i] *= signs2[i];
    }
}

// Inverse rotation: y = D1^T · H^T · D2^T · x = D1 · H · D2 · x
// (D and H are self-inverse up to normalization)
static void apply_rotation_inverse(
    const float * x, float * work,
    const float * signs1, const float * signs2,
    int dim, int padded_dim
) {
    memcpy(work, x, dim * sizeof(float));
    memset(work + dim, 0, (padded_dim - dim) * sizeof(float));

    // Reverse order: D2, H, D1
    for (int i = 0; i < padded_dim; i++) {
        work[i] *= signs2[i];
    }
    fast_wht(work, padded_dim);
    for (int i = 0; i < padded_dim; i++) {
        work[i] *= signs1[i];
    }
}

// ─── Nearest centroid (binary search on sorted array) ────────────────────────

static inline int nearest_centroid(float val, const float * centroids, int n) {
    // Find nearest centroid using boundary midpoints (sorted centroids)
    // For small n (2, 4, 8) this unrolls well
    int lo = 0, hi = n - 1;
    if (n == 2) {
        return (val < (centroids[0] + centroids[1]) * 0.5f) ? 0 : 1;
    }
    // General: find insertion point among (n-1) boundaries
    float boundary;
    int mid;
    while (lo < hi) {
        mid = (lo + hi) / 2;
        boundary = (centroids[mid] + centroids[mid + 1]) * 0.5f;
        if (val <= boundary) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// ─── Bit packing helpers ─────────────────────────────────────────────────────
// Use absolute bit offset to avoid stateful bugs

// Pack n 3-bit values into bytes. out must be zeroed first.
static void pack_3bit(const uint8_t * indices, uint8_t * out, int n) {
    int n_bytes = (n * 3 + 7) / 8;
    memset(out, 0, (size_t)n_bytes);
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx   = bit_offset / 8;
        int bit_shift  = bit_offset % 8;
        uint8_t v = indices[i] & 0x7;
        out[byte_idx] |= (uint8_t)(v << bit_shift);
        if (bit_shift + 3 > 8) {
            out[byte_idx + 1] |= (uint8_t)(v >> (8 - bit_shift));
        }
    }
}

static void unpack_3bit(const uint8_t * in, uint8_t * indices, int n) {
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx   = bit_offset / 8;
        int bit_shift  = bit_offset % 8;
        uint8_t v = (in[byte_idx] >> bit_shift) & 0x7;
        if (bit_shift + 3 > 8) {
            v |= (uint8_t)((in[byte_idx + 1] << (8 - bit_shift)) & 0x7);
        }
        indices[i] = v;
    }
}

// Pack n 2-bit values into bytes (4 per byte)
static void pack_2bit(const uint8_t * indices, uint8_t * out, int n) {
    memset(out, 0, (size_t)((n + 3) / 4));
    for (int i = 0; i < n; i++) {
        int byte_i = i / 4;
        int shift  = (i % 4) * 2;
        out[byte_i] |= (uint8_t)((indices[i] & 0x3) << shift);
    }
}

static void unpack_2bit(const uint8_t * in, uint8_t * indices, int n) {
    for (int i = 0; i < n; i++) {
        int byte_i = i / 4;
        int shift  = (i % 4) * 2;
        indices[i] = (in[byte_i] >> shift) & 0x3;
    }
}

// Pack n 1-bit values into bytes (8 per byte)
static void pack_1bit(const uint8_t * indices, uint8_t * out, int n) {
    memset(out, 0, (size_t)((n + 7) / 8));
    for (int i = 0; i < n; i++) {
        int byte_i = i / 8;
        int shift  = i % 8;
        out[byte_i] |= (uint8_t)((indices[i] & 0x1) << shift);
    }
}

static void unpack_1bit(const uint8_t * in, uint8_t * indices, int n) {
    for (int i = 0; i < n; i++) {
        int byte_i = i / 8;
        int shift  = i % 8;
        indices[i] = (in[byte_i] >> shift) & 0x1;
    }
}

// ─── QJL: sign-projection of residual ────────────────────────────────────────
// Structured QJL: S = H @ D  where D = random diagonal ±1, H = normalized WHT.
// O(d log d) — reuses fast_wht already used by PolarQuant.
// Each output bit depends on ALL d inputs (full mixing via WHT), unlike diagonal
// which only sees 1 element. Inverse: S^T = D^T @ H^T = D @ H (since D²=I, H²=I).
//
// Quantize:  proj = fast_wht(D * residual),  signs = sign(proj)
// Dequant:   r̃   = scale * D * fast_wht(signs)

static void qjl_quantize(
    const float * residual, uint8_t * signs_out,
    float * res_norm_out, uint32_t seed, int d
) {
    float norm2 = 0.0f;
    for (int i = 0; i < d; i++) norm2 += residual[i] * residual[i];
    *res_norm_out = sqrtf(norm2);

    // Apply D (random diagonal ±1) then WHT
    float proj[TURBO_BLOCK_SIZE];
    uint32_t state = seed ^ 0xF00DCAFE;
    for (int i = 0; i < d; i++) {
        proj[i] = ((xorshift32(&state) & 1) ? 1.0f : -1.0f) * residual[i];
    }
    fast_wht(proj, d);

    uint8_t raw_signs[TURBO_BLOCK_SIZE];
    for (int i = 0; i < d; i++) {
        raw_signs[i] = (proj[i] >= 0.0f) ? 1 : 0;
    }
    pack_1bit(raw_signs, signs_out, d);
}

static void qjl_dequantize(
    const uint8_t * signs_in, float * out,
    float res_norm, uint32_t seed, int d
) {
    uint8_t raw_signs[TURBO_BLOCK_SIZE];
    unpack_1bit(signs_in, raw_signs, d);

    // S^T = D @ H: apply WHT then same D
    // r̃ = √(π/2) / d * res_norm * D * fast_wht(signs)
    float scale = sqrtf((float)M_PI / 2.0f) / (float)d * res_norm;

    for (int i = 0; i < d; i++) out[i] = raw_signs[i] ? 1.0f : -1.0f;
    fast_wht(out, d);

    uint32_t state = seed ^ 0xF00DCAFE;
    for (int i = 0; i < d; i++) {
        float d_sign = (xorshift32(&state) & 1) ? 1.0f : -1.0f;
        out[i] *= scale * d_sign;
    }
}

// ─── Core quantize logic (shared across turbo2/3/4) ──────────────────────────

typedef struct {
    int pq_bits;         // bits for PolarQuant stage (bit_width - 1)
    int total_bits;      // total bits per element
    const float * codebook;  // PolarQuant codebook
    int n_centroids;     // 2^pq_bits
} turbo_config_t;

static const turbo_config_t TURBO4_CFG = { 3, 4, TURBO_CODEBOOK_3BIT, 8 };
static const turbo_config_t TURBO3_CFG = { 2, 3, TURBO_CODEBOOK_2BIT, 4 };
static const turbo_config_t TURBO2_CFG = { 1, 2, TURBO_CODEBOOK_1BIT, 2 };

// Quantize one block of `block_size` f32 elements.
// x_rot_work: scratch buffer of size padded_dim
static void turbo_quantize_block(
    const float * x,
    uint8_t * pq_out,     // packed PolarQuant indices
    uint8_t * qjl_out,    // packed QJL signs
    uint16_t * vec_norm_out,
    uint16_t * res_norm_out,
    int dim, int padded_dim,
    const float * signs1, const float * signs2,
    uint32_t qjl_seed,
    const turbo_config_t * cfg,
    float * work           // scratch float[padded_dim]
) {
    // 1. Extract norm and normalize
    float norm2 = 0.0f;
    for (int i = 0; i < dim; i++) norm2 += x[i] * x[i];
    float norm = sqrtf(norm2);
    *vec_norm_out = f32_to_f16(norm);

    float x_norm[TURBO_BLOCK_SIZE];
    float inv_norm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;
    for (int i = 0; i < dim; i++) x_norm[i] = x[i] * inv_norm;

    // 2. Rotate: work = D2 · H · D1 · x_norm
    apply_rotation(x_norm, work, signs1, signs2, dim, padded_dim);

    // 3. PolarQuant: find nearest centroid per coordinate
    // Scale codebook for this dimension
    float scale = codebook_scale(dim);
    uint8_t indices[TURBO_BLOCK_SIZE];
    float reconstructed_rot[TURBO_BLOCK_SIZE]; // centroid values in rotated domain
    for (int i = 0; i < dim; i++) {
        int idx = nearest_centroid(work[i], cfg->codebook, cfg->n_centroids);
        indices[i] = (uint8_t)idx;
        reconstructed_rot[i] = cfg->codebook[idx] * scale;
    }

    // Pack PolarQuant indices
    if (cfg->pq_bits == 3) pack_3bit(indices, pq_out, dim);
    else if (cfg->pq_bits == 2) pack_2bit(indices, pq_out, dim);
    else pack_1bit(indices, pq_out, dim);

    // 4. Reconstruct PolarQuant approximation: inverse rotate reconstructed_rot
    float recon_work[TURBO_BLOCK_SIZE];
    // Norm correction: re-normalize reconstructed vector to unit norm before inverse rotate
    float recon_norm2 = 0.0f;
    for (int i = 0; i < dim; i++) recon_norm2 += reconstructed_rot[i] * reconstructed_rot[i];
    float recon_norm = sqrtf(recon_norm2);
    if (recon_norm > 1e-12f) {
        float inv_rn = 1.0f / recon_norm;
        for (int i = 0; i < dim; i++) reconstructed_rot[i] *= inv_rn;
    }

    apply_rotation_inverse(reconstructed_rot, recon_work, signs1, signs2, dim, padded_dim);

    // Rescale by original norm
    float x_hat[TURBO_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) x_hat[i] = recon_work[i] * norm;

    // 5. Residual
    float residual[TURBO_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) residual[i] = x[i] - x_hat[i];

    // 6. QJL on residual
    float res_norm_f;
    qjl_quantize(residual, qjl_out, &res_norm_f, qjl_seed, dim);
    *res_norm_out = f32_to_f16(res_norm_f);
}

static void turbo_dequantize_block(
    const uint8_t * pq_in,
    const uint8_t * qjl_in,
    uint16_t vec_norm_enc,
    uint16_t res_norm_enc,
    float * out,
    int dim, int padded_dim,
    const float * signs1, const float * signs2,
    uint32_t qjl_seed,
    const turbo_config_t * cfg,
    float * work
) {
    float vec_norm = f16_to_f32(vec_norm_enc);
    float res_norm = f16_to_f32(res_norm_enc);
    float scale    = codebook_scale(dim);

    // 1. Unpack PolarQuant indices → centroid values in rotated domain
    uint8_t indices[TURBO_BLOCK_SIZE];
    if (cfg->pq_bits == 3) unpack_3bit(pq_in, indices, dim);
    else if (cfg->pq_bits == 2) unpack_2bit(pq_in, indices, dim);
    else unpack_1bit(pq_in, indices, dim);

    float recon_rot[TURBO_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) {
        recon_rot[i] = cfg->codebook[indices[i]] * scale;
    }

    // Norm correction: unit-normalize
    float rn2 = 0.0f;
    for (int i = 0; i < dim; i++) rn2 += recon_rot[i] * recon_rot[i];
    float rn = sqrtf(rn2);
    if (rn > 1e-12f) {
        float inv_rn = 1.0f / rn;
        for (int i = 0; i < dim; i++) recon_rot[i] *= inv_rn;
    }

    // Inverse rotate → x_hat_unit
    apply_rotation_inverse(recon_rot, work, signs1, signs2, dim, padded_dim);

    // Rescale by vec_norm
    float x_hat[TURBO_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) x_hat[i] = work[i] * vec_norm;

    // 2. QJL correction
    float qjl_correction[TURBO_BLOCK_SIZE];
    qjl_dequantize(qjl_in, qjl_correction, res_norm, qjl_seed, dim);

    // 3. Final output
    for (int i = 0; i < dim; i++) out[i] = x_hat[i] + qjl_correction[i];
}

// ─── Public API ──────────────────────────────────────────────────────────────

// Seed construction: encodes context_id (layer+head) and block index
static inline uint32_t make_seed(uint32_t ctx_seed, int64_t block_idx) {
    return ctx_seed ^ (uint32_t)(block_idx * 2654435761ULL);
}

#define IMPLEMENT_QUANTIZE(bits, BlockType, cfg_ptr, pq_field, pq_bytes)      \
void quantize_row_turbo##bits(                                                 \
    const float * x, BlockType * y, int64_t n, uint32_t seed                 \
) {                                                                            \
    int64_t n_full  = (n / TURBO_BLOCK_SIZE) * TURBO_BLOCK_SIZE;             \
    int64_t n_blocks = n / TURBO_BLOCK_SIZE;                                  \
    int64_t remainder = n - n_full;                                            \
    int dim = TURBO_BLOCK_SIZE;                                               \
    int padded = dim;                                                          \
    float signs1[TURBO_BLOCK_SIZE], signs2[TURBO_BLOCK_SIZE];                 \
    turbo_wht_signs(seed, signs1, signs2, dim);                                \
    float work[TURBO_BLOCK_SIZE];                                              \
    for (int64_t b = 0; b < n_blocks; b++) {                                  \
        memset(&y[b], 0, sizeof(BlockType));                                  \
        uint32_t qjl_seed = make_seed(seed ^ 0x1234, b);                      \
        turbo_quantize_block(                                                  \
            x + b * dim,                                                      \
            y[b].pq_indices, y[b].qjl_signs,                                  \
            &y[b].vec_norm, &y[b].res_norm,                                   \
            dim, padded, signs1, signs2, qjl_seed,                            \
            (cfg_ptr), work                                                    \
        );                                                                     \
    }                                                                          \
    /* Handle partial last block: zero-pad to TURBO_BLOCK_SIZE */             \
    if (remainder > 0) {                                                       \
        float pad[TURBO_BLOCK_SIZE];                                          \
        memset(pad, 0, sizeof(pad));                                           \
        memcpy(pad, x + n_full, remainder * sizeof(float));                   \
        memset(&y[n_blocks], 0, sizeof(BlockType));                           \
        uint32_t qjl_seed = make_seed(seed ^ 0x1234, n_blocks);              \
        turbo_quantize_block(                                                  \
            pad,                                                               \
            y[n_blocks].pq_indices, y[n_blocks].qjl_signs,                   \
            &y[n_blocks].vec_norm, &y[n_blocks].res_norm,                     \
            dim, padded, signs1, signs2, qjl_seed,                            \
            (cfg_ptr), work                                                    \
        );                                                                     \
    }                                                                          \
}

#define IMPLEMENT_DEQUANTIZE(bits, BlockType, cfg_ptr)                        \
void dequantize_row_turbo##bits(                                               \
    const BlockType * x, float * y, int64_t n, uint32_t seed                 \
) {                                                                            \
    int64_t n_full   = (n / TURBO_BLOCK_SIZE) * TURBO_BLOCK_SIZE;            \
    int64_t n_blocks  = n / TURBO_BLOCK_SIZE;                                 \
    int64_t remainder = n - n_full;                                            \
    int dim = TURBO_BLOCK_SIZE;                                               \
    int padded = dim;                                                          \
    float signs1[TURBO_BLOCK_SIZE], signs2[TURBO_BLOCK_SIZE];                 \
    turbo_wht_signs(seed, signs1, signs2, dim);                                \
    float work[TURBO_BLOCK_SIZE];                                              \
    for (int64_t b = 0; b < n_blocks; b++) {                                  \
        uint32_t qjl_seed = make_seed(seed ^ 0x1234, b);                      \
        turbo_dequantize_block(                                                \
            x[b].pq_indices, x[b].qjl_signs,                                  \
            x[b].vec_norm, x[b].res_norm,                                     \
            y + b * dim,                                                       \
            dim, padded, signs1, signs2, qjl_seed,                            \
            (cfg_ptr), work                                                    \
        );                                                                     \
    }                                                                          \
    /* Handle partial last block: dequantize to temp buffer, copy remainder */\
    if (remainder > 0) {                                                       \
        float tmp[TURBO_BLOCK_SIZE];                                           \
        uint32_t qjl_seed = make_seed(seed ^ 0x1234, n_blocks);              \
        turbo_dequantize_block(                                                \
            x[n_blocks].pq_indices, x[n_blocks].qjl_signs,                   \
            x[n_blocks].vec_norm, x[n_blocks].res_norm,                       \
            tmp,                                                               \
            dim, padded, signs1, signs2, qjl_seed,                            \
            (cfg_ptr), work                                                    \
        );                                                                     \
        memcpy(y + n_full, tmp, remainder * sizeof(float));                   \
    }                                                                          \
}

IMPLEMENT_QUANTIZE(4, block_turbo4, &TURBO4_CFG, pq_indices, 48)
IMPLEMENT_QUANTIZE(3, block_turbo3, &TURBO3_CFG, pq_indices, 32)
IMPLEMENT_QUANTIZE(2, block_turbo2, &TURBO2_CFG, pq_indices, 16)

IMPLEMENT_DEQUANTIZE(4, block_turbo4, &TURBO4_CFG)
IMPLEMENT_DEQUANTIZE(3, block_turbo3, &TURBO3_CFG)
IMPLEMENT_DEQUANTIZE(2, block_turbo2, &TURBO2_CFG)

// ─── Default wrappers (seed=0) for ggml type_traits to_float/from_float ─────
// These are called by ggml ops that don't carry seed context (e.g. ggml_cast).
// Seed=0 gives deterministic but per-block-consistent results for ggml internals.

void dequantize_row_turbo4_default(const block_turbo4 * x, float * y, int64_t k) {
    dequantize_row_turbo4(x, y, k, 0);
}
void quantize_row_turbo4_default(const float * x, void * y, int64_t k) {
    quantize_row_turbo4(x, (block_turbo4 *)y, k, 0);
}

void dequantize_row_turbo3_default(const block_turbo3 * x, float * y, int64_t k) {
    dequantize_row_turbo3(x, y, k, 0);
}
void quantize_row_turbo3_default(const float * x, void * y, int64_t k) {
    quantize_row_turbo3(x, (block_turbo3 *)y, k, 0);
}

void dequantize_row_turbo2_default(const block_turbo2 * x, float * y, int64_t k) {
    dequantize_row_turbo2(x, y, k, 0);
}
void quantize_row_turbo2_default(const float * x, void * y, int64_t k) {
    quantize_row_turbo2(x, (block_turbo2 *)y, k, 0);
}

// ─── CPU vec_dot: dequantize then F32 dot product ────────────────────────────
// Used by ggml-cpu flash_attn for KV cache dot products.
// Signature matches ggml_vec_dot_t: x is turboN (quantized K), y is F32 (query).

void ggml_vec_dot_turbo4_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    float tmp[TURBO_WHT_MAX_DIM];
    dequantize_row_turbo4_default((const block_turbo4 *)x, tmp, (int64_t)n);
    float sum = 0.0f;
    const float * yf = (const float *)y;
    for (int i = 0; i < n; i++) sum += tmp[i] * yf[i];
    *s = sum;
}

void ggml_vec_dot_turbo3_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    float tmp[TURBO_WHT_MAX_DIM];
    dequantize_row_turbo3_default((const block_turbo3 *)x, tmp, (int64_t)n);
    float sum = 0.0f;
    const float * yf = (const float *)y;
    for (int i = 0; i < n; i++) sum += tmp[i] * yf[i];
    *s = sum;
}

void ggml_vec_dot_turbo2_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc) {
    (void)bs; (void)bx; (void)by; (void)nrc;
    float tmp[TURBO_WHT_MAX_DIM];
    dequantize_row_turbo2_default((const block_turbo2 *)x, tmp, (int64_t)n);
    float sum = 0.0f;
    const float * yf = (const float *)y;
    for (int i = 0; i < n; i++) sum += tmp[i] * yf[i];
    *s = sum;
}
