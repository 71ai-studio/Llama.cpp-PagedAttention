/**
 * ggml-turbo.h — TurboQuant KV cache quantization for llama.cpp
 *
 * Implements turbo2/turbo3/turbo4 GGML types:
 *   - PolarQuant (b-1 bits): Walsh-Hadamard rotation + optimal scalar quantization
 *   - QJL (1 bit): sign quantization of PolarQuant residual
 *
 * Memory layout per block of TURBO_BLOCK_SIZE elements:
 *
 *   turbo4 (3-bit PolarQuant + 1-bit QJL = 4 bits/elem):
 *     [48B] 3-bit PolarQuant indices  (128 × 3 bits, packed)
 *     [16B] 1-bit QJL signs           (128 × 1 bit, packed)
 *     [ 2B] f16 vector norm
 *     [ 2B] f16 residual norm
 *     = 68 bytes / 128 elements  →  3.76× vs f16
 *
 *   turbo3 (2-bit PolarQuant + 1-bit QJL = 3 bits/elem):
 *     [32B] 2-bit PolarQuant indices  (128 × 2 bits, packed)
 *     [16B] 1-bit QJL signs
 *     [ 2B] f16 vector norm
 *     [ 2B] f16 residual norm
 *     = 52 bytes / 128 elements  →  4.92× vs f16
 *
 *   turbo2 (1-bit PolarQuant + 1-bit QJL = 2 bits/elem):
 *     [16B] 1-bit PolarQuant indices
 *     [16B] 1-bit QJL signs
 *     [ 2B] f16 vector norm
 *     [ 2B] f16 residual norm
 *     = 36 bytes / 128 elements  →  7.11× vs f16
 *
 * Rotation: per-head WHT (Walsh-Hadamard Transform) with deterministic random
 * sign flips seeded by (layer_idx << 16 | head_idx). O(d log d) per vector.
 *
 * Copyright 2026. Apache 2.0.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Constants ────────────────────────────────────────────────────────────────

// TURBO_BLOCK_SIZE=64 so that ggml_row_size() is correct for all standard
// head_dims (64, 128, 256). head_dim must be a multiple of TURBO_BLOCK_SIZE.
// (128 caused buffer overflow on models with head_dim=64: 68*64/128=34 bytes
//  allocated but 68 bytes needed, overwriting adjacent KV cache memory.)
#define TURBO_BLOCK_SIZE  64    // elements per quantization block
#define TURBO_WHT_MAX_DIM 128   // max head_dim supported by WHT (power of 2)

// Block sizes in bytes (64 elements each)
#define TURBO4_BLOCK_BYTES  36  // 24 + 8 + 2 + 2  (64×3b PQ + 64×1b QJL + 2×f16)
#define TURBO3_BLOCK_BYTES  28  // 16 + 8 + 2 + 2  (64×2b PQ + 64×1b QJL + 2×f16)
#define TURBO2_BLOCK_BYTES  20  //  8 + 8 + 2 + 2  (64×1b PQ + 64×1b QJL + 2×f16)

// ─── Block layout structs ──────────────────────────────────────────────────────

// turbo4 block: 3-bit PolarQuant + 1-bit QJL (64 elements)
typedef struct {
    uint8_t  pq_indices[24]; // 64 × 3 bits packed
    uint8_t  qjl_signs[8];   // 64 × 1 bit packed
    uint16_t vec_norm;        // f16 — original vector L2 norm
    uint16_t res_norm;        // f16 — residual L2 norm after PolarQuant
} block_turbo4;

// C11 static assert
#ifndef __cplusplus
#  define TURBO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
#  define TURBO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#endif

TURBO_STATIC_ASSERT(sizeof(block_turbo4) == TURBO4_BLOCK_BYTES, "block_turbo4 size mismatch");

// turbo3 block: 2-bit PolarQuant + 1-bit QJL (64 elements)
typedef struct {
    uint8_t  pq_indices[16]; // 64 × 2 bits packed
    uint8_t  qjl_signs[8];   // 64 × 1 bit packed
    uint16_t vec_norm;        // f16
    uint16_t res_norm;        // f16
} block_turbo3;

TURBO_STATIC_ASSERT(sizeof(block_turbo3) == TURBO3_BLOCK_BYTES, "block_turbo3 size mismatch");

// turbo2 block: 1-bit PolarQuant + 1-bit QJL (64 elements)
typedef struct {
    uint8_t  pq_indices[8];  // 64 × 1 bit packed
    uint8_t  qjl_signs[8];   // 64 × 1 bit packed
    uint16_t vec_norm;        // f16
    uint16_t res_norm;        // f16
} block_turbo2;

TURBO_STATIC_ASSERT(sizeof(block_turbo2) == TURBO2_BLOCK_BYTES, "block_turbo2 size mismatch");

// ─── Codebooks (precomputed optimal Lloyd centroids) ──────────────────────────
// Scaled for head_dim=128. For other dims, scale by sqrt(128/d).

// 1-bit codebook: [-c, +c], c = sqrt(2/(π·d))
// 2-bit codebook: optimal for Gaussian N(0, 1/d)
// 3-bit codebook: optimal for Gaussian N(0, 1/d)

extern const float TURBO_CODEBOOK_1BIT[2];
extern const float TURBO_CODEBOOK_2BIT[4];
extern const float TURBO_CODEBOOK_3BIT[8];

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Quantize a block of f32 KV vectors into turbo4 format.
 *
 * @param x       Input: f32 array of n_blocks * TURBO_BLOCK_SIZE elements
 * @param y       Output: array of n_blocks block_turbo4 structs
 * @param n       Total number of f32 elements (must be multiple of TURBO_BLOCK_SIZE)
 * @param seed    WHT rotation seed (encode layer_idx and head_idx)
 */
void quantize_row_turbo4(const float * x, block_turbo4 * y, int64_t n, uint32_t seed);
void quantize_row_turbo3(const float * x, block_turbo3 * y, int64_t n, uint32_t seed);
void quantize_row_turbo2(const float * x, block_turbo2 * y, int64_t n, uint32_t seed);

/**
 * Dequantize a block of turbo-compressed data back to f32.
 */
void dequantize_row_turbo4(const block_turbo4 * x, float * y, int64_t n, uint32_t seed);
void dequantize_row_turbo3(const block_turbo3 * x, float * y, int64_t n, uint32_t seed);
void dequantize_row_turbo2(const block_turbo2 * x, float * y, int64_t n, uint32_t seed);

/**
 * Generate deterministic WHT rotation sign vectors for a given seed.
 * signs1 and signs2 must be float arrays of length padded_dim (next power of 2 >= dim).
 */
void turbo_wht_signs(uint32_t seed, float * signs1, float * signs2, int dim);

/**
 * Fast Walsh-Hadamard Transform in-place, O(n log n).
 * n must be a power of 2. Result is normalized by 1/sqrt(n).
 */
void fast_wht(float * x, int n);

/**
 * Default wrappers with seed=0 for ggml type_traits to_float/from_float.
 * Used by ggml_cast and other ggml ops that don't carry seed context.
 * NOTE: these use seed=0 — correct seed injection happens at KV write/read time in llama.cpp.
 */
void dequantize_row_turbo4_default(const block_turbo4 * x, float * y, int64_t k);
void quantize_row_turbo4_default(const float * x, void * y, int64_t k);
void dequantize_row_turbo3_default(const block_turbo3 * x, float * y, int64_t k);
void quantize_row_turbo3_default(const float * x, void * y, int64_t k);
void dequantize_row_turbo2_default(const block_turbo2 * x, float * y, int64_t k);
void quantize_row_turbo2_default(const float * x, void * y, int64_t k);

/**
 * CPU vec_dot functions: dequantize turboN block then F32 dot product.
 * Signature matches ggml_vec_dot_t from ggml-cpu.h.
 * x = quantized K (turboN), y = F32 query.
 */
void ggml_vec_dot_turbo4_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc);
void ggml_vec_dot_turbo3_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc);
void ggml_vec_dot_turbo2_f32(int n, float * s, size_t bs,
                              const void * x, size_t bx,
                              const void * y, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
