/*
 * Native GGML K-quant GEMM for the Qwen3.8 Q6_K GGUF target.
 *
 * The checkpoint stays in its 144/176/210/34-byte block format.  The kernels
 * decode one weight value in registers while multiplying it with BF16
 * activations; no full BF16 copy of a layer is created.  The small-M path is
 * intentionally a warp-per-output GEMV, because autoregressive decode is the
 * dominant Qwen3.8 workload.  The tiled path computes eight output rows and
 * eight input rows per CTA and is used for prefill/verify.
 *
 * Layout and decode order follow llama.cpp's block_q4_K/block_q5_K/
 * block_q6_K/block_q8_0 definitions.  This file has a raw pointer ABI so it
 * does not depend on libtorch, vLLM, or a Python ABI.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>

#define QSR_EXPORT __attribute__((visibility("default")))

namespace {

enum QuantType : int {
    kQ4K = 0,
    kQ5K = 1,
    kQ6K = 2,
    kQ8_0 = 3,
    // Internal storage layout: Q6_K blocks are padded from 210 to 224 bytes
    // so every block and every 32-bit payload load is naturally aligned.
    kQ6KAligned = 4,
    // Internal storage layout: the 208-byte Q6 payload is kept contiguous per
    // row and the 2-byte block scales are moved to a row-tail array.  The
    // total row footprint remains the standard 210 bytes per block.
    kQ6KSplit = 5,
    // Internal storage layout: Q8_0's 32-byte code payload is kept contiguous
    // per row and the 2-byte block scales are moved to a row-tail array.  The
    // total row footprint remains the standard 34 bytes per block.
    kQ8_0Split = 6,
};

struct Q8_1Block {
    half2 ds;
    int8_t qs[32];
};

static_assert(sizeof(Q8_1Block) == 36, "GGML Q8_1 layout must stay packed");

__device__ __forceinline__ float load_f16(const uint8_t *ptr) {
    return __half2float(*reinterpret_cast<const __half *>(ptr));
}

__device__ __forceinline__ float load_bf16(const __nv_bfloat16 *ptr) {
    return __bfloat162float(*ptr);
}

__device__ __forceinline__ float load_activation(const __nv_bfloat16 *ptr) {
    return load_bf16(ptr);
}

__device__ __forceinline__ float load_activation(const float *ptr) {
    return *ptr;
}

__device__ __forceinline__ void store_activation(__nv_bfloat16 *ptr, float value) {
    *ptr = __float2bfloat16_rn(value);
}

__device__ __forceinline__ void store_activation(float *ptr, float value) {
    *ptr = value;
}

// The correctness oracle materializes each decoded GGML value as BF16 before
// F.linear consumes the matrix.  Keep the direct packed path on the same
// contract; multiplying the unrounded FP32 decode is a subtly different
// weight, especially in the recurrent GDN projections.
__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

template <typename T>
__device__ __forceinline__ float weight_for_compute(float value) {
    return value;
}

template <>
__device__ __forceinline__ float weight_for_compute<__nv_bfloat16>(float value) {
    return round_bf16(value);
}

__device__ __forceinline__ float k_scale(const uint8_t *scales, int index) {
    if (index < 4) {
        return static_cast<float>(scales[index] & 0x3f);
    }
    return static_cast<float>(
        (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4));
}

__device__ __forceinline__ float k_min(const uint8_t *scales, int index) {
    if (index < 4) {
        return static_cast<float>(scales[index + 4] & 0x3f);
    }
    return static_cast<float>(
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

__device__ __forceinline__ float q4_k_value(const uint8_t *block, int index) {
    const int chunk = index / 64;
    const int local = index % 64;
    const int nibble = local / 32;
    const int qbyte = local % 32;
    const uint8_t packed = block[16 + chunk * 32 + qbyte];
    const int q = (packed >> (nibble * 4)) & 0x0f;
    const uint8_t *scales = block + 4;
    const float d = load_f16(block);
    const float dmin = load_f16(block + 2);
    const int scale_index = chunk * 2 + nibble;
    return d * k_scale(scales, scale_index) * static_cast<float>(q)
        - dmin * k_min(scales, scale_index);
}

__device__ __forceinline__ float q5_k_value(const uint8_t *block, int index) {
    const int chunk = index / 64;
    const int local = index % 64;
    const int nibble = local / 32;
    const int qbyte = local % 32;
    const uint8_t low = block[48 + chunk * 32 + qbyte];
    const uint8_t high = block[16 + qbyte];
    const int high_bit = (high >> (chunk * 2 + nibble)) & 1;
    const int q = ((low >> (nibble * 4)) & 0x0f) | (high_bit << 4);
    const uint8_t *scales = block + 4;
    const float d = load_f16(block);
    const float dmin = load_f16(block + 2);
    const int scale_index = chunk * 2 + nibble;
    return d * k_scale(scales, scale_index) * static_cast<float>(q)
        - dmin * k_min(scales, scale_index);
}

__device__ __forceinline__ float q6_k_value_impl(
    const uint8_t *block, int index, float d) {
    const int half = index / 128;
    const int local = index % 128;
    const int group = local / 32;
    const int j = local % 32;
    const int low_offset = half * 64 + (group & 1) * 32 + j;
    const int high_offset = half * 32 + j;
    const int low_shift = group >= 2 ? 4 : 0;
    const int high_shift = group * 2;
    const int q = ((block[low_offset] >> low_shift) & 0x0f)
        | (((block[128 + high_offset] >> high_shift) & 0x03) << 4);
    const int scale_index = half * 8 + group * 2 + j / 16;
    const int8_t scale = reinterpret_cast<const int8_t *>(block + 192)[scale_index];
    return d * static_cast<float>(scale)
        * static_cast<float>(q - 32);
}

__device__ __forceinline__ float q6_k_value(const uint8_t *block, int index) {
    return q6_k_value_impl(block, index, load_f16(block + 208));
}

__device__ __forceinline__ float q6_k_split_value(
    const uint8_t *block, int index, const uint8_t *d_ptr) {
    return q6_k_value_impl(block, index, load_f16(d_ptr));
}

__device__ __forceinline__ float q8_0_value_impl(
    const uint8_t *q_data, int index, float d) {
    const int8_t q = reinterpret_cast<const int8_t *>(q_data)[index];
    return d * static_cast<float>(q);
}

__device__ __forceinline__ float q8_0_value(const uint8_t *block, int index) {
    return q8_0_value_impl(block + 2, index, load_f16(block));
}

__device__ __forceinline__ float q8_0_split_value(
    const uint8_t *block, int index, const uint8_t *d_ptr) {
    return q8_0_value_impl(block, index, load_f16(d_ptr));
}

__host__ __device__ __forceinline__ int block_bytes(int type);

__device__ __forceinline__ uint32_t load_u32(const uint8_t *ptr) {
    // GGML K-quant payloads are at least 2-byte aligned, even when a block
    // starts at the 210-byte Q6_K stride and is not 4-byte aligned.  Two
    // halfword loads preserve that unaligned-safe contract while matching
    // llama.cpp/SGLang's get_int_from_uint8 implementation; four independent
    // byte loads needlessly expand every DP4A input into scalar instructions.
    const auto *halfwords = reinterpret_cast<const uint16_t *>(ptr);
    return static_cast<uint32_t>(halfwords[0])
        | (static_cast<uint32_t>(halfwords[1]) << 16);
}

__device__ __forceinline__ int load_i32_aligned(const int8_t *ptr) {
    // Q8_1Block::qs starts at byte 4 and every block is 36 bytes, so this
    // path is 4-byte aligned even though the source GGML weights are not.
    return *reinterpret_cast<const int *>(ptr);
}

__device__ __forceinline__ uint32_t load_u32_aligned(const uint8_t *ptr) {
    return *reinterpret_cast<const uint32_t *>(ptr);
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t *ptr) {
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

__device__ __forceinline__ float q8_1_scale(const Q8_1Block *block) {
    return __low2float(block->ds);
}

__device__ __forceinline__ float q4_k_q8_1_dot(
    const uint8_t *block, const Q8_1Block *activations, int iqs) {
    int values[2];
    int quantized[4];
    float activation_scales[2];
    const int activation_offset = 2 * ((iqs / 2) / 4);
    const uint8_t *q4 = block + 16 + 16 * activation_offset + 4 * ((iqs / 2) % 4);
    values[0] = static_cast<int>(load_u32(q4));
    values[1] = static_cast<int>(load_u32(q4 + 16));

    const uint8_t *scales = block + 4;
    const int scale_group = activation_offset / 2;
    uint16_t aux[2];
    if (scale_group < 2) {
        aux[0] = load_u16(scales + 2 * (scale_group + 0)) & 0x3f3f;
        aux[1] = load_u16(scales + 2 * (scale_group + 2)) & 0x3f3f;
    } else {
        aux[0] = ((load_u16(scales + 2 * (scale_group + 2)) >> 0) & 0x0f0f)
            | ((load_u16(scales + 2 * (scale_group - 2)) & 0xc0c0) >> 2);
        aux[1] = ((load_u16(scales + 2 * (scale_group + 2)) >> 4) & 0x0f0f)
            | ((load_u16(scales + 2 * scale_group) & 0xc0c0) >> 2);
    }
    const uint8_t *scale_bytes = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t *mins = scale_bytes + 2;
    float dot = 0.0f;
    float minimum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q8_1Block *activation = activations + activation_offset + i;
        const int *q8 = reinterpret_cast<const int *>(activation->qs)
            + ((iqs / 2) % 4);
        quantized[2 * i + 0] = q8[0];
        quantized[2 * i + 1] = q8[4];
        activation_scales[i] = q8_1_scale(activation);
        dot += activation_scales[i]
            * (__dp4a((values[0] >> (4 * i)) & 0x0f0f0f0f, quantized[2 * i], 0)
                + __dp4a((values[1] >> (4 * i)) & 0x0f0f0f0f, quantized[2 * i + 1], 0))
            * scale_bytes[i];
        minimum += activation_scales[i]
            * (__dp4a(0x01010101, quantized[2 * i], 0)
                + __dp4a(0x01010101, quantized[2 * i + 1], 0))
            * mins[i];
    }
    const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(block));
    return dm.x * dot - dm.y * minimum;
}

__device__ __forceinline__ float q5_k_q8_1_dot(
    const uint8_t *block, const Q8_1Block *activations, int iqs) {
    int low[2];
    int high[2];
    int quantized[4];
    float activation_scales[2];
    const int activation_offset = 2 * ((iqs / 2) / 4);
    const uint8_t *ql = block + 48 + 16 * activation_offset + 4 * ((iqs / 2) % 4);
    const uint8_t *qh = block + 16 + 4 * ((iqs / 2) % 4);
    low[0] = static_cast<int>(load_u32(ql));
    low[1] = static_cast<int>(load_u32(ql + 16));
    high[0] = static_cast<int>(load_u32(qh)) >> activation_offset;
    high[1] = static_cast<int>(load_u32(qh + 16)) >> activation_offset;

    const uint8_t *scales = block + 4;
    const int scale_group = activation_offset / 2;
    uint16_t aux[2];
    if (scale_group < 2) {
        aux[0] = load_u16(scales + 2 * (scale_group + 0)) & 0x3f3f;
        aux[1] = load_u16(scales + 2 * (scale_group + 2)) & 0x3f3f;
    } else {
        aux[0] = ((load_u16(scales + 2 * (scale_group + 2)) >> 0) & 0x0f0f)
            | ((load_u16(scales + 2 * (scale_group - 2)) & 0xc0c0) >> 2);
        aux[1] = ((load_u16(scales + 2 * (scale_group + 2)) >> 4) & 0x0f0f)
            | ((load_u16(scales + 2 * scale_group) & 0xc0c0) >> 2);
    }
    const uint8_t *scale_bytes = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t *mins = scale_bytes + 2;
    float dot = 0.0f;
    float minimum = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q8_1Block *activation = activations + activation_offset + i;
        const int *q8 = reinterpret_cast<const int *>(activation->qs)
            + ((iqs / 2) % 4);
        quantized[2 * i + 0] = q8[0];
        quantized[2 * i + 1] = q8[4];
        activation_scales[i] = q8_1_scale(activation);
        const int value0 = (low[0] >> (4 * i)) & 0x0f0f0f0f;
        const int value1 = (low[1] >> (4 * i)) & 0x0f0f0f0f;
        const int high0 = ((high[0] >> i) << 4) & 0x10101010;
        const int high1 = ((high[1] >> i) << 4) & 0x10101010;
        dot += activation_scales[i]
            * (__dp4a(value0 | high0, quantized[2 * i], 0)
                + __dp4a(value1 | high1, quantized[2 * i + 1], 0))
            * scale_bytes[i];
        minimum += activation_scales[i]
            * (__dp4a(0x01010101, quantized[2 * i], 0)
                + __dp4a(0x01010101, quantized[2 * i + 1], 0))
            * mins[i];
    }
    const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(block));
    return dm.x * dot - dm.y * minimum;
}

template <bool aligned>
__device__ __forceinline__ float q6_k_q8_1_dot_impl(
    const uint8_t *block,
    const Q8_1Block *activations,
    int iqs,
    float d_scale) {
    const int activation_offset = 2 * 2 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int high_shift = 2 * ((iqs % 16) / 8);
    const int low = static_cast<int>(aligned
        ? load_u32_aligned(block + 4 * iqs)
        : load_u32(block + 4 * iqs));
    const int high_index = 8 * (iqs / 16) + iqs % 8;
    const int high = static_cast<int>(aligned
        ? load_u32_aligned(block + 128 + 4 * high_index)
        : load_u32(block + 128 + 4 * high_index)) >> high_shift;
    const int8_t *scales = reinterpret_cast<const int8_t *>(block + 192) + scale_offset;
    int q8[2];
    float activation_scales[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q8_1Block *activation = activations + activation_offset + 2 * i;
        q8[i] = load_i32_aligned(activation->qs + 4 * (iqs % 8));
        activation_scales[i] = q8_1_scale(activation);
    }
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int values_low = (low >> (4 * i)) & 0x0f0f0f0f;
        const int values_high = ((high >> (4 * i)) << 4) & 0x30303030;
        const int values = __vsubss4(values_low | values_high, 0x20202020);
        dot += activation_scales[i] * __dp4a(values, q8[i], 0) * scales[4 * i];
    }
    return d_scale * dot;
}

__device__ __forceinline__ float q6_k_q8_1_dot(
    const uint8_t *block, const Q8_1Block *activations, int iqs) {
    return q6_k_q8_1_dot_impl<false>(
        block,
        activations,
        iqs,
        __half2float(*reinterpret_cast<const half *>(block + 208)));
}

__device__ __forceinline__ float q6_k_aligned_q8_1_dot(
    const uint8_t *block, const Q8_1Block *activations, int iqs) {
    return q6_k_q8_1_dot_impl<true>(
        block,
        activations,
        iqs,
        __half2float(*reinterpret_cast<const half *>(block + 208)));
}

template <bool aligned>
__device__ __forceinline__ float q6_k_split_q8_1_dot(
    const uint8_t *block,
    const uint8_t *d_ptr,
    const Q8_1Block *activations,
    int iqs) {
    return q6_k_q8_1_dot_impl<aligned>(
        block,
        activations,
        iqs,
        __half2float(*reinterpret_cast<const half *>(d_ptr)));
}

// M=8 verify reuses one quantized weight row for eight input vectors.  Keep
// the Q6 payload decode outside that input-vector loop: the old path loaded
// ql/qh/scales once per vector even though they are lane-local and invariant
// across all eight accumulators.
__device__ __forceinline__ float q6_k_q8_1_decoded_dot(
    int values0,
    int values1,
    const int8_t scale0,
    const int8_t scale1,
    float d_scale,
    const Q8_1Block *activation,
    int iqs) {
    const int activation_offset = 2 * 2 * (iqs / 16) + (iqs % 16) / 8;
    const int q8_offset = 4 * (iqs % 8);
    const Q8_1Block *activation0 = activation + activation_offset;
    const Q8_1Block *activation1 = activation0 + 2;
    const int q80 = load_i32_aligned(activation0->qs + q8_offset);
    const int q81 = load_i32_aligned(activation1->qs + q8_offset);
    float dot = q8_1_scale(activation0)
            * __dp4a(values0, q80, 0)
            * static_cast<float>(scale0);
    dot += q8_1_scale(activation1)
        * __dp4a(values1, q81, 0)
        * static_cast<float>(scale1);
    return d_scale * dot;
}

template <bool aligned>
__device__ __forceinline__ float q8_0_q8_1_dot_impl(
    const uint8_t *block,
    const Q8_1Block *activations,
    int iqs,
    float d_scale) {
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int weight = static_cast<int>(aligned
            ? load_u32_aligned(block + 4 * (iqs + i))
            : load_u32(block + 4 * (iqs + i)));
        const int activation = load_i32_aligned(activations->qs + 4 * (iqs + i));
        dot = __dp4a(weight, activation, dot);
    }
    return d_scale * q8_1_scale(activations) * dot;
}

__device__ __forceinline__ float q8_0_q8_1_dot(
    const uint8_t *block, const Q8_1Block *activations, int iqs) {
    return q8_0_q8_1_dot_impl<false>(
        block + 2, activations, iqs, load_f16(block));
}

template <bool aligned>
__device__ __forceinline__ float q8_0_split_q8_1_dot(
    const uint8_t *block,
    const uint8_t *d_ptr,
    const Q8_1Block *activations,
    int iqs) {
    return q8_0_q8_1_dot_impl<aligned>(
        block, activations, iqs, load_f16(d_ptr));
}

// As with Q6, the split Q8_0 payload and scale are invariant across the
// eight M=8 input rows.  Decode them once per lane/block and only load the
// two activation words needed by each input row below.
__device__ __forceinline__ float q8_0_q8_1_decoded_dot(
    int weight0,
    int weight1,
    float d_scale,
    const Q8_1Block *activation,
    int iqs) {
    const int activation0 = load_i32_aligned(activation->qs + 4 * iqs);
    const int activation1 = load_i32_aligned(activation->qs + 4 * (iqs + 1));
    const int dot = __dp4a(weight0, activation0, 0)
        + __dp4a(weight1, activation1, 0);
    return d_scale * q8_1_scale(activation) * dot;
}

template <typename T>
__global__ void quantize_q8_1_kernel(
    const T *x, Q8_1Block *out, int k, int padded_k) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= padded_k) {
        return;
    }
    const int row = blockIdx.y;
    const int index = row * padded_k + ix;
    const int block = index / 32;
    const int within = index % 32;
    const float value = ix < k
        ? load_activation(x + static_cast<int64_t>(row) * k + ix)
        : 0.0f;
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    const float scale = amax == 0.0f ? 0.0f : amax / 127.0f;
    out[block].qs[within] = static_cast<int8_t>(scale == 0.0f ? 0 : roundf(value / scale));
    if (within == 0) {
        out[block].ds = __halves2half2(__float2half(scale), __float2half(sum));
    }
}

template <bool aligned>
__device__ __forceinline__ float gguf_gemv_q6_split_lane_accum(
    const Q8_1Block *activations,
    const uint8_t *row,
    int k,
    int row_bytes,
    int lane) {
    const int blocks_per_row = k / 256;
    const uint8_t *d_values = row + blocks_per_row * block_bytes(kQ6KSplit);
    (void)row_bytes;
    float accum = 0.0f;
#pragma unroll 4
    for (int block = 0; block < blocks_per_row; ++block) {
        const uint8_t *weight = row + block * block_bytes(kQ6KSplit);
        const Q8_1Block *activation = activations + block * 8;
        accum += q6_k_split_q8_1_dot<aligned>(
            weight, d_values + block * 2, activation, lane);
    }
    return accum;
}

template <bool aligned>
__device__ __forceinline__ float gguf_gemv_q8_split_lane_accum(
    const Q8_1Block *activations,
    const uint8_t *row,
    int k,
    int row_bytes,
    int lane) {
    const int blocks_per_row = k / 32;
    const uint8_t *d_values = row + blocks_per_row * block_bytes(kQ8_0Split);
    (void)row_bytes;
    const int lane_group = 4;
    const int blocks_per_warp = 8;
    const int iqs = 2 * (lane % lane_group);
    float accum = 0.0f;
#pragma unroll 4
    for (int block = lane / lane_group;
         block < blocks_per_row;
         block += blocks_per_warp) {
        const uint8_t *weight = row + block * block_bytes(kQ8_0Split);
        const Q8_1Block *activation = activations + block;
        accum += q8_0_split_q8_1_dot<aligned>(
            weight, d_values + block * 2, activation, iqs);
    }
    return accum;
}

template <int type>
__device__ __forceinline__ float gguf_gemv_q8_lane_accum(
    const Q8_1Block *activations,
    const uint8_t *row,
    int k,
    int row_bytes,
    int lane) {
    (void)row_bytes;
    constexpr int elements = type == kQ8_0 || type == kQ8_0Split ? 32 : 256;
    constexpr int quantization_index_count = type == kQ8_0 || type == kQ8_0Split ? 8 : 32;
    constexpr int values_per_thread =
        (type == kQ6K || type == kQ6KAligned || type == kQ6KSplit) ? 1 : 2;
    constexpr int blocks_per_warp = values_per_thread * 32 / quantization_index_count;
    const int blocks_per_row = k / elements;
    const int lane_group = quantization_index_count / values_per_thread;
    if constexpr (type == kQ6KSplit) {
        return (reinterpret_cast<uintptr_t>(row) & 3u) == 0u
            ? gguf_gemv_q6_split_lane_accum<true>(activations, row, k, row_bytes, lane)
            : gguf_gemv_q6_split_lane_accum<false>(activations, row, k, row_bytes, lane);
    }
    if constexpr (type == kQ8_0Split) {
        return (reinterpret_cast<uintptr_t>(row) & 3u) == 0u
            ? gguf_gemv_q8_split_lane_accum<true>(activations, row, k, row_bytes, lane)
            : gguf_gemv_q8_split_lane_accum<false>(activations, row, k, row_bytes, lane);
    }
    float accum = 0.0f;
    for (int block = lane / lane_group; block < blocks_per_row; block += blocks_per_warp) {
        const int iqs = values_per_thread * (lane % lane_group);
        const uint8_t *weight = row + block * block_bytes(type);
        const Q8_1Block *activation = activations + block * (elements / 32);
        if constexpr (type == kQ4K) {
            accum += q4_k_q8_1_dot(weight, activation, iqs);
        } else if constexpr (type == kQ5K) {
            accum += q5_k_q8_1_dot(weight, activation, iqs);
        } else if constexpr (type == kQ6K) {
            accum += q6_k_q8_1_dot(weight, activation, iqs);
        } else if constexpr (type == kQ6KAligned) {
            accum += q6_k_aligned_q8_1_dot(weight, activation, iqs);
        } else {
            accum += q8_0_q8_1_dot(weight, activation, iqs);
        }
    }
    return accum;
}

template <int type, typename output_t = __nv_bfloat16>
__global__ void gguf_gemv_q8_kernel(
    output_t *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    const int output_row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_q8_lane_accum<type>(activations, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

struct GgufMixedProjection {
    uint64_t packed_ptr;
    int32_t output_offset;
    int32_t output_rows;
    int32_t row_bytes;
    int32_t type;
    int32_t reserved0;
    int32_t reserved1;
};

static_assert(sizeof(GgufMixedProjection) == 32, "GGUF mixed descriptor ABI must stay packed");

template <typename output_t = __nv_bfloat16>
__global__ void gguf_gemv_q8_mixed_kernel(
    output_t *out,
    const Q8_1Block *activations,
    const GgufMixedProjection *projections,
    int projection_count,
    int total_n,
    int k) {
    const int output_row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (output_row >= total_n) {
        return;
    }
    int projection_index = 0;
    for (; projection_index + 1 < projection_count; ++projection_index) {
        if (output_row < projections[projection_index + 1].output_offset) {
            break;
        }
    }
    const GgufMixedProjection projection = projections[projection_index];
    const int local_row = output_row - projection.output_offset;
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(projection.packed_ptr);
    const uint8_t *row = packed + static_cast<int64_t>(local_row) * projection.row_bytes;
    float accum = 0.0f;
    switch (projection.type) {
        case kQ4K:
            accum = gguf_gemv_q8_lane_accum<kQ4K>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ5K:
            accum = gguf_gemv_q8_lane_accum<kQ5K>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ6K:
            accum = gguf_gemv_q8_lane_accum<kQ6K>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ6KAligned:
            accum = gguf_gemv_q8_lane_accum<kQ6KAligned>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ6KSplit:
            accum = gguf_gemv_q8_lane_accum<kQ6KSplit>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0:
            accum = gguf_gemv_q8_lane_accum<kQ8_0>(
                activations, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0Split:
            accum = gguf_gemv_q8_lane_accum<kQ8_0Split>(
                activations, row, k, projection.row_bytes, lane);
            break;
        default:
            return;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

// Decode-side Q8_1 activation staging for M=1.  GGML's MMVQ path assigns one
// warp to each output row, which makes every row stream the same activation
// vector from global memory.  Q6_K/Q5_K rows are wide enough that L1 does not
// reliably retain that vector while the weight rows advance.  Eight output
// warps share one CTA-local Q8_1 copy here; the packed weight traversal and
// DP4A arithmetic stay identical to the ordinary kernel.
template <int type, typename output_t = __nv_bfloat16>
__global__ void gguf_gemv_q8_cached_kernel(
    output_t *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    extern __shared__ unsigned char q8_activation_storage[];
    auto *activation_cache = reinterpret_cast<Q8_1Block *>(q8_activation_storage);
    const int padded_k = ((k + 511) / 512) * 512;
    const int activation_blocks = padded_k / 32;
    for (int index = threadIdx.x; index < activation_blocks; index += blockDim.x) {
        activation_cache[index] = activations[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_q8_lane_accum<type>(
        activation_cache, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename output_t = __nv_bfloat16>
__global__ void gguf_gemv_q8_mixed_cached_kernel(
    output_t *out,
    const Q8_1Block *activations,
    const GgufMixedProjection *projections,
    int projection_count,
    int total_n,
    int k) {
    extern __shared__ unsigned char q8_activation_storage[];
    auto *activation_cache = reinterpret_cast<Q8_1Block *>(q8_activation_storage);
    const int padded_k = ((k + 511) / 512) * 512;
    const int activation_blocks = padded_k / 32;
    for (int index = threadIdx.x; index < activation_blocks; index += blockDim.x) {
        activation_cache[index] = activations[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    if (output_row >= total_n) {
        return;
    }
    int projection_index = 0;
    for (; projection_index + 1 < projection_count; ++projection_index) {
        if (output_row < projections[projection_index + 1].output_offset) {
            break;
        }
    }
    const GgufMixedProjection projection = projections[projection_index];
    const int local_row = output_row - projection.output_offset;
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(projection.packed_ptr);
    const uint8_t *row = packed + static_cast<int64_t>(local_row) * projection.row_bytes;
    float accum = 0.0f;
    switch (projection.type) {
        case kQ4K:
            accum = gguf_gemv_q8_lane_accum<kQ4K>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ5K:
            accum = gguf_gemv_q8_lane_accum<kQ5K>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6K:
            accum = gguf_gemv_q8_lane_accum<kQ6K>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6KAligned:
            accum = gguf_gemv_q8_lane_accum<kQ6KAligned>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6KSplit:
            accum = gguf_gemv_q8_lane_accum<kQ6KSplit>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0:
            accum = gguf_gemv_q8_lane_accum<kQ8_0>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0Split:
            accum = gguf_gemv_q8_lane_accum<kQ8_0Split>(
                activation_cache, row, k, projection.row_bytes, lane);
            break;
        default:
            return;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

// SGLang's Q5_K MMQ specialization.  Qwen3.8's UD checkpoint is mixed K-
// quantized: several wide gate/up projections are Q5_K rather than Q6_K.
// Keeping the Q5 decode in the same four-warp tile as Q6 lets those layers
// reuse the eight Q8_1 activation blocks across four verify vectors instead
// of falling back to the BF16 decode tile.
constexpr int kQ5MmqX = 8;
constexpr int kQ5MmqY = 32;
constexpr int kQ5MmqWarps = 8;

__device__ __forceinline__ float q5_k_q8_1_mmq_dot(
    const int *x_q,
    const float *x_d,
    const float *x_dmin,
    const float *x_scales,
    const float *x_mins,
    const int *y_q,
    const float *y_scales,
    int row,
    int vector) {
    const int *weight = x_q + row * 64;
    const int *activation = y_q + vector * 64;
    float sum = 0.0f;

#pragma unroll
    for (int block = 0; block < 8; ++block) {
        int dot = 0;
        int minimum = 0;
#pragma unroll
        for (int word = 0; word < 8; ++word) {
            const int q8 = activation[block * 8 + word];
            dot = __dp4a(weight[block * 8 + word], q8, dot);
            minimum = __dp4a(0x01010101, q8, minimum);
        }
        const float activation_scale = y_scales[vector * 8 + block];
        sum += activation_scale * (
            x_d[row] * x_scales[row * 8 + block] * static_cast<float>(dot)
            - x_dmin[row] * x_mins[row * 8 + block] * static_cast<float>(minimum));
    }
    return sum;
}

__global__ void gguf_gemm_q8_q5_mmq_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row_base = blockIdx.x * kQ5MmqY;
    const int vec_base = blockIdx.y * kQ5MmqX;
    const int output_row = row_base + lane;
    const int blocks_per_row = k / 256;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;

    // One Q5_K block has eight Q8_1 activation blocks.  The q values are
    // stored as eight packed int32 words per activation block; scales/mins
    // are decoded once per output row and reused by all four vectors.
    __shared__ int tile_x_q[kQ5MmqY * 64];
    __shared__ float tile_x_d[kQ5MmqY];
    __shared__ float tile_x_dmin[kQ5MmqY];
    __shared__ float tile_x_scales[kQ5MmqY * 8];
    __shared__ float tile_x_mins[kQ5MmqY * 8];
    __shared__ int tile_y_q[kQ5MmqX * 64];
    __shared__ float tile_y_scales[kQ5MmqX * 8];

    float accum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        // Four warps cover the 32 output rows.  Each lane expands one aligned
        // Q5 word; the SGLang layout places the low/high nibble words into
        // four 16-word groups so the dot loop can consume all 256 values as
        // 64 DP4A words.
        for (int i0 = 0; i0 < kQ5MmqY; i0 += kQ5MmqWarps) {
            const int row_slot = i0 + warp;
            const int source_row = min(row_base + row_slot, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *block = row + static_cast<int64_t>(block_index) * 176;
            const uint32_t low = load_u32(block + 48 + 4 * lane);
            const uint32_t high = load_u32(block + 16 + 4 * (lane % 8));
            const int high_shift = 2 * (lane / 8);
            const uint32_t q0 = (low & 0x0f0f0f0f)
                | (((high >> high_shift) << 4) & 0x10101010);
            const uint32_t q1 = ((low >> 4) & 0x0f0f0f0f)
                | (((high >> (high_shift + 1)) << 4) & 0x10101010);
            const int target_base = (lane / 8) * 16 + (lane % 8);
            int *target = tile_x_q + row_slot * 64 + target_base;
            target[0] = static_cast<int>(q0);
            target[8] = static_cast<int>(q1);
        }

        // One lane per output row owns the two super-block scales.  Every
        // warp contributes two scale/min entries for all rows, avoiding a
        // second per-row decode in the dot loop.
        if (warp == 0) {
            const int source_row = min(row_base + lane, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *block = row + static_cast<int64_t>(block_index) * 176;
            tile_x_d[lane] = load_f16(block);
            tile_x_dmin[lane] = load_f16(block + 2);
        }
        const int scale_row = lane;
        const int source_row = min(row_base + scale_row, n - 1);
        const uint8_t *scale_row_ptr = packed
            + static_cast<int64_t>(source_row) * row_bytes
            + static_cast<int64_t>(block_index) * 176 + 4;
        if (warp < kQ5MmqY / 8) {
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int scale_index = warp * 2 + part;
                tile_x_scales[scale_row * 8 + scale_index] = k_scale(
                    scale_row_ptr,
                    scale_index);
                tile_x_mins[scale_row * 8 + scale_index] = k_min(
                    scale_row_ptr,
                    scale_index);
            }
        }
        __syncthreads();

        const int vector = min(vec_base + warp, m - 1);
        const Q8_1Block *vector_blocks = activations
            + static_cast<int64_t>(vector) * activation_row_blocks
            + block_index * 8;
        if (lane < 8) {
#pragma unroll
            for (int activation_block = 0; activation_block < 8; ++activation_block) {
                tile_y_q[warp * 64 + activation_block * 8 + lane]
                    = load_i32_aligned(vector_blocks[activation_block].qs + 4 * lane);
                tile_y_scales[warp * 8 + activation_block]
                    = q8_1_scale(vector_blocks + activation_block);
            }
        }
        __syncthreads();

        if (output_row < n && vec_base + warp < m) {
            accum += q5_k_q8_1_mmq_dot(
                tile_x_q,
                tile_x_d,
                tile_x_dmin,
                tile_x_scales,
                tile_x_mins,
                tile_y_q,
                tile_y_scales,
                lane,
                warp);
        }
        __syncthreads();
    }

    if (output_row < n && vec_base + warp < m) {
        out[static_cast<int64_t>(vec_base + warp) * n + output_row]
            = __float2bfloat16_rn(accum);
    }
}

// SGLang/llama.cpp-style MMQ tile for the private Q6_K_SPLIT layout.
//
// The ordinary large-M path below decodes every weight tile to BF16 and feeds
// it to tensor cores.  That is a good general fallback, but it throws away
// the reuse available when M is only a handful of rows (DFlash2 verify).  MMQ
// keeps one 32-row Q6 tile and one 4-vector Q8_1 tile in shared memory, then
// performs the same Q6_K x Q8_1 DP4A arithmetic as the GGML reference kernels.
// It is intentionally a separate ABI path: Q8_1 activation quantization is
// approximate and must not silently replace the BF16 tensor-core contract.
// SGLang/llama.cpp use MMQ_X_Q6_K=4, MMQ_Y_Q6_K=32, NWARPS_Q6_K=4.  The
// SM120 endpoint A/B showed that literal geometry is slower for this private
// row-tail layout: eight vectors/eight warps improves the measured verify
// schedule, so keep the reference layout/DP4A arithmetic but tune the CTA
// shape for this architecture and storage contract.
constexpr int kQ6MmqX = 8;
constexpr int kQ6MmqY = 32;
constexpr int kQ6MmqWarps = 8;

__device__ __forceinline__ float q6_k_q8_1_mmq_dot(
    const int *x_ql,
    const float *x_dm,
    const int *x_sc,
    const int *y_qs,
    const float *y_ds,
    int i,
    int j,
    int k,
    int y_stride,
    int y_base) {
    const int8_t *sc = reinterpret_cast<const int8_t *>(
        &x_sc[i * 4 + i / 8 + k / 8]);
    const int index_x = i * (2 * 32 + 1) + 2 * k;
    const int index_y = j * y_stride + y_base + (2 * k) % 32;
    const int *v = &x_ql[index_x];
    const int *u = &y_qs[index_y];
    const float *d8 = &y_ds[index_y / 8];
    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < 8; i0 += 4) {
        int2 sumi_d = {0, 0};

#pragma unroll
        for (int ii = i0; ii < i0 + 2; ++ii) {
            sumi_d.x = __dp4a(v[2 * ii + 0], u[2 * ii + 0], sumi_d.x);
            sumi_d.x = __dp4a(v[2 * ii + 1], u[2 * ii + 1], sumi_d.x);
            sumi_d.y = __dp4a(v[2 * ii + 4], u[2 * ii + 4], sumi_d.y);
            sumi_d.y = __dp4a(v[2 * ii + 5], u[2 * ii + 5], sumi_d.y);
        }
        sumf_d += d8[i0 / 4]
            * (static_cast<float>(sc[i0 / 2 + 0]) * sumi_d.x
                + static_cast<float>(sc[i0 / 2 + 1]) * sumi_d.y);
    }
    return x_dm[i] * sumf_d;
}

template <bool NeedCheck>
__global__ void gguf_gemm_q8_q6_mmq_split_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    // This follows GGML/SGLang's Q6_K MMQ ownership model: each warp owns one
    // input vector and its 32 lanes own the output rows.  SM120 uses the local
    // eight-warp retune declared above for this private split-row layout.
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row_base = blockIdx.x * kQ6MmqY;
    const int vec_base = blockIdx.y * kQ6MmqX;
    const int blocks_per_row = k / 256;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;

    // The transformed Q6 tile is the same layout consumed by the upstream
    // MMQ dot routine: two packed ql integers per 16-value half, one d per
    // row, and four packed scale integers per row.  Keeping this compact
    // representation avoids re-reading ql/qh/scales for each of the four
    // input vectors.
    __shared__ int tile_x_ql[kQ6MmqY * (2 * 32 + 1)];
    __shared__ float tile_x_dm[kQ6MmqY + 1];
    __shared__ int tile_x_sc[kQ6MmqY * 4 + 4];
    // Load both Q8_1 halves of a 256-value Q6 block before the first
    // barrier.  The original translation mirrored the reference loop and
    // loaded/synchronized once per half; the two halves are independent in
    // shared memory, so keeping them side by side removes one barrier and
    // one repeated setup phase per weight block.
    __shared__ int tile_y_qs[kQ6MmqX * 64];
    __shared__ float tile_y_ds[kQ6MmqX * 8];

    float accum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        // Load one 208-byte Q6 payload per output row.  Each lane handles one
        // packed 32-bit ql/qh word for every row assigned to its warp.
        for (int i0 = 0; i0 < kQ6MmqY; i0 += kQ6MmqWarps) {
            const int row_slot = i0 + warp;
            const int source_row = NeedCheck
                ? min(row_base + row_slot, n - 1)
                : row_base + row_slot;
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *block = row + static_cast<int64_t>(block_index) * 208;
            const int ql = static_cast<int>(load_u32(block + 4 * lane));
            const int ql0 = (ql >> 0) & 0x0f0f0f0f;
            const int ql1 = (ql >> 4) & 0x0f0f0f0f;
            const int high_index = 8 * (lane / 16) + lane % 8;
            const int high = static_cast<int>(load_u32(block + 128 + 4 * high_index));
            const int high_shift = 2 * ((lane % 16) / 8);
            const int qh0 = ((high >> high_shift) << 4) & 0x30303030;
            const int qh1 = (high >> high_shift) & 0x30303030;
            const int kq0 = (lane < 16 ? lane : 32 + lane - 16);
            const int kq1 = kq0 + 16;
            int *x_ql = &tile_x_ql[row_slot * (2 * 32 + 1)];
            x_ql[kq0] = __vsubss4(ql0 | qh0, 0x20202020);
            x_ql[kq1] = __vsubss4(ql1 | qh1, 0x20202020);
        }

        // One lane per output row loads the row-tail FP16 d value.  The
        // split representation stores all d values after the 208-byte data
        // blocks, with no extra residency.
        if (warp == 0) {
            const int source_row = NeedCheck
                ? min(row_base + lane, n - 1)
                : row_base + lane;
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *d_ptr = row + static_cast<int64_t>(blocks_per_row) * 208
                + static_cast<int64_t>(block_index) * 2;
            tile_x_dm[lane] = load_f16(d_ptr);
        }

        // Four lanes per output row pack the sixteen int8 scales into the
        // conflict-avoiding MMQ layout.
        if (warp < kQ6MmqY / 8) {
            const int scale_row = warp * 8 + lane / 4;
            const int scale_source_row = NeedCheck
                ? min(row_base + scale_row, n - 1)
                : row_base + scale_row;
            const uint8_t *scale_row_ptr = packed
                + static_cast<int64_t>(scale_source_row) * row_bytes
                + static_cast<int64_t>(block_index) * 208 + 192;
            const int scale_part = lane % 4;
            tile_x_sc[scale_row * 4 + scale_row / 8 + scale_part]
                = static_cast<int>(load_u32(scale_row_ptr + 4 * scale_part));
        }
        __syncthreads();

        const int vector = NeedCheck ? min(vec_base + warp, m - 1) : vec_base + warp;
        const Q8_1Block *vector_blocks = activations
            + static_cast<int64_t>(vector) * activation_row_blocks
            + block_index * 8;

        for (int ir = 0; ir < 2; ++ir) {
            const int kqs = ir * 32 + lane;
            const int activation_block = kqs / 8;
            tile_y_qs[warp * 64 + ir * 32 + lane] = load_i32_aligned(
                vector_blocks[activation_block].qs + 4 * (lane % 8));

            if (warp == 0 && lane < kQ6MmqX * 4) {
                const int vector_index = lane / 4;
                const int vector_for_scale = min(vec_base + vector_index, m - 1);
                const int scale_index = lane % 4;
                const Q8_1Block *scale_block = activations
                    + static_cast<int64_t>(vector_for_scale) * activation_row_blocks
                    + block_index * 8 + ir * 4 + scale_index;
                tile_y_ds[vector_index * 8 + ir * 4 + scale_index]
                    = q8_1_scale(scale_block);
            }
        }
        __syncthreads();

        // Each thread owns one (output row, input vector) pair and four
        // eight-value dot slices.  Both Q8 halves are resident now, so the
        // block needs only one load barrier and one reuse barrier.
#pragma unroll
        for (int ir = 0; ir < 2; ++ir) {
#pragma unroll
            for (int dot_k = ir * 16; dot_k < (ir + 1) * 16; dot_k += 8) {
                if constexpr (!NeedCheck) {
                    accum += q6_k_q8_1_mmq_dot(
                        tile_x_ql,
                        tile_x_dm,
                        tile_x_sc,
                        tile_y_qs,
                        tile_y_ds,
                        lane,
                        warp,
                        dot_k,
                        64,
                        ir * 32);
                } else if (row_base + lane < n && vec_base + warp < m) {
                    accum += q6_k_q8_1_mmq_dot(
                        tile_x_ql,
                        tile_x_dm,
                        tile_x_sc,
                        tile_y_qs,
                        tile_y_ds,
                        lane,
                        warp,
                        dot_k,
                        64,
                        ir * 32);
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!NeedCheck) {
        out[static_cast<int64_t>(vec_base + warp) * n + row_base + lane]
            = __float2bfloat16_rn(accum);
    } else if (row_base + lane < n && vec_base + warp < m) {
        out[static_cast<int64_t>(vec_base + warp) * n + row_base + lane]
            = __float2bfloat16_rn(accum);
    }
}

// This is the CUDA MMQ layout used by llama.cpp/ds4 for the Q6_K tensor-core
// path.  The existing runtime MMQ kernel above intentionally stays on the
// simpler DP4A implementation.  This candidate keeps the same Q8_1 contract,
// but feeds centered Q6 bytes and Q8 bytes to m16n8k16 integer MMA.  It is
// selected only by QSR_GGUF_MMQ_MMA=1 while the candidate is being validated.
constexpr int kQ6MmqMmaX = 8;
constexpr int kQ6MmqMmaY = 128;
constexpr int kQ6MmqMmaWarps = 8;
// The upstream 71-int row packs metadata into the tail, but that makes the
// ldmatrix row addresses 4-byte skewed.  SM120 reports a misaligned-address
// fault for that form, so keep the Q6 payload on a 16-byte aligned 72-int
// stride and store d/scales in separate aligned arrays.
constexpr int kQ6MmqMmaRowStride = 72;

struct Q6MmaTileA {
    int x[2];
};

struct Q6MmaTileB {
    int x[1];
};

struct Q6MmaTileC {
    int x[4];
};

__device__ __forceinline__ void q6_mma_load_a(
    Q6MmaTileA &tile,
    const int *shared,
    int stride) {
    const int *source = shared + (threadIdx.x % 16) * stride;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(tile.x[0]), "=r"(tile.x[1])
        : "l"(source));
#else
    tile.x[0] = 0;
    tile.x[1] = 0;
#endif
}

__device__ __forceinline__ void q6_mma_load_b(
    Q6MmaTileB &tile,
    const int *shared,
    int stride) {
    tile.x[0] = shared[(threadIdx.x / 4) * stride + threadIdx.x % 4];
}

__device__ __forceinline__ void q6_mma_dot(
    Q6MmaTileC &tile,
    const Q6MmaTileA &a,
    const Q6MmaTileB &b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(tile.x[0]), "+r"(tile.x[1]), "+r"(tile.x[2]), "+r"(tile.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(b.x[0]));
#else
    tile.x[0] = 0;
    tile.x[1] = 0;
    tile.x[2] = 0;
    tile.x[3] = 0;
#endif
}

__device__ __forceinline__ int q6_mma_row(int fragment) {
    return (fragment / 2) * 8 + threadIdx.x / 4;
}

__device__ __forceinline__ int q6_mma_vector(int fragment) {
    return (threadIdx.x % 4) * 2 + fragment % 2;
}

__global__ void gguf_gemm_q8_q6_mmq_split_mma_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row_base = blockIdx.x * kQ6MmqMmaY;
    const int vec_base = blockIdx.y * kQ6MmqMmaX;
    const int blocks_per_row = k / 256;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;

    __shared__ __align__(16) int tile_x_qs[kQ6MmqMmaY * kQ6MmqMmaRowStride];
    __shared__ __align__(16) float tile_x_dm[kQ6MmqMmaY];
    __shared__ __align__(16) int tile_x_sc[kQ6MmqMmaY * 4];
    __shared__ __align__(16) int tile_y_qs[kQ6MmqMmaX * 64];
    __shared__ __align__(16) float tile_y_ds[kQ6MmqMmaX * 8];

    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        // Pack one Q6_K block per weight row into the exact row-major shared
        // layout expected by ldmatrix.  The source is the runtime's split
        // 208-byte payload plus the row-tail FP16 d array.
        for (int row_slot = warp; row_slot < kQ6MmqMmaY; row_slot += kQ6MmqMmaWarps) {
            const int source_row = min(row_base + row_slot, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *block = row + static_cast<int64_t>(block_index) * 208;
            const int ql = static_cast<int>(load_u32(block + 4 * lane));
            const int ql0 = (ql >> 0) & 0x0f0f0f0f;
            const int ql1 = (ql >> 4) & 0x0f0f0f0f;
            const int high_index = 8 * (lane / 16) + lane % 8;
            const int high = static_cast<int>(load_u32(block + 128 + 4 * high_index));
            const int high_shift = 2 * ((lane % 16) / 8);
            const int qh0 = ((high >> high_shift) << 4) & 0x30303030;
            const int qh1 = (high >> high_shift) & 0x30303030;
            const int kq0 = lane < 16 ? lane : 32 + lane - 16;
            const int kq1 = kq0 + 16;
            int *row_qs = tile_x_qs + row_slot * kQ6MmqMmaRowStride;
            row_qs[kq0] = __vsubss4(ql0 | qh0, 0x20202020);
            row_qs[kq1] = __vsubss4(ql1 | qh1, 0x20202020);
            // ldmatrix addresses a full 16-byte-aligned row.  The Q6
            // payload itself is only 64 ints; make the aligned padding
            // deterministic instead of letting the last row of a 128-row
            // CTA consume stale shared-memory words.
            if (lane < 8) {
                row_qs[64 + lane] = 0;
            }
        }

        // Metadata occupies the otherwise unused padded tail of every Q6
        // row, exactly as the upstream MMA tile does.
        const int linear = warp * 32 + lane;
        if (linear < kQ6MmqMmaY) {
            const int source_row = min(row_base + linear, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const uint8_t *d_ptr = row + static_cast<int64_t>(blocks_per_row) * 208
                + static_cast<int64_t>(block_index) * 2;
            tile_x_dm[linear] = load_f16(d_ptr);

            const uint8_t *scale_row = row + static_cast<int64_t>(block_index) * 208 + 192;
            int *scale_words = tile_x_sc + linear * 4;
#pragma unroll
            for (int scale_part = 0; scale_part < 4; ++scale_part) {
                scale_words[scale_part] = static_cast<int>(load_u32(scale_row + 4 * scale_part));
            }
        }
        __syncthreads();

        // Keep the two 128-value Q8_1 halves contiguous per input vector.  A
        // warp owns one vector, matching the Q6 MMQ source implementation.
        const int vector = min(vec_base + warp, m - 1);
        const Q8_1Block *vector_blocks = activations
            + static_cast<int64_t>(vector) * activation_row_blocks
            + block_index * 8;
        for (int half = 0; half < 2; ++half) {
            const int activation_block = half * 4 + lane / 8;
            tile_y_qs[warp * 64 + half * 32 + lane] = load_i32_aligned(
                vector_blocks[activation_block].qs + 4 * (lane % 8));
        }
        for (int index = lane; index < kQ6MmqMmaX * 8; index += 32) {
            const int vector_index = index / 8;
            const int vector_for_scale = min(vec_base + vector_index, m - 1);
            const int activation_block = index % 8;
            const Q8_1Block *scale_block = activations
                + static_cast<int64_t>(vector_for_scale) * activation_row_blocks
                + block_index * 8 + activation_block;
            tile_y_ds[index] = q8_1_scale(scale_block);
        }
        __syncthreads();

        // Two integer MMA operations cover 32 Q6 values for each group of
        // eight Q8 values.  The scale application follows llama.cpp's Q6_K
        // MMQ path: each 16-value MMA half has its own int8 weight scale.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int *y_qs = tile_y_qs + half * 32;
            const int y_scale_base = half * 4;
#pragma unroll
            for (int k01 = 0; k01 < 32; k01 += 8) {
                const int k0 = half * 32 + k01;
                Q6MmaTileA a[2];
                const int row_tile = warp * 16;
                q6_mma_load_a(
                    a[0], tile_x_qs + row_tile * kQ6MmqMmaRowStride + k0,
                    kQ6MmqMmaRowStride);
                q6_mma_load_a(
                    a[1], tile_x_qs + row_tile * kQ6MmqMmaRowStride + k0 + 4,
                    kQ6MmqMmaRowStride);

#pragma unroll
                for (int j0 = 0; j0 < kQ6MmqMmaX; j0 += 8) {
                    Q6MmaTileB b[2];
                    q6_mma_load_b(b[0], y_qs + j0 * 64 + k01, 64);
                    q6_mma_load_b(b[1], y_qs + j0 * 64 + k01 + 4, 64);
                    Q6MmaTileC c[2];
#pragma unroll
                    for (int c_index = 0; c_index < 2; ++c_index) {
#pragma unroll
                        for (int fragment = 0; fragment < 4; ++fragment) {
                            c[c_index].x[fragment] = 0;
                        }
                    }
                    q6_mma_dot(c[0], a[0], b[0]);
                    q6_mma_dot(c[1], a[1], b[1]);

#pragma unroll
                    for (int fragment = 0; fragment < 4; ++fragment) {
                        const int row = row_tile + q6_mma_row(fragment);
                        const int vector_local = q6_mma_vector(fragment);
                        const int scale_word_index = k0 / 16;
                        const int8_t *scales = reinterpret_cast<const int8_t *>(
                            tile_x_sc + row * 4 + scale_word_index);
                        const int scale_index = (k01 / 4) & 3;
                        const float d_weight = tile_x_dm[row];
                        const float d_activation = tile_y_ds[
                            (j0 + vector_local) * 8 + y_scale_base + k01 / 8];
                        accum[fragment] += d_weight * d_activation * (
                            static_cast<float>(c[0].x[fragment]) * scales[scale_index]
                            + static_cast<float>(c[1].x[fragment]) * scales[scale_index + 1]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int fragment = 0; fragment < 4; ++fragment) {
        const int row = row_base + warp * 16 + q6_mma_row(fragment);
        const int vector = vec_base + q6_mma_vector(fragment);
        if (row < n && vector < m) {
            out[static_cast<int64_t>(vector) * n + row]
                = __float2bfloat16_rn(accum[fragment]);
        }
    }
}

// SGLang's Q8_0 MMQ specialization.  Q8_0 has no per-subgroup scale or
// zero-point decode: four weight blocks fit in one 32-int shared tile row and
// each dot is eight DP4A instructions.  The DFlash2 M=8 specialization keeps
// the same eight-row input reuse as the Q6 tile above.
constexpr int kQ8MmqX = 8;
constexpr int kQ8MmqY = 32;
constexpr int kQ8MmqWarps = 8;

template <bool split>
__global__ void gguf_gemm_q8_q8_mmq_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row_base = blockIdx.x * kQ8MmqY;
    const int vec_base = blockIdx.y * kQ8MmqX;
    const int output_row = row_base + lane;
    const int blocks_per_row = k / 32;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;

    __shared__ int tile_x_qs[kQ8MmqY * (32 + 1)];
    __shared__ float tile_x_d[kQ8MmqY * 4];
    __shared__ int tile_y_qs[kQ8MmqX * 32];
    __shared__ float tile_y_d[kQ8MmqX * 4];

    float accum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; block_index += 4) {
        // Each warp loads four rows; each lane loads one packed int from one
        // of the four Q8_0 blocks in this K tile.
        for (int i0 = 0; i0 < kQ8MmqY; i0 += kQ8MmqWarps) {
            const int row_slot = i0 + warp;
            const int source_row = min(row_base + row_slot, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const int block = block_index + lane / 8;
            const uint8_t *weight = row + static_cast<int64_t>(block) * (split ? 32 : 34)
                + (split ? 0 : 2);
            tile_x_qs[row_slot * 33 + lane] = static_cast<int>(
                load_u32(weight + 4 * (lane % 8)));
        }

        // Four lanes per output row load the four FP16 weight scales.  The
        // split layout stores these scales in a row tail; standard GGUF keeps
        // them at the beginning of each 34-byte block.
        if (warp < kQ8MmqWarps / 2) {
            const int scale_row = warp * 8 + lane / 4;
            const int source_row = min(row_base + scale_row, n - 1);
            const uint8_t *row = packed + static_cast<int64_t>(source_row) * row_bytes;
            const int block = block_index + lane % 4;
            const uint8_t *d_ptr = split
                ? row + static_cast<int64_t>(blocks_per_row) * 32 + block * 2
                : row + static_cast<int64_t>(block) * 34;
            tile_x_d[scale_row * 4 + lane % 4] = load_f16(d_ptr);
        }
        __syncthreads();

        // Four Q8_1 activation blocks (128 values) match the four Q8_0
        // weight blocks in the shared tile.  The second half of K is handled
        // by the next block_index iteration.
        const int vector = min(vec_base + warp, m - 1);
        const Q8_1Block *vector_blocks = activations
            + static_cast<int64_t>(vector) * activation_row_blocks + block_index;
        tile_y_qs[warp * 32 + lane] = load_i32_aligned(
            vector_blocks[lane / 8].qs + 4 * (lane % 8));
        if (lane % 8 == 0) {
            tile_y_d[warp * 4 + lane / 8] = q8_1_scale(vector_blocks + lane / 8);
        }
        __syncthreads();

#pragma unroll
        for (int dot_k = 0; dot_k < 32; dot_k += 8) {
            int sum = 0;
#pragma unroll
            for (int index = 0; index < 8; ++index) {
                sum = __dp4a(
                    tile_x_qs[lane * 33 + dot_k + index],
                    tile_y_qs[warp * 32 + dot_k + index],
                    sum);
            }
            accum += tile_x_d[lane * 4 + dot_k / 8]
                * tile_y_d[warp * 4 + dot_k / 8] * static_cast<float>(sum);
        }
        __syncthreads();
    }

    if (output_row < n && vec_base + warp < m) {
        out[static_cast<int64_t>(vec_base + warp) * n + output_row]
            = __float2bfloat16_rn(accum);
    }
}

template <int type>
__global__ void gguf_gemm_q8_tile_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    const int input_row = blockIdx.y * 8;
    if (output_row >= n) {
        return;
    }
    constexpr int elements = type == kQ8_0 ? 32 : 256;
    constexpr int quantization_index_count = type == kQ8_0 ? 8 : 32;
    constexpr int values_per_thread =
        (type == kQ6K || type == kQ6KAligned) ? 1 : 2;
    constexpr int blocks_per_warp = values_per_thread * 32 / quantization_index_count;
    const int blocks_per_row = k / elements;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;
    const int lane_group = quantization_index_count / values_per_thread;
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    const uint8_t *d_values = type == kQ6KSplit
        ? row + blocks_per_row * block_bytes(kQ6KSplit)
        : nullptr;
    float accum[8] = {};

    for (int block = lane / lane_group; block < blocks_per_row; block += blocks_per_warp) {
        const int iqs = values_per_thread * (lane % lane_group);
        const uint8_t *weight = row + block * block_bytes(type);
        const Q8_1Block *activation = activations + block * (elements / 32);
#pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            if (input_row + mi >= m) {
                continue;
            }
            const Q8_1Block *row_activation = activation
                + static_cast<int64_t>(input_row + mi) * activation_row_blocks;
            if constexpr (type == kQ4K) {
                accum[mi] += q4_k_q8_1_dot(weight, row_activation, iqs);
            } else if constexpr (type == kQ5K) {
                accum[mi] += q5_k_q8_1_dot(weight, row_activation, iqs);
            } else if constexpr (type == kQ6K) {
                accum[mi] += q6_k_q8_1_dot(weight, row_activation, iqs);
            } else if constexpr (type == kQ6KAligned) {
                accum[mi] += q6_k_aligned_q8_1_dot(weight, row_activation, iqs);
            } else if constexpr (type == kQ6KSplit) {
                accum[mi] += (reinterpret_cast<uintptr_t>(row) & 3u) == 0u
                    ? q6_k_split_q8_1_dot<true>(
                        weight, d_values + block * 2, row_activation, iqs)
                    : q6_k_split_q8_1_dot<false>(
                        weight, d_values + block * 2, row_activation, iqs);
            } else {
                accum[mi] += q8_0_q8_1_dot(weight, row_activation, iqs);
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            accum[mi] += __shfl_xor_sync(0xffffffff, accum[mi], delta);
        }
        if (lane == 0 && input_row + mi < m) {
            out[static_cast<int64_t>(input_row + mi) * n + output_row]
                = __float2bfloat16_rn(accum[mi]);
        }
    }
}

template <bool aligned>
__global__ void gguf_gemm_q8_split_tile_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    const int input_row = blockIdx.y * 8;
    if (output_row >= n) {
        return;
    }
    const int blocks_per_row = k / 256;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    const uint8_t *d_values = row + blocks_per_row * 208;
    float accum[8] = {};

    for (int block = 0; block < blocks_per_row; ++block) {
        const uint8_t *weight = row + block * 208;
        const int iqs = lane;
        const int high_index = 8 * (iqs / 16) + iqs % 8;
        const int high_shift = 2 * ((iqs % 16) / 8);
        const int low = static_cast<int>(aligned
            ? load_u32_aligned(weight + 4 * iqs)
            : load_u32(weight + 4 * iqs));
        const int high = static_cast<int>(aligned
            ? load_u32_aligned(weight + 128 + 4 * high_index)
            : load_u32(weight + 128 + 4 * high_index)) >> high_shift;
        const int values0 = __vsubss4(
            (low & 0x0f0f0f0f)
                | (((high >> 0) << 4) & 0x30303030),
            0x20202020);
        const int values1 = __vsubss4(
            ((low >> 4) & 0x0f0f0f0f)
                | (((high >> 4) << 4) & 0x30303030),
            0x20202020);
        const int8_t *scales = reinterpret_cast<const int8_t *>(weight + 192)
            + 8 * (iqs / 16) + (iqs % 16) / 4;
        const float d_scale = load_f16(d_values + block * 2);
        const Q8_1Block *activation = activations + block * 8;
#pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            if (input_row + mi >= m) {
                continue;
            }
            const Q8_1Block *row_activation = activation
                + static_cast<int64_t>(input_row + mi) * activation_row_blocks;
            accum[mi] += q6_k_q8_1_decoded_dot(
                values0,
                values1,
                scales[0],
                scales[4],
                d_scale,
                row_activation,
                iqs);
        }
    }

#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            accum[mi] += __shfl_xor_sync(0xffffffff, accum[mi], delta);
        }
        if (lane == 0 && input_row + mi < m) {
            out[static_cast<int64_t>(input_row + mi) * n + output_row]
                = __float2bfloat16_rn(accum[mi]);
        }
    }
}

// M=8 verify variant of the split Q6 tile.  The regular tile reuses weights
// across the eight input rows but leaves Q8_1 activations in global memory;
// stage the complete fixed verify bundle once per CTA so every output-row
// warp consumes the same activation blocks from shared memory.  DFlash2's
// graph fixes M=8, which keeps this allocation at 46,080 bytes for K=5120.
template <bool aligned>
__global__ void gguf_gemm_q8_split_tile_cached_activation_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    extern __shared__ unsigned char activation_storage[];
    auto *activation_cache = reinterpret_cast<Q8_1Block *>(activation_storage);
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;
    const int activation_count = m * activation_row_blocks;
    for (int index = threadIdx.x; index < activation_count; index += blockDim.x) {
        activation_cache[index] = activations[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    const int input_row = blockIdx.y * 8;
    if (output_row >= n) {
        return;
    }
    const int blocks_per_row = k / 256;
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    const uint8_t *d_values = row + blocks_per_row * 208;
    float accum[8] = {};

    for (int block = 0; block < blocks_per_row; ++block) {
        const uint8_t *weight = row + block * 208;
        const int iqs = lane;
        const int high_index = 8 * (iqs / 16) + iqs % 8;
        const int high_shift = 2 * ((iqs % 16) / 8);
        const int low = static_cast<int>(aligned
            ? load_u32_aligned(weight + 4 * iqs)
            : load_u32(weight + 4 * iqs));
        const int high = static_cast<int>(aligned
            ? load_u32_aligned(weight + 128 + 4 * high_index)
            : load_u32(weight + 128 + 4 * high_index)) >> high_shift;
        const int values0 = __vsubss4(
            (low & 0x0f0f0f0f)
                | (((high >> 0) << 4) & 0x30303030),
            0x20202020);
        const int values1 = __vsubss4(
            ((low >> 4) & 0x0f0f0f0f)
                | (((high >> 4) << 4) & 0x30303030),
            0x20202020);
        const int8_t *scales = reinterpret_cast<const int8_t *>(weight + 192)
            + 8 * (iqs / 16) + (iqs % 16) / 4;
        const float d_scale = load_f16(d_values + block * 2);
        const Q8_1Block *activation = activation_cache + block * 8;
#pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            if (input_row + mi >= m) {
                continue;
            }
            const Q8_1Block *row_activation = activation
                + static_cast<int64_t>(input_row + mi) * activation_row_blocks;
            accum[mi] += q6_k_q8_1_decoded_dot(
                values0,
                values1,
                scales[0],
                scales[4],
                d_scale,
                row_activation,
                iqs);
        }
    }

#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            accum[mi] += __shfl_xor_sync(0xffffffff, accum[mi], delta);
        }
        if (lane == 0 && input_row + mi < m) {
            out[static_cast<int64_t>(input_row + mi) * n + output_row]
                = __float2bfloat16_rn(accum[mi]);
        }
    }
}

template <bool aligned>
__global__ void gguf_gemm_q8_0_split_tile_kernel(
    __nv_bfloat16 *out,
    const Q8_1Block *activations,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    const int input_row = blockIdx.y * 8;
    if (output_row >= n) {
        return;
    }
    const int blocks_per_row = k / 32;
    const int activation_row_blocks = ((k + 511) / 512) * 512 / 32;
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    const uint8_t *d_values = row + blocks_per_row * 32;
    const int iqs = 2 * (lane % 4);
    float accum[8] = {};

    for (int block = lane / 4; block < blocks_per_row; block += 8) {
        const uint8_t *weight = row + block * 32;
        const int weight0 = static_cast<int>(aligned
            ? load_u32_aligned(weight + 4 * iqs)
            : load_u32(weight + 4 * iqs));
        const int weight1 = static_cast<int>(aligned
            ? load_u32_aligned(weight + 4 * (iqs + 1))
            : load_u32(weight + 4 * (iqs + 1)));
        const float d_scale = load_f16(d_values + block * 2);
        const Q8_1Block *activation = activations + block;
#pragma unroll
        for (int mi = 0; mi < 8; ++mi) {
            if (input_row + mi >= m) {
                continue;
            }
            const Q8_1Block *row_activation = activation
                + static_cast<int64_t>(input_row + mi) * activation_row_blocks;
            accum[mi] += q8_0_q8_1_decoded_dot(
                weight0,
                weight1,
                d_scale,
                row_activation,
                iqs);
        }
    }

#pragma unroll
    for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            accum[mi] += __shfl_xor_sync(0xffffffff, accum[mi], delta);
        }
        if (lane == 0 && input_row + mi < m) {
            out[static_cast<int64_t>(input_row + mi) * n + output_row]
                = __float2bfloat16_rn(accum[mi]);
        }
    }
}

__host__ __device__ __forceinline__ int base_quant_type(int type) {
    return type == kQ6KAligned || type == kQ6KSplit ? kQ6K
        : type == kQ8_0Split ? kQ8_0
        : type;
}

__host__ __device__ __forceinline__ int block_bytes(int type) {
    const int base = base_quant_type(type);
    return base == kQ4K ? 144
        : base == kQ5K ? 176
        : base == kQ6K ? (type == kQ6KAligned ? 224 : type == kQ6KSplit ? 208 : 210)
        : type == kQ8_0Split ? 32
        : 34;
}

__host__ __device__ __forceinline__ int elements_per_block(int type) {
    return base_quant_type(type) == kQ8_0 ? 32 : 256;
}

__device__ __forceinline__ float decode_value(
    const uint8_t *row, int index, int row_bytes, int type) {
    const int base_type = base_quant_type(type);
    const int elements = elements_per_block(type);
    const int block = index / elements;
    const int within = index - block * elements;
    // The row stride is passed by the host, but the block stride is fixed by
    // the format. Keeping this switch here avoids a second kernel family and
    // makes the ABI stable for all four GGML types used by the checkpoint.
    const uint8_t *payload;
    const uint8_t *d_ptr = nullptr;
    if (type == kQ6KSplit) {
        const int blocks_per_row = row_bytes / 210;
        payload = row + block * 208;
        d_ptr = row + blocks_per_row * 208 + block * 2;
    } else if (type == kQ8_0Split) {
        const int blocks_per_row = row_bytes / 34;
        payload = row + block * 32;
        d_ptr = row + blocks_per_row * 32 + block * 2;
    } else {
        payload = row + block * block_bytes(type);
    }
    if (base_type == kQ4K) {
        return q4_k_value(payload, within);
    }
    if (base_type == kQ5K) {
        return q5_k_value(payload, within);
    }
    if (base_type == kQ6K) {
        return type == kQ6KSplit
            ? q6_k_split_value(payload, within, d_ptr)
            : q6_k_value(payload, within);
    }
    return type == kQ8_0Split
        ? q8_0_split_value(payload, within, d_ptr)
        : q8_0_value(payload, within);
}

template <typename T, int M_TILE>
__global__ void gguf_gemm_tile_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int m,
    int n,
    int k,
    int row_bytes,
    int type) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int output_row = blockIdx.x * 8 + warp;
    const int input_row = blockIdx.y * M_TILE;
    if (output_row >= n) {
        return;
    }

    float accum[M_TILE] = {};
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    for (int index = lane; index < k; index += 32) {
        const float weight = weight_for_compute<T>(decode_value(row, index, row_bytes, type));
#pragma unroll
        for (int mi = 0; mi < M_TILE; ++mi) {
            const int row_index = input_row + mi;
            if (row_index < m) {
                accum[mi] = fmaf(load_activation(
                                     x + static_cast<int64_t>(row_index) * k + index),
                                 weight, accum[mi]);
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < M_TILE; ++mi) {
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            accum[mi] += __shfl_down_sync(0xffffffff, accum[mi], delta);
        }
        if (lane == 0 && input_row + mi < m) {
            store_activation(
                out + static_cast<int64_t>(input_row + mi) * n + output_row, accum[mi]);
        }
    }
}

template <typename T>
__global__ void gguf_gemv_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes,
    int type) {
    const int output_row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = 0.0f;
    for (int index = lane; index < k; index += 32) {
        const float weight = weight_for_compute<T>(decode_value(row, index, row_bytes, type));
        accum = fmaf(load_activation(x + index), weight, accum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_down_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T, bool split>
__device__ __forceinline__ float gguf_gemv_q8_exact_lane_accum(
    const T *x,
    const uint8_t *row,
    int k,
    int row_bytes,
    int lane) {
    const int blocks_per_row = k / 32;
    const uint8_t *d_values = split ? row + blocks_per_row * 32 : nullptr;
    float accum = 0.0f;
    const int group = lane / 8;
    for (int block = group; block < blocks_per_row; block += 4) {
        const uint8_t *block_ptr = row + block * (split ? 32 : 34);
        const uint8_t *q_values = block_ptr + (split ? 0 : 2);
        const float d = split
            ? load_f16(d_values + block * 2)
            : load_f16(block_ptr);
        const int value_offset = (lane & 7) * 4;
#pragma unroll
        for (int value = 0; value < 4; ++value) {
            const float weight = weight_for_compute<T>(
                d * static_cast<float>(reinterpret_cast<const int8_t *>(q_values)[
                    value_offset + value]));
            accum = fmaf(
                load_activation(x + block * 32 + value_offset + value),
                weight,
                accum);
        }
    }
    return accum;
}

template <typename T, bool split, int warps_per_block = 8>
__global__ void gguf_gemv_q8_exact_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_q8_exact_lane_accum<T, split>(
        x, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_down_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

// The decode input is shared by every output row in one CTA.  The ordinary
// warp-per-row kernel relies on L1 for that reuse, but a 5120-wide F32 row is
// large enough that eight warps can evict each other's lines while streaming
// different packed rows.  This graph-safe variant stages the input once in
// dynamic shared memory and leaves the packed-weight traversal unchanged.
template <typename T, bool split, int warps_per_block = 8>
__global__ void gguf_gemv_q8_exact_cached_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    extern __shared__ unsigned char q8_activation_storage[];
    T *q8_activation_cache = reinterpret_cast<T *>(q8_activation_storage);
    for (int index = threadIdx.x; index < k; index += blockDim.x) {
        q8_activation_cache[index] = x[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_q8_exact_lane_accum<T, split>(
        q8_activation_cache, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_down_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T, int type>
__device__ __forceinline__ float gguf_gemv_k_exact_lane_accum(
    const T *x,
    const uint8_t *row,
    int k,
    int row_bytes,
    int lane) {
    const int blocks_per_row = k / 256;
    const uint8_t *d_values = type == kQ6KSplit
        ? row + blocks_per_row * 208
        : nullptr;
    float accum = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        const uint8_t *block_ptr = row + block * block_bytes(type);
        const float d_local = lane == 0
            ? (type == kQ6KSplit
                ? load_f16(d_values + block * 2)
                : type == kQ6KAligned || type == kQ6K
                    ? load_f16(block_ptr + 208)
                    : load_f16(block_ptr))
            : 0.0f;
        const float d = __shfl_sync(0xffffffff, d_local, 0);
        const float dmin_local = lane == 0 && (type == kQ4K || type == kQ5K)
            ? load_f16(block_ptr + 2)
            : 0.0f;
        const float dmin = __shfl_sync(0xffffffff, dmin_local, 0);
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int index = lane + step * 32;
            float weight;
            if constexpr (type == kQ4K || type == kQ5K) {
                const int chunk = index / 64;
                const int local = index % 64;
                const int nibble = local / 32;
                const int qbyte = local % 32;
                const int low_base = type == kQ5K ? 48 : 16;
                const uint8_t low = block_ptr[low_base + chunk * 32 + qbyte];
                const int high = type == kQ5K
                    ? ((block_ptr[16 + qbyte] >> (chunk * 2 + nibble)) & 1)
                    : 0;
                const int q = ((low >> (nibble * 4)) & 0x0f) | (high << 4);
                const int scale_index = chunk * 2 + nibble;
                weight = d * k_scale(block_ptr + 4, scale_index) * static_cast<float>(q)
                    - dmin * k_min(block_ptr + 4, scale_index);
            } else {
                const int half = index / 128;
                const int local = index % 128;
                const int group = local / 32;
                const int j = local % 32;
                const int low_offset = half * 64 + (group & 1) * 32 + j;
                const int high_offset = half * 32 + j;
                const int low_shift = group >= 2 ? 4 : 0;
                const int high_shift = group * 2;
                const int q = ((block_ptr[low_offset] >> low_shift) & 0x0f)
                    | (((block_ptr[128 + high_offset] >> high_shift) & 0x03) << 4);
                const int scale_index = half * 8 + group * 2 + j / 16;
                const int8_t scale = reinterpret_cast<const int8_t *>(
                    block_ptr + 192)[scale_index];
                weight = d * static_cast<float>(scale) * static_cast<float>(q - 32);
            }
            accum = fmaf(
                load_activation(x + block * 256 + index),
                weight_for_compute<T>(weight),
                accum);
        }
    }
    return accum;
}

template <typename T, int type, int warps_per_block = 8>
__global__ void gguf_gemv_k_exact_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_k_exact_lane_accum<T, type>(
        x, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_down_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T, int type, int warps_per_block = 8>
__global__ void gguf_gemv_k_exact_cached_kernel(
    T *out,
    const T *x,
    const uint8_t *packed,
    int n,
    int k,
    int row_bytes) {
    extern __shared__ unsigned char k_activation_storage[];
    T *k_activation_cache = reinterpret_cast<T *>(k_activation_storage);
    for (int index = threadIdx.x; index < k; index += blockDim.x) {
        k_activation_cache[index] = x[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= n) {
        return;
    }
    const uint8_t *row = packed + static_cast<int64_t>(output_row) * row_bytes;
    float accum = gguf_gemv_k_exact_lane_accum<T, type>(
        k_activation_cache, row, k, row_bytes, lane);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_down_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T, int warps_per_block = 8>
__global__ void gguf_gemv_direct_mixed_kernel(
    T *out,
    const T *x,
    const GgufMixedProjection *projections,
    int projection_count,
    int total_n,
    int k) {
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= total_n) {
        return;
    }
    int projection_index = 0;
    for (; projection_index + 1 < projection_count; ++projection_index) {
        if (output_row < projections[projection_index + 1].output_offset) {
            break;
        }
    }
    const GgufMixedProjection projection = projections[projection_index];
    const int local_row = output_row - projection.output_offset;
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(projection.packed_ptr);
    const uint8_t *row = packed + static_cast<int64_t>(local_row) * projection.row_bytes;
    float accum;
    switch (projection.type) {
        case kQ4K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ4K>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ5K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ5K>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ6K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6K>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ6KAligned:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6KAligned>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ6KSplit:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6KSplit>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0:
            accum = gguf_gemv_q8_exact_lane_accum<T, false>(
                x, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0Split:
            accum = gguf_gemv_q8_exact_lane_accum<T, true>(
                x, row, k, projection.row_bytes, lane);
            break;
        default:
            return;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T, int warps_per_block = 8>
__global__ void gguf_gemv_direct_mixed_cached_kernel(
    T *out,
    const T *x,
    const GgufMixedProjection *projections,
    int projection_count,
    int total_n,
    int k) {
    extern __shared__ unsigned char mixed_activation_storage[];
    T *mixed_activation_cache = reinterpret_cast<T *>(mixed_activation_storage);
    for (int index = threadIdx.x; index < k; index += blockDim.x) {
        mixed_activation_cache[index] = x[index];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int lane = threadIdx.x & 31;
    if (output_row >= total_n) {
        return;
    }
    int projection_index = 0;
    for (; projection_index + 1 < projection_count; ++projection_index) {
        if (output_row < projections[projection_index + 1].output_offset) {
            break;
        }
    }
    const GgufMixedProjection projection = projections[projection_index];
    const int local_row = output_row - projection.output_offset;
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(projection.packed_ptr);
    const uint8_t *row = packed + static_cast<int64_t>(local_row) * projection.row_bytes;
    float accum;
    switch (projection.type) {
        case kQ4K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ4K>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ5K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ5K>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6K:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6K>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6KAligned:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6KAligned>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ6KSplit:
            accum = gguf_gemv_k_exact_lane_accum<T, kQ6KSplit>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0:
            accum = gguf_gemv_q8_exact_lane_accum<T, false>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        case kQ8_0Split:
            accum = gguf_gemv_q8_exact_lane_accum<T, true>(
                mixed_activation_cache, row, k, projection.row_bytes, lane);
            break;
        default:
            return;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        accum += __shfl_xor_sync(0xffffffff, accum, delta);
    }
    if (lane == 0) {
        store_activation(out + output_row, accum);
    }
}

template <typename T>
__global__ void gguf_dequant_rows_kernel(
    T *out,
    const int64_t *ids,
    const uint8_t *packed,
    int rows,
    int k,
    int row_bytes,
    int type) {
    const int row_index = blockIdx.x;
    if (row_index >= rows) {
        return;
    }
    const int64_t source_row = ids[row_index];
    const uint8_t *row = packed + source_row * row_bytes;
    for (int index = threadIdx.x; index < k; index += blockDim.x) {
        store_activation(
            out + static_cast<int64_t>(row_index) * k + index,
            decode_value(row, index, row_bytes, type));
    }
}

bool valid_geometry(int m, int n, int k, int row_bytes, int type) {
    if (m <= 0 || n <= 0 || k <= 0 || row_bytes <= 0) {
        return false;
    }
    const int base_type = base_quant_type(type);
    if (base_type < kQ4K || base_type > kQ8_0) {
        return false;
    }
    const int elements = elements_per_block(type);
    const int expected_row_bytes = type == kQ6KSplit
        ? (k / elements) * 210
        : type == kQ8_0Split
            ? (k / elements) * 34
        : (k / elements) * block_bytes(type);
    return k % elements == 0 && row_bytes == expected_row_bytes;
}

bool q6_mmq_mma_enabled() {
    const char *value = std::getenv("QSR_GGUF_MMQ_MMA");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int launch_gguf_gemm_q8_mmq(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (m < 1 || n < 1 || k <= 0) {
        return 1;
    }
    if (type == kQ6KSplit) {
        if (k % 256 != 0 || row_bytes != (k / 256) * 210) {
            return 1;
        }
        if (q6_mmq_mma_enabled()) {
            const dim3 grid((n + kQ6MmqMmaY - 1) / kQ6MmqMmaY,
                (m + kQ6MmqMmaX - 1) / kQ6MmqMmaX,
                1);
            const dim3 block(32, kQ6MmqMmaWarps, 1);
            gguf_gemm_q8_q6_mmq_split_mma_kernel<<<grid, block, 0, stream>>>(
                static_cast<__nv_bfloat16 *>(out_ptr),
                static_cast<const Q8_1Block *>(activation_ptr),
                static_cast<const uint8_t *>(packed_ptr),
                m,
                n,
                k,
                row_bytes);
        } else {
            const dim3 grid((n + kQ6MmqY - 1) / kQ6MmqY,
                (m + kQ6MmqX - 1) / kQ6MmqX,
                1);
            const dim3 block(32, kQ6MmqWarps, 1);
            if (m % kQ6MmqX == 0 && n % kQ6MmqY == 0) {
                gguf_gemm_q8_q6_mmq_split_kernel<false><<<grid, block, 0, stream>>>(
                    static_cast<__nv_bfloat16 *>(out_ptr),
                    static_cast<const Q8_1Block *>(activation_ptr),
                    static_cast<const uint8_t *>(packed_ptr),
                    m,
                    n,
                    k,
                    row_bytes);
            } else {
                gguf_gemm_q8_q6_mmq_split_kernel<true><<<grid, block, 0, stream>>>(
                    static_cast<__nv_bfloat16 *>(out_ptr),
                    static_cast<const Q8_1Block *>(activation_ptr),
                    static_cast<const uint8_t *>(packed_ptr),
                    m,
                    n,
                    k,
                    row_bytes);
            }
        }
    } else if (type == kQ5K) {
        if (k % 256 != 0 || row_bytes != (k / 256) * 176) {
            return 1;
        }
        const dim3 grid((n + kQ5MmqY - 1) / kQ5MmqY,
            (m + kQ5MmqX - 1) / kQ5MmqX,
            1);
        const dim3 block(32, kQ5MmqWarps, 1);
        gguf_gemm_q8_q5_mmq_kernel<<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(out_ptr),
            static_cast<const Q8_1Block *>(activation_ptr),
            static_cast<const uint8_t *>(packed_ptr),
            m,
            n,
            k,
            row_bytes);
    } else if (type == kQ8_0 || type == kQ8_0Split) {
        if (k % 128 != 0 || row_bytes != (k / 32) * 34) {
            return 1;
        }
        const dim3 grid((n + kQ8MmqY - 1) / kQ8MmqY,
            (m + kQ8MmqX - 1) / kQ8MmqX,
            1);
        const dim3 block(32, kQ8MmqWarps, 1);
        if (type == kQ8_0Split) {
            gguf_gemm_q8_q8_mmq_kernel<true><<<grid, block, 0, stream>>>(
                static_cast<__nv_bfloat16 *>(out_ptr),
                static_cast<const Q8_1Block *>(activation_ptr),
                static_cast<const uint8_t *>(packed_ptr),
                m,
                n,
                k,
                row_bytes);
        } else {
            gguf_gemm_q8_q8_mmq_kernel<false><<<grid, block, 0, stream>>>(
                static_cast<__nv_bfloat16 *>(out_ptr),
                static_cast<const Q8_1Block *>(activation_ptr),
                static_cast<const uint8_t *>(packed_ptr),
                m,
                n,
                k,
                row_bytes);
        }
    } else {
        return 1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int launch_gguf_gemm_q8(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    auto *out = static_cast<__nv_bfloat16 *>(out_ptr);
    const auto *activations = static_cast<const Q8_1Block *>(activation_ptr);
    const auto *packed = static_cast<const uint8_t *>(packed_ptr);
    if (m == 1) {
        switch (type) {
            case kQ4K:
                gguf_gemv_q8_kernel<kQ4K><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ5K:
                gguf_gemv_q8_kernel<kQ5K><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ6K:
                gguf_gemv_q8_kernel<kQ6K><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ6KAligned:
                gguf_gemv_q8_kernel<kQ6KAligned><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ6KSplit:
                gguf_gemv_q8_kernel<kQ6KSplit><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ8_0:
                gguf_gemv_q8_kernel<kQ8_0><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            case kQ8_0Split:
                gguf_gemv_q8_kernel<kQ8_0Split><<<n, 32, 0, stream>>>(
                    out, activations, packed, n, k, row_bytes);
                break;
            default:
                return 1;
        }
    } else {
        const dim3 grid((n + 7) / 8, (m + 7) / 8, 1);
        switch (type) {
            case kQ4K:
                gguf_gemm_q8_tile_kernel<kQ4K><<<grid, 256, 0, stream>>>(
                    out, activations, packed, m, n, k, row_bytes);
                break;
            case kQ5K:
                gguf_gemm_q8_tile_kernel<kQ5K><<<grid, 256, 0, stream>>>(
                    out, activations, packed, m, n, k, row_bytes);
                break;
            case kQ6K:
                gguf_gemm_q8_tile_kernel<kQ6K><<<grid, 256, 0, stream>>>(
                    out, activations, packed, m, n, k, row_bytes);
                break;
            case kQ6KAligned:
                gguf_gemm_q8_tile_kernel<kQ6KAligned><<<grid, 256, 0, stream>>>(
                    out, activations, packed, m, n, k, row_bytes);
                break;
            case kQ6KSplit:
                if ((reinterpret_cast<uintptr_t>(packed) & 3u) == 0u
                    && (row_bytes & 3) == 0) {
                    gguf_gemm_q8_split_tile_kernel<true><<<grid, 256, 0, stream>>>(
                        out, activations, packed, m, n, k, row_bytes);
                } else {
                    gguf_gemm_q8_split_tile_kernel<false><<<grid, 256, 0, stream>>>(
                        out, activations, packed, m, n, k, row_bytes);
                }
                break;
            case kQ8_0:
                gguf_gemm_q8_tile_kernel<kQ8_0><<<grid, 256, 0, stream>>>(
                    out, activations, packed, m, n, k, row_bytes);
                break;
            case kQ8_0Split:
                if ((reinterpret_cast<uintptr_t>(packed) & 3u) == 0u
                    && (row_bytes & 3) == 0) {
                    gguf_gemm_q8_0_split_tile_kernel<true><<<grid, 256, 0, stream>>>(
                        out, activations, packed, m, n, k, row_bytes);
                } else {
                    gguf_gemm_q8_0_split_tile_kernel<false><<<grid, 256, 0, stream>>>(
                        out, activations, packed, m, n, k, row_bytes);
                }
                break;
            default:
                return 1;
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int launch_gguf_gemm_q8_f32(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    auto *out = static_cast<float *>(out_ptr);
    const auto *activations = static_cast<const Q8_1Block *>(activation_ptr);
    const auto *packed = static_cast<const uint8_t *>(packed_ptr);
    switch (type) {
        case kQ4K:
            gguf_gemv_q8_kernel<kQ4K, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ5K:
            gguf_gemv_q8_kernel<kQ5K, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ6K:
            gguf_gemv_q8_kernel<kQ6K, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ6KAligned:
            gguf_gemv_q8_kernel<kQ6KAligned, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ6KSplit:
            gguf_gemv_q8_kernel<kQ6KSplit, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ8_0:
            gguf_gemv_q8_kernel<kQ8_0, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        case kQ8_0Split:
            gguf_gemv_q8_kernel<kQ8_0Split, float><<<n, 32, 0, stream>>>(
                out, activations, packed, n, k, row_bytes);
            break;
        default:
            return 1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

constexpr int kCachedActivationMaxBytes = 48 * 1024;

constexpr int q8_activation_shared_bytes(int k) {
    const int padded_k = ((k + 511) / 512) * 512;
    return (padded_k / 32) * static_cast<int>(sizeof(Q8_1Block));
}

template <typename output_t>
int launch_gguf_gemm_q8_cached(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (q8_activation_shared_bytes(k) > kCachedActivationMaxBytes) {
        return 1;
    }
    if (m == 8 && sizeof(output_t) == sizeof(__nv_bfloat16) && type == kQ6KSplit) {
        const int shared_bytes = q8_activation_shared_bytes(k) * m;
        if (shared_bytes > kCachedActivationMaxBytes) {
            return 1;
        }
        const dim3 grid((n + 7) / 8, 1, 1);
        const dim3 block(256, 1, 1);
        if ((reinterpret_cast<uintptr_t>(packed_ptr) & 3u) == 0u
            && (row_bytes & 3) == 0) {
            gguf_gemm_q8_split_tile_cached_activation_kernel<true>
                <<<grid, block, shared_bytes, stream>>>(
                    static_cast<__nv_bfloat16 *>(out_ptr),
                    static_cast<const Q8_1Block *>(activation_ptr),
                    static_cast<const uint8_t *>(packed_ptr),
                    m,
                    n,
                    k,
                    row_bytes);
        } else {
            gguf_gemm_q8_split_tile_cached_activation_kernel<false>
                <<<grid, block, shared_bytes, stream>>>(
                    static_cast<__nv_bfloat16 *>(out_ptr),
                    static_cast<const Q8_1Block *>(activation_ptr),
                    static_cast<const uint8_t *>(packed_ptr),
                    m,
                    n,
                    k,
                    row_bytes);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 2;
    }
    if (m != 1) {
        return 1;
    }
    const dim3 grid((n + 7) / 8, 1, 1);
    const size_t shared_bytes = static_cast<size_t>(q8_activation_shared_bytes(k));
    auto *out = static_cast<output_t *>(out_ptr);
    const auto *activations = static_cast<const Q8_1Block *>(activation_ptr);
    const auto *packed = static_cast<const uint8_t *>(packed_ptr);
    switch (type) {
        case kQ4K:
            gguf_gemv_q8_cached_kernel<kQ4K, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ5K:
            gguf_gemv_q8_cached_kernel<kQ5K, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ6K:
            gguf_gemv_q8_cached_kernel<kQ6K, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ6KAligned:
            gguf_gemv_q8_cached_kernel<kQ6KAligned, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ6KSplit:
            gguf_gemv_q8_cached_kernel<kQ6KSplit, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ8_0:
            gguf_gemv_q8_cached_kernel<kQ8_0, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        case kQ8_0Split:
            gguf_gemv_q8_cached_kernel<kQ8_0Split, output_t>
                <<<grid, 256, shared_bytes, stream>>>(out, activations, packed, n, k, row_bytes);
            break;
        default:
            return 1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <typename output_t>
int launch_gguf_gemm_q8_mixed_cached(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (q8_activation_shared_bytes(k) > kCachedActivationMaxBytes) {
        return 1;
    }
    const dim3 grid((total_n + 7) / 8, 1, 1);
    const size_t shared_bytes = static_cast<size_t>(q8_activation_shared_bytes(k));
    gguf_gemv_q8_mixed_cached_kernel<output_t><<<grid, 256, shared_bytes, stream>>>(
        static_cast<output_t *>(out_ptr),
        static_cast<const Q8_1Block *>(activation_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
int launch_gguf_gemm_direct_cached(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (k * static_cast<int>(sizeof(T)) > kCachedActivationMaxBytes) {
        return 1;
    }
    auto *out = static_cast<T *>(out_ptr);
    const auto *x = static_cast<const T *>(activation_ptr);
    const auto *packed = static_cast<const uint8_t *>(packed_ptr);
    const dim3 grid((n + 7) / 8, 1, 1);
    const size_t shared_bytes = static_cast<size_t>(k) * sizeof(T);
    if (type == kQ8_0) {
        gguf_gemv_q8_exact_cached_kernel<T, false><<<grid, 256, shared_bytes, stream>>>(
            out, x, packed, n, k, row_bytes);
    } else if (type == kQ8_0Split) {
        gguf_gemv_q8_exact_cached_kernel<T, true><<<grid, 256, shared_bytes, stream>>>(
            out, x, packed, n, k, row_bytes);
    } else {
        switch (type) {
            case kQ4K:
                gguf_gemv_k_exact_cached_kernel<T, kQ4K>
                    <<<grid, 256, shared_bytes, stream>>>(out, x, packed, n, k, row_bytes);
                break;
            case kQ5K:
                gguf_gemv_k_exact_cached_kernel<T, kQ5K>
                    <<<grid, 256, shared_bytes, stream>>>(out, x, packed, n, k, row_bytes);
                break;
            case kQ6K:
                gguf_gemv_k_exact_cached_kernel<T, kQ6K>
                    <<<grid, 256, shared_bytes, stream>>>(out, x, packed, n, k, row_bytes);
                break;
            case kQ6KAligned:
                gguf_gemv_k_exact_cached_kernel<T, kQ6KAligned>
                    <<<grid, 256, shared_bytes, stream>>>(out, x, packed, n, k, row_bytes);
                break;
            case kQ6KSplit:
                gguf_gemv_k_exact_cached_kernel<T, kQ6KSplit>
                    <<<grid, 256, shared_bytes, stream>>>(out, x, packed, n, k, row_bytes);
                break;
            default:
                return 1;
        }
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <typename T>
int launch_gguf_gemm_direct_mixed_cached(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (k * static_cast<int>(sizeof(T)) > kCachedActivationMaxBytes) {
        return 1;
    }
    const dim3 grid((total_n + 7) / 8, 1, 1);
    const size_t shared_bytes = static_cast<size_t>(k) * sizeof(T);
    gguf_gemv_direct_mixed_cached_kernel<T><<<grid, 256, shared_bytes, stream>>>(
        static_cast<T *>(out_ptr),
        static_cast<const T *>(activation_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}  // namespace

extern "C" {

QSR_EXPORT int qsr_gguf_qk_abi_version() {
    return 12;
}

QSR_EXPORT int qsr_gguf_quantize_q8_sm120(
    const void *x_ptr,
    void *activation_ptr,
    int m,
    int k,
    cudaStream_t stream) {
    if (!x_ptr || !activation_ptr || m <= 0 || k <= 0) {
        return 1;
    }
    const int padded_k = ((k + 511) / 512) * 512;
    const dim3 quant_grid((padded_k + 255) / 256, m, 1);
    quantize_q8_1_kernel<__nv_bfloat16><<<quant_grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(x_ptr),
        static_cast<Q8_1Block *>(activation_ptr),
        k,
        padded_k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_quantize_q8_f32_sm120(
    const void *x_ptr,
    void *activation_ptr,
    int m,
    int k,
    cudaStream_t stream) {
    if (!x_ptr || !activation_ptr || m <= 0 || k <= 0) {
        return 1;
    }
    const int padded_k = ((k + 511) / 512) * 512;
    const dim3 quant_grid((padded_k + 255) / 256, m, 1);
    quantize_q8_1_kernel<float><<<quant_grid, 256, 0, stream>>>(
        static_cast<const float *>(x_ptr),
        static_cast<Q8_1Block *>(activation_ptr),
        k,
        padded_k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_q8_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    void *activation_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || !activation_ptr
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    const int padded_k = ((k + 511) / 512) * 512;
    const dim3 quant_grid((padded_k + 255) / 256, m, 1);
    quantize_q8_1_kernel<__nv_bfloat16><<<quant_grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(x_ptr),
        static_cast<Q8_1Block *>(activation_ptr),
        k,
        padded_k);
    if (cudaGetLastError() != cudaSuccess) {
        return 2;
    }
    return launch_gguf_gemm_q8(
        out_ptr, activation_ptr, packed_ptr, m, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_prequantized_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !packed_ptr
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_q8(
        out_ptr, activation_ptr, packed_ptr, m, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_prequantized_cached_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !packed_ptr
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_q8_cached<__nv_bfloat16>(
        out_ptr, activation_ptr, packed_ptr, m, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_mmq_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !packed_ptr
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_q8_mmq(
        out_ptr, activation_ptr, packed_ptr, m, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_prequantized_f32_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !packed_ptr || m != 1
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_q8_f32(
        out_ptr, activation_ptr, packed_ptr, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_prequantized_cached_f32_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !packed_ptr || m != 1
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_q8_cached<float>(
        out_ptr, activation_ptr, packed_ptr, m, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_f32_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    void *activation_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || !activation_ptr
        || m != 1 || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    const int padded_k = ((k + 511) / 512) * 512;
    const dim3 quant_grid((padded_k + 255) / 256, m, 1);
    quantize_q8_1_kernel<float><<<quant_grid, 256, 0, stream>>>(
        static_cast<const float *>(x_ptr),
        static_cast<Q8_1Block *>(activation_ptr),
        k,
        padded_k);
    if (cudaGetLastError() != cudaSuccess) {
        return 2;
    }
    return launch_gguf_gemm_q8_f32(
        out_ptr, activation_ptr, packed_ptr, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_mixed_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    gguf_gemv_q8_mixed_kernel<__nv_bfloat16><<<total_n, 32, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(out_ptr),
        static_cast<const Q8_1Block *>(activation_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_q8_mixed_f32_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    gguf_gemv_q8_mixed_kernel<float><<<total_n, 32, 0, stream>>>(
        static_cast<float *>(out_ptr),
        static_cast<const Q8_1Block *>(activation_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_q8_mixed_cached_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    return launch_gguf_gemm_q8_mixed_cached<__nv_bfloat16>(
        out_ptr,
        activation_ptr,
        projections_ptr,
        projection_count,
        total_n,
        k,
        stream);
}

QSR_EXPORT int qsr_gguf_gemm_q8_mixed_cached_f32_sm120(
    void *out_ptr,
    const void *activation_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !activation_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    return launch_gguf_gemm_q8_mixed_cached<float>(
        out_ptr,
        activation_ptr,
        projections_ptr,
        projection_count,
        total_n,
        k,
        stream);
}

QSR_EXPORT int qsr_gguf_gemm_direct_mixed_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    gguf_gemv_direct_mixed_kernel<__nv_bfloat16><<<(total_n + 7) / 8, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(out_ptr),
        static_cast<const __nv_bfloat16 *>(x_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_direct_mixed_f32_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    gguf_gemv_direct_mixed_kernel<float><<<(total_n + 7) / 8, 256, 0, stream>>>(
        static_cast<float *>(out_ptr),
        static_cast<const float *>(x_ptr),
        static_cast<const GgufMixedProjection *>(projections_ptr),
        projection_count,
        total_n,
        k);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_direct_mixed_cached_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    return launch_gguf_gemm_direct_mixed_cached<__nv_bfloat16>(
        out_ptr, x_ptr, projections_ptr, projection_count, total_n, k, stream);
}

QSR_EXPORT int qsr_gguf_gemm_direct_mixed_cached_f32_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *projections_ptr,
    int projection_count,
    int total_n,
    int k,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !projections_ptr
        || projection_count <= 0 || total_n <= 0 || k <= 0) {
        return 1;
    }
    return launch_gguf_gemm_direct_mixed_cached<float>(
        out_ptr, x_ptr, projections_ptr, projection_count, total_n, k, stream);
}

QSR_EXPORT int qsr_gguf_gemm_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    auto *out = static_cast<__nv_bfloat16 *>(out_ptr);
    auto *x = static_cast<const __nv_bfloat16 *>(x_ptr);
    auto *packed = static_cast<const uint8_t *>(packed_ptr);
    if (m == 1) {
        if (type == kQ8_0) {
            gguf_gemv_q8_exact_kernel<__nv_bfloat16, false>
                <<<(n + 7) / 8, 256, 0, stream>>>(
                out, x, packed, n, k, row_bytes);
        } else if (type == kQ8_0Split) {
            gguf_gemv_q8_exact_kernel<__nv_bfloat16, true>
                <<<(n + 7) / 8, 256, 0, stream>>>(
                out, x, packed, n, k, row_bytes);
        } else {
            switch (type) {
                case kQ4K:
                    gguf_gemv_k_exact_kernel<__nv_bfloat16, kQ4K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ5K:
                    gguf_gemv_k_exact_kernel<__nv_bfloat16, kQ5K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ6K:
                    gguf_gemv_k_exact_kernel<__nv_bfloat16, kQ6K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ6KAligned:
                    gguf_gemv_k_exact_kernel<__nv_bfloat16, kQ6KAligned>
                        <<<(n + 7) / 8, 256, 0, stream>>>(out, x, packed, n, k, row_bytes);
                    break;
                case kQ6KSplit:
                    gguf_gemv_k_exact_kernel<__nv_bfloat16, kQ6KSplit>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                default:
                    return 1;
            }
        }
    } else {
        dim3 grid((n + 7) / 8, (m + 7) / 8, 1);
        gguf_gemm_tile_kernel<__nv_bfloat16, 8><<<grid, 256, 0, stream>>>(
            out, x, packed, m, n, k, row_bytes, type);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_f32_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    auto *out = static_cast<float *>(out_ptr);
    auto *x = static_cast<const float *>(x_ptr);
    auto *packed = static_cast<const uint8_t *>(packed_ptr);
    if (m == 1) {
        if (type == kQ8_0) {
            gguf_gemv_q8_exact_kernel<float, false>
                <<<(n + 7) / 8, 256, 0, stream>>>(
                out, x, packed, n, k, row_bytes);
        } else if (type == kQ8_0Split) {
            gguf_gemv_q8_exact_kernel<float, true>
                <<<(n + 7) / 8, 256, 0, stream>>>(
                out, x, packed, n, k, row_bytes);
        } else {
            switch (type) {
                case kQ4K:
                    gguf_gemv_k_exact_kernel<float, kQ4K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ5K:
                    gguf_gemv_k_exact_kernel<float, kQ5K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ6K:
                    gguf_gemv_k_exact_kernel<float, kQ6K>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                case kQ6KAligned:
                    gguf_gemv_k_exact_kernel<float, kQ6KAligned>
                        <<<(n + 7) / 8, 256, 0, stream>>>(out, x, packed, n, k, row_bytes);
                    break;
                case kQ6KSplit:
                    gguf_gemv_k_exact_kernel<float, kQ6KSplit>
                        <<<(n + 7) / 8, 256, 0, stream>>>(
                        out, x, packed, n, k, row_bytes);
                    break;
                default:
                    return 1;
            }
        }
    } else {
        dim3 grid((n + 7) / 8, (m + 7) / 8, 1);
        gguf_gemm_tile_kernel<float, 8><<<grid, 256, 0, stream>>>(
            out, x, packed, m, n, k, row_bytes, type);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_gemm_direct_cached_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || m != 1
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_direct_cached<__nv_bfloat16>(
        out_ptr, x_ptr, packed_ptr, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_gemm_direct_cached_f32_sm120(
    void *out_ptr,
    const void *x_ptr,
    const void *packed_ptr,
    int m,
    int n,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !x_ptr || !packed_ptr || m != 1
        || !valid_geometry(m, n, k, row_bytes, type)) {
        return 1;
    }
    return launch_gguf_gemm_direct_cached<float>(
        out_ptr, x_ptr, packed_ptr, n, k, row_bytes, type, stream);
}

QSR_EXPORT int qsr_gguf_dequant_rows_sm120(
    void *out_ptr,
    const void *ids_ptr,
    const void *packed_ptr,
    int rows,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !ids_ptr || !packed_ptr || !valid_geometry(rows, 1, k, row_bytes, type)) {
        return 1;
    }
    gguf_dequant_rows_kernel<__nv_bfloat16><<<rows, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(out_ptr),
        static_cast<const int64_t *>(ids_ptr),
        static_cast<const uint8_t *>(packed_ptr),
        rows,
        k,
        row_bytes,
        type);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

QSR_EXPORT int qsr_gguf_dequant_rows_f32_sm120(
    void *out_ptr,
    const void *ids_ptr,
    const void *packed_ptr,
    int rows,
    int k,
    int row_bytes,
    int type,
    cudaStream_t stream) {
    if (!out_ptr || !ids_ptr || !packed_ptr || !valid_geometry(rows, 1, k, row_bytes, type)) {
        return 1;
    }
    gguf_dequant_rows_kernel<float><<<rows, 256, 0, stream>>>(
        static_cast<float *>(out_ptr),
        static_cast<const int64_t *>(ids_ptr),
        static_cast<const uint8_t *>(packed_ptr),
        rows,
        k,
        row_bytes,
        type);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

}
