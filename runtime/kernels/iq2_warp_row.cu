// IQ2_XS warp-per-row dp4a MoE GEMM for DSV4 prefill (SM120).
// One warp owns one output row; its 32 lanes own the 32 codes of each
// 256-element block.  Per lane: decode one 16-bit code (grid + ksigns
// lookups), pack the 8 j magnitudes into two int32, dp4a against the
// int8-quantized activation, scale by (d * nibble * activation scale).
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
#define DP4A(a, b, c) __dp4a((a), (b), (c))
#else
#define DP4A(a, b, c) ((a) * (b) + (c))
#endif

extern "C" __global__ void __launch_bounds__(256)
iq2_warp_row_kernel(
    const int8_t* __restrict__ xq,       // [E, COLS] int8 activations
    const float* __restrict__ xs,        // [E, COLS/32] activation scales
    const uint8_t* __restrict__ packed,  // [256 experts, ROWS*STRIDE]
    const int64_t* __restrict__ eids,    // [E]
    const int64_t* __restrict__ grid,    // [512]
    const int32_t* __restrict__ ksigns,  // [128]
    float* __restrict__ out,             // [E, ROWS]
    int E,
    int ROWS,
    int COLS,
    int STRIDE)
{
    __shared__ int64_t s_grid[512];
    __shared__ int32_t s_ksigns[128];
    if (threadIdx.x < 512 && threadIdx.y == 0) s_grid[threadIdx.x] = grid[threadIdx.x];
    if (threadIdx.x < 128 && threadIdx.y == 0) s_ksigns[threadIdx.x] = ksigns[threadIdx.x];
    __syncthreads();

    const int e = blockIdx.y;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;  // 0..31 -> code index within a 256 block
    const int eid = (int)eids[e];
    const uint8_t* w = packed + (int64_t)eid * ROWS * STRIDE + (int64_t)row * STRIDE;
    const int8_t* xqrow = xq + e * COLS;
    const float* xsrow = xs + e * (COLS / 32);
    float acc = 0.f;

    const int nblk = COLS / 256;
    for (int kb = 0; kb < nblk; ++kb) {
        const uint8_t* blk = w + (int64_t)kb * 74;
        const uint16_t code =
            (uint16_t)(blk[2 + lane * 2] | (blk[3 + lane * 2] << 8));
        const __half dh = *(const __half*)blk;
        const float d = __half2float(dh);
        const uint8_t sc = blk[66 + lane / 4];
        const float nib = (lane % 4) < 2 ? (float)(sc & 0xF) : (float)(sc >> 4);
        const int64_t g = s_grid[code & 511];
        const int32_t sb = s_ksigns[code >> 9];

        int32_t m0 = 0, m1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int m = (int)((g >> (8 * j)) & 0xFF);
            if (sb & (1 << j)) m = -m;
            m0 |= (m & 0xFF) << (8 * j);
        }
#pragma unroll
        for (int j = 4; j < 8; ++j) {
            int m = (int)((g >> (8 * j)) & 0xFF);
            if (sb & (1 << j)) m = -m;
            m1 |= (m & 0xFF) << (8 * (j - 4));
        }
        const int8_t* xc = xqrow + kb * 256 + lane * 8;
        int32_t x0 = 0, x1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) x0 |= ((int32_t)xc[j] & 0xFF) << (8 * j);
#pragma unroll
        for (int j = 4; j < 8; ++j) x1 |= ((int32_t)xc[j] & 0xFF) << (8 * (j - 4));
        const int32_t sum = DP4A(m0, x0, 0) + DP4A(m1, x1, 0);
        const float xscale = xsrow[kb * 8 + lane / 4];
        acc += (float)sum * d * (0.5f + nib) * 0.25f * xscale;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[e * ROWS + row] = acc;
}

extern "C" void iq2_warp_row_launch(
    const int8_t* xq, const float* xs, const uint8_t* packed,
    const int64_t* eids, const int64_t* grid_t, const int32_t* ksigns,
    float* out, int E, int ROWS, int COLS, int STRIDE)
{
    dim3 block(32, 8);
    dim3 gridc((ROWS + 7) / 8, E);
    iq2_warp_row_kernel<<<gridc, block>>>(
        xq, xs, packed, eids, grid_t, ksigns, out, E, ROWS, COLS, STRIDE);
}
