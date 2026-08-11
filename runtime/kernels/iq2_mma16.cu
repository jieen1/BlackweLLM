// Phase 2: IQ2_XS in-kernel decode -> INT8 MMA (m16n8k16) grouped MoE, SM120.
// Exact IQ2 semantics: nibble scale varies per 8 values; two consecutive
// codes (one K16 partial) always share a nibble value (even code -> lo,
// odd code -> hi of the same scales[] entry), so a K16 partial has a
// uniform d*(0.5+nibble)*0.25*actscale.  mma.sync.m16n8k16 row.col s8 x s8.
//
// Fragment layouts (32 lanes/warp):
//   A [16,16] row-major (M=tokens, K): lane l regs a0,a1.
//     a0 -> A[l/4 + 0, (l%4)*4 + 0..3];  a1 -> A[l/4 + 8, (l%4)*4 + 0..3]
//   B [16,8] col-major (K x N=rows): lane l reg b0.
//     b0 -> B[(l%4)*4 + 0..3, l/4]
//   C/D [16,8] s32: 4 regs.
// One warp: M16 tokens x N8 output rows of one expert (blockIdx.y tile).
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ void mma16(
    int32_t c[4], const int32_t a[2], const int32_t b[1]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]));
#endif
}

extern "C" __global__ void __launch_bounds__(128)
iq2_mma16_kernel(
    const int8_t* __restrict__ xq,       // [E, M_PAD, K]
    const float* __restrict__ xs,        // [E, M_PAD, K/32]
    const uint8_t* __restrict__ packed,  // [256, ROWS*STRIDE]
    const int64_t* __restrict__ eids,
    const int64_t* __restrict__ grid,
    const int32_t* __restrict__ ksigns,
    float* __restrict__ out,             // [E, M_PAD, ROWS]
    int E, int ROWS, int COLS, int STRIDE)
{
    const int e = blockIdx.x;
    const int rowblk = blockIdx.y * 8;   // 8 output rows per warp
    const int eid = (int)eids[e];
    const int lane = threadIdx.x;        // 0..31
    const int l4 = lane % 4, lg = lane / 4;
    const int m0 = lg, k0 = l4 * 4, n0 = lg;

    const uint8_t* wp = packed + (int64_t)eid * ROWS * STRIDE
                       + (int64_t)rowblk * STRIDE;
    const int64_t xbase = (int64_t)e * 16 * COLS;
    const int64_t xsbase = (int64_t)e * 16 * (COLS / 32);

    int32_t c[4] = {0, 0, 0, 0};
    for (int k16 = 0; k16 < COLS / 16; ++k16) {
        const int code_a = k16 * 2;  // even code of this K16
        const int kb = code_a / 32, ci = code_a % 32;
        const uint8_t* blk = wp + (int64_t)kb * 74;
        // d + scale for this K16 (codes code_a/odd share nibble)
        const __half d_h = *(const __half*)blk;
        const float d = __half2float(d_h);
        const uint8_t scv = blk[66 + ci / 4];
        const float nib = (ci % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);

        // A = activation [M16, K16]: lane regs a0,a1 = xq[m, k0..k0+3].
        int32_t a[2];
#pragma unroll
        for (int reg = 0; reg < 2; ++reg) {
            const int tok = m0 + 8 * reg;
            int32_t ab = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                ab |= ((int32_t)xq[xbase + tok * COLS + k16 * 16 + k0 + j] & 0xFF) << (8 * j);
            a[reg] = ab;
        }

        // B = weight [K16, N8]: lane b0 = B[k0..k0+3, n0] decoded int8.
        int32_t b = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = k0 + j;          // 0..15 within K16
            const int code = k16 * 2 + kk / 8;
            const int jj = kk % 8;
            const int kb2 = code / 32, ci2 = code % 32;
            const uint8_t* blk2 = wp + (int64_t)kb2 * 74;
            const uint16_t cd = (uint16_t)(blk2[2 + ci2 * 2] | (blk2[3 + ci2 * 2] << 8));
            const int64_t g = grid[cd & 511];
            const int32_t sb = ksigns[cd >> 9];
            int m = (int)((g >> (8 * jj)) & 0xFF);
            if (sb & (1 << jj)) m = -m;
            b |= (m & 0xFF) << (8 * j);
        }

        mma16(c, a, &b);

        // per-K16 exact scale (same nibble for both codes of this K16);
        // activation scale per (token row, K32 block): row m0 -> xs[m0*K/32 + k16/2]
        const float xs_v = xs[xsbase + m0 * (COLS / 32) + k16 / 2];
        const float sc = d * (0.5f + nib) * 0.25f * xs_v;
        // scale only applies to the rows m0/m0+8 in this C (per-token xscale
        // differs per token; for the proto the M16 group uses one xscale).
        c[0] = (int32_t)(c[0] * sc);
        c[1] = (int32_t)(c[1] * sc);
        c[2] = (int32_t)(c[2] * sc);
        c[3] = (int32_t)(c[3] * sc);
    }
    // proto: dump lane0 accumulator (full reduction is the next step)
    if (lane == 0)
        out[(int64_t)e * 16 * ROWS + rowblk] = (float)c[0];
}
