// Phase 2: IQ2_XS in-kernel decode -> INT8 MMA (m16n8k16) grouped MoE, SM120.
// Exact IQ2: two codes of a K16 partial share one nibble (even->lo, odd->hi)
// so per-K16 scale = d*(0.5+nibble)*0.25*actscale is exact.
// Fragment layouts (m16n8k16 s8, 32 lanes) -- PTX-ISA documented, verified
// 32/32 on SM120 with a full-warp A/B probe (2026-08-11; earlier "hardware-
// hardcoded B" conclusion was a probe sign-extension artifact, see notes):
//   A row-major [16,16]: a0=A[lg, l4*4+0..3], a1=A[lg+8, l4*4+0..3]
//   B col-major [16,8]:  b0 byte j = B[l4*4+j, lg]
//   C [16,8] s32: c0=C[lg,l4*2], c1=C[lg,l4*2+1], c2=C[lg+8,l4*2], c3=C[lg+8,l4*2+1]
// One warp: M16 tokens x N8 rows of one expert (blockIdx.y = row tile/8).
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
    int32_t* __restrict__ dump_buf,      // [2] debug
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
    const int rowblk = (blockIdx.y * 4 + threadIdx.y) * 8;
    const int eid = (int)eids[e];
    const int lane = threadIdx.x;
    const int l4 = lane % 4, lg = lane / 4;
    const int k0 = l4 * 4;   // k index within the K16 partial

    const int64_t xbase = (int64_t)e * 16 * COLS;
    const int64_t xsbase = (int64_t)e * 16 * (COLS / 32);

    float facc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k16 = 0; k16 < COLS / 16; ++k16) {
        // Per-K16 scale of B column n = weight row (rowblk+n)'s d/nibble.
        // c0/c2 accumulate over B column l4*2 ; c1/c3 over column l4*2+1,
        // so the two halves use different weight rows' scales.
        float sc[2];
#pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int nrow = l4 * 2 + col;
            const int code_a = k16 * 2;
            const int kb = code_a / 32, ci = code_a % 32;
            const uint8_t* blk = packed + (int64_t)eid * ROWS * STRIDE
                               + (int64_t)(rowblk + nrow) * STRIDE + (int64_t)kb * 74;
            const __half d_h = *(const __half*)blk;
            const float d = __half2float(d_h);
            const uint8_t scv = blk[66 + ci / 4];
            const float nib = (ci % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);
            sc[col] = d * (0.5f + nib) * 0.25f;
        }

        // A = activation [M16, K16]: a0/a1 = xq[token, k0+0..3].
        int32_t a[2];
#pragma unroll
        for (int reg = 0; reg < 2; ++reg) {
            const int tok = lg + 8 * reg;
            int32_t ab = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                ab |= ((int32_t)xq[xbase + (int64_t)tok * COLS + k16 * 16 + k0 + j] & 0xFF) << (8 * j);
            a[reg] = ab;
        }

        // B = weight [K16, N8]: b0 byte j = B[l4*4+j, lg] (PTX: col=groupID).
        int32_t b = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ncol = lg;
            const int kk = k0 + j;
            const int code = k16 * 2 + kk / 8;
            const int jj = kk % 8;
            const int kb2 = code / 32, ci2 = code % 32;
            const uint8_t* blk2 = packed + (int64_t)eid * ROWS * STRIDE
                                + (int64_t)(rowblk + ncol) * STRIDE + kb2 * 74;
            const uint16_t cd = (uint16_t)(blk2[2 + ci2 * 2] | (blk2[3 + ci2 * 2] << 8));
            const int64_t g = grid[cd & 511];
            const int32_t sb = ksigns[cd >> 9];
            int m = (int)((g >> (8 * jj)) & 0xFF);
            if (sb & (1 << jj)) m = -m;
            b |= (m & 0xFF) << (8 * j);
        }

        int32_t c[4] = {0, 0, 0, 0};
        mma16(c, a, &b);
        if (lane == 0 && k16 == 0) {
            dump_buf[0] = b;
            dump_buf[1] = c[0];
        }

        const float xs0 = xs[xsbase + (int64_t)lg * (COLS / 32) + k16 / 2];
        const float xs1 = xs[xsbase + (int64_t)(lg + 8) * (COLS / 32) + k16 / 2];
        facc[0] += (float)c[0] * sc[0] * xs0;
        facc[1] += (float)c[1] * sc[1] * xs0;
        facc[2] += (float)c[2] * sc[0] * xs1;
        facc[3] += (float)c[3] * sc[1] * xs1;
    }
    const int64_t obase = (int64_t)e * 16 * ROWS + rowblk;
    out[obase + (int64_t)lg * ROWS + l4 * 2] = facc[0];
    out[obase + (int64_t)lg * ROWS + l4 * 2 + 1] = facc[1];
    out[obase + (int64_t)(lg + 8) * ROWS + l4 * 2] = facc[2];
    out[obase + (int64_t)(lg + 8) * ROWS + l4 * 2 + 1] = facc[3];
}

extern "C" void iq2_mma16_launch(
    int32_t* dump_buf,
    const int8_t* xq, const float* xs, const uint8_t* packed,
    const int64_t* eids, const int64_t* grid, const int32_t* ksigns,
    float* out, int E, int ROWS, int COLS, int STRIDE)
{
    dim3 block(32, 4);                    // 32 lanes x 4 warps = 128 threads
    dim3 gridc(E, (ROWS + 31) / 32);      // each warp covers 8 rows; block covers 32
    iq2_mma16_kernel<<<gridc, block>>>(
        dump_buf, xq, xs, packed, eids, grid, ksigns, out, E, ROWS, COLS, STRIDE);
}
