// Phase 2B-0: IQ2_XS scale-amortized INT8 MMA (m16n8k16) grouped MoE, SM120.
// K-scale group = 32 (2 K16).  The per-K16 IQ2 delta is folded into the INT8
// B fragment (qB = sign * round(mag * delta_j / sB)); activations use the
// existing per-32 xs as sA.  INT32 mma accumulates both K16 of the group, and
// the I2F+FFMA float scale happens ONCE per K-group per accumulator.
//
//   delta_j = d * (0.5 + nibble_j) * 0.25     (per K16 = 2 codes)
//   sA = xs (per 32 activation scale, prequantized)
//   sB = 43 * max_j(|delta_j| over K-group) / 127
//   qB = sign * round(mag * delta_j / sB)     (int8, folded at decode)
//   acc32 = sum_{K-group}(qA * qB)            (INT32, 2 x m16n8k16 mma)
//   partial = float(acc32) * sA * sB          (ONE I2F + FFMA per group)
//
// grid = (E, ROWS/32), block = 128 (4 warps, each 8 rows of N32).  Numerics
// verified in tools/prescreen_iq2_kgroup_fold.py: K32 gate cos >= 0.9999.
#define QSR_EXPORT __attribute__((visibility("default")))
#include <cuda_fp16.h>
#include <cstdint>

#define MAX_MTILES 8    // M_PAD <= 128


#define KG 32           // K-scale group (2 K16)

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

template <int M_PAD_C>
__global__ void __launch_bounds__(128)
iq2_mma16_tc_kernel(
    const int8_t* __restrict__ xq,
    const float* __restrict__ xs,
    const uint8_t* __restrict__ packed_gate,
    const uint8_t* __restrict__ packed_up,
    const int64_t* __restrict__ eids,
    const int64_t* __restrict__ grid,
    const int32_t* __restrict__ ksigns,
    float* __restrict__ out_gate,
    float* __restrict__ out_up,
    int E, int ROWS, int COLS, int STRIDE, int M_PAD)
{
    const int e = blockIdx.x;
    const int rowbase = blockIdx.y * 32;
    const int eid = (int)eids[e];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = rowbase + warp * 8;
    const int l4 = lane % 4, lg = lane / 4;
    const int k0 = l4 * 4;
    const int n_mtiles = M_PAD_C / 16;
    const int n_kblocks = COLS / 256;
    const int n_groups = COLS / KG;          // K-groups per row

    extern __shared__ uint8_t sraw[];
    int8_t* sid = (int8_t*)sraw;  // [2][32][256] folded qB
    float* sB_g_s = (float*)(sraw + 2 * 32 * 256);   // [32][8] per (row,grp)
    float* sB_u_s = sB_g_s + 32 * 8;

    const int64_t xbase = (int64_t)e * M_PAD * COLS;
    const int64_t xsbase = (int64_t)e * M_PAD * (COLS / 32);

    float facc_g[M_PAD_C / 16][4] = {};
    float facc_u[M_PAD_C / 16][4] = {};

    const uint8_t* wg_base = packed_gate + (int64_t)eid * ROWS * STRIDE + (int64_t)rowbase * STRIDE;
    const uint8_t* wu_base = packed_up + (int64_t)eid * ROWS * STRIDE + (int64_t)rowbase * STRIDE;

    for (int kb = 0; kb < n_kblocks; ++kb) {
        // Fused decode+fold reads weights directly from global (no raw staging).
        // Fused decode+fold: per (mat,row,group) thread decodes 4 codes and
        // folds delta into qB in one pass.  2 mats*32 rows*8 groups = 512.
        {
            for (int t = tid; t < 2 * 32 * 8; t += 128) {
                const int mat = t / (32 * 8);
                const int r = (t / 8) % 32;
                const int grp = t % 8;
                const uint8_t* blk = (mat == 0 ? wg_base : wu_base)
                                    + (int64_t)kb * 74 + (int64_t)r * STRIDE;
                const __half d_h = *(const __half*)blk;
                const float d = __half2float(d_h);
                // max |delta| over the 4 codes
                float mx = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint8_t scv = blk[66 + (grp * 4 + c) / 4];
                    const float nib = ((grp * 4 + c) % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);
                    mx = fmaxf(mx, fabsf(d * (0.5f + nib) * 0.25f));
                }
                const float sB = 43.0f * mx / 127.0f;
                const float inv_sB = (sB > 1e-12f) ? (1.0f / sB) : 0.f;
                if (mat == 0) sB_g_s[r * 8 + grp] = sB;
                else          sB_u_s[r * 8 + grp] = sB;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint16_t cd = (uint16_t)(blk[2 + (grp * 4 + c) * 2]
                                                 | (blk[3 + (grp * 4 + c) * 2] << 8));
                    const int64_t g = grid[cd & 511];
                    const int32_t sb = ksigns[cd >> 9];
                    const uint8_t scv = blk[66 + (grp * 4 + c) / 4];
                    const float nib = ((grp * 4 + c) % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);
                    const float dd = d * (0.5f + nib) * 0.25f;
                    int8_t* dst = &sid[(mat * 32 + r) * 256 + (grp * 4 + c) * 8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        int m = (int)((g >> (8 * j)) & 0xFF);
                        if (sb & (1 << j)) m = -m;
                        const float v = (float)m * dd * inv_sB;
                        dst[j] = (int8_t)lrintf(v);
                    }
                }
            }
            __syncthreads();
        }

                // ---- mma: K-group = 2 K16, INT32 accumulate, ONE scale per group ----
        const int kb_groups = 8;   // 256 values / 32
#pragma unroll 1
        for (int grp = 0; grp < kb_groups; ++grp) {
            // B fragments for the 2 K16 of this group (folded qB in smem)
            int32_t b_g[2], b_u[2];
            const int row_lg = warp * 8 + lg;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k16 = kb * 16 + grp * 2 + h;   // global K16
                const int pos = (k16 % 16) * 16;          // within-block position
                const int8_t* pg = &sid[(0 * 32 + row_lg) * 256 + pos + l4 * 4];
                const int8_t* pu = &sid[(1 * 32 + row_lg) * 256 + pos + l4 * 4];
                b_g[h] = *(const int32_t*)pg;
                b_u[h] = *(const int32_t*)pu;
            }
            // sB for this group (per mat,row,group) from the fold pass: reuse
            // the same max-delta -> sB recompute here is cheap (4 codes).
            // sB per C-output row: c0/c2 -> B column l4*2, c1/c3 -> column l4*2+1.
            // Each B column is a weight row (row_lg for c0/c2, row_lg+1 for c1/c3).
            float sB_g[2], sB_u[2];
            sB_g[0] = sB_g_s[(warp * 8 + l4 * 2) * 8 + grp];
            sB_g[1] = sB_g_s[(warp * 8 + l4 * 2 + 1) * 8 + grp];
            sB_u[0] = sB_u_s[(warp * 8 + l4 * 2) * 8 + grp];
            sB_u[1] = sB_u_s[(warp * 8 + l4 * 2 + 1) * 8 + grp];
            // activation per token (c0/c2 -> token lg, c1/c3 -> token lg+8)
#pragma unroll
            for (int mt = 0; mt < n_mtiles; ++mt) {
                const int64_t xb = xbase + (int64_t)mt * 16 * COLS;
                const int64_t xsb = xsbase + (int64_t)mt * 16 * (COLS / 32);
                int32_t a[2];
                const int k16 = kb * 16 + grp * 2;   // A at group start
                a[0] = *(const int32_t*)(xq + xb + (int64_t)lg * COLS + k16 * 16 + k0);
                a[1] = *(const int32_t*)(xq + xb + (int64_t)(lg + 8) * COLS + k16 * 16 + k0);
                int32_t cg[4] = {0, 0, 0, 0};
                int32_t cu[4] = {0, 0, 0, 0};
                // K16 lo
                mma16(cg, a, &b_g[0]);
                mma16(cu, a, &b_u[0]);
                // K16 hi (different A columns, same tokens)
                a[0] = *(const int32_t*)(xq + xb + (int64_t)lg * COLS + (k16 + 1) * 16 + k0);
                a[1] = *(const int32_t*)(xq + xb + (int64_t)(lg + 8) * COLS + (k16 + 1) * 16 + k0);
                mma16(cg, a, &b_g[1]);
                mma16(cu, a, &b_u[1]);
                // ONE scale per K-group per accumulator.
                const float xs0 = xs[xsb + (int64_t)lg * (COLS / 32) + k16 / 2];
                const float xs1 = xs[xsb + (int64_t)(lg + 8) * (COLS / 32) + k16 / 2];
                facc_g[mt][0] += (float)cg[0] * sB_g[0] * xs0;
                facc_g[mt][1] += (float)cg[1] * sB_g[1] * xs0;
                facc_g[mt][2] += (float)cg[2] * sB_g[0] * xs1;
                facc_g[mt][3] += (float)cg[3] * sB_g[1] * xs1;
                facc_u[mt][0] += (float)cu[0] * sB_u[0] * xs0;
                facc_u[mt][1] += (float)cu[1] * sB_u[1] * xs0;
                facc_u[mt][2] += (float)cu[2] * sB_u[0] * xs1;
                facc_u[mt][3] += (float)cu[3] * sB_u[1] * xs1;
            }
        }
    }

#pragma unroll 1
    for (int mt = 0; mt < n_mtiles; ++mt) {
        const int64_t obase = (int64_t)e * M_PAD * ROWS + (int64_t)mt * 16 * ROWS + row0;
        out_gate[obase + (int64_t)lg * ROWS + l4 * 2] = facc_g[mt][0];
        out_gate[obase + (int64_t)lg * ROWS + l4 * 2 + 1] = facc_g[mt][1];
        out_gate[obase + (int64_t)(lg + 8) * ROWS + l4 * 2] = facc_g[mt][2];
        out_gate[obase + (int64_t)(lg + 8) * ROWS + l4 * 2 + 1] = facc_g[mt][3];
        out_up[obase + (int64_t)lg * ROWS + l4 * 2] = facc_u[mt][0];
        out_up[obase + (int64_t)lg * ROWS + l4 * 2 + 1] = facc_u[mt][1];
        out_up[obase + (int64_t)(lg + 8) * ROWS + l4 * 2] = facc_u[mt][2];
        out_up[obase + (int64_t)(lg + 8) * ROWS + l4 * 2 + 1] = facc_u[mt][3];
    }
}

extern "C" QSR_EXPORT void iq2_mma16_tc_launch(
    const int8_t* xq, const float* xs, const uint8_t* packed_gate,
    const uint8_t* packed_up, const int64_t* eids, const int64_t* grid,
    const int32_t* ksigns, float* out_gate, float* out_up,
    int E, int ROWS, int COLS, int STRIDE, int M_PAD)
{
    const int smem_bytes = 2 * 32 * 256 + 2 * 32 * 8 * 4;        // sid + sB planes
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_kernel<16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_kernel<32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_kernel<48>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_kernel<64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dim3 block(128);
    dim3 gridc(E, ROWS / 32);
    if (M_PAD == 16) {
        iq2_mma16_tc_kernel<16><<<gridc, block, smem_bytes>>>(
            xq, xs, packed_gate, packed_up, eids, grid, ksigns,
            out_gate, out_up, E, ROWS, COLS, STRIDE, M_PAD);
    } else if (M_PAD == 32) {
        iq2_mma16_tc_kernel<32><<<gridc, block, smem_bytes>>>(
            xq, xs, packed_gate, packed_up, eids, grid, ksigns,
            out_gate, out_up, E, ROWS, COLS, STRIDE, M_PAD);
    } else if (M_PAD == 48) {
        iq2_mma16_tc_kernel<48><<<gridc, block, smem_bytes>>>(
            xq, xs, packed_gate, packed_up, eids, grid, ksigns,
            out_gate, out_up, E, ROWS, COLS, STRIDE, M_PAD);
    } else {
        iq2_mma16_tc_kernel<64><<<gridc, block, smem_bytes>>>(
            xq, xs, packed_gate, packed_up, eids, grid, ksigns,
            out_gate, out_up, E, ROWS, COLS, STRIDE, M_PAD);
    }
}
