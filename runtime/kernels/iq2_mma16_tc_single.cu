// Phase 1K: single-output IQ2_K32 GEMM (down).
// Dedicated kernel: computes x @ W^T for ONE packed matrix.  No gate/up dual.
// Same decode+fold as iq2_mma16_tc but single mat, single accumulator set.
#define QSR_EXPORT __attribute__((visibility("default")))
#include <cuda_fp16.h>
#include <cstdint>

#define MAX_MTILES 8
#define KG 32

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
iq2_mma16_tc_single_kernel(
    const int8_t* __restrict__ xq,
    const float* __restrict__ xs,
    const uint8_t* __restrict__ packed,
    const int64_t* __restrict__ eids,
    const int64_t* __restrict__ grid,
    const int32_t* __restrict__ ksigns,
    float* __restrict__ out,
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

    extern __shared__ uint8_t sraw[];
    int8_t* sid = (int8_t*)sraw;                 // [32][256] folded qB
    float* sB_s = (float*)(sraw + 32 * 256);     // [32][8]

    const int64_t xbase = (int64_t)e * M_PAD_C * COLS;
    const int64_t xsbase = (int64_t)e * M_PAD_C * (COLS / 32);

    float facc[M_PAD_C / 16][4] = {};

    const uint8_t* wbase = packed + (int64_t)eid * ROWS * STRIDE + (int64_t)rowbase * STRIDE;

    for (int kb = 0; kb < n_kblocks; ++kb) {
        {
            for (int t = tid; t < 32 * 8; t += 128) {
                const int r = (t / 8) % 32;
                const int grp = t % 8;
                const uint8_t* blk = wbase + (int64_t)kb * 74 + (int64_t)r * STRIDE;
                const float d = __half2float(*(const __half*)blk);
                float mx = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint8_t scv = blk[66 + (grp * 4 + c) / 4];
                    const float nib = ((grp * 4 + c) % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);
                    mx = fmaxf(mx, fabsf(d * (0.5f + nib) * 0.25f));
                }
                const float sB = 43.0f * mx / 127.0f;
                const float inv_sB = (sB > 1e-12f) ? (1.0f / sB) : 0.f;
                sB_s[r * 8 + grp] = sB;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint16_t cd = (uint16_t)(blk[2 + (grp * 4 + c) * 2]
                                                 | (blk[3 + (grp * 4 + c) * 2] << 8));
                    const int64_t g = grid[cd & 511];
                    const int32_t sb = ksigns[cd >> 9];
                    const uint8_t scv = blk[66 + (grp * 4 + c) / 4];
                    const float nib = ((grp * 4 + c) % 4) < 2 ? (float)(scv & 0xF) : (float)(scv >> 4);
                    const float dd = d * (0.5f + nib) * 0.25f;
                    int8_t* dst = &sid[r * 256 + (grp * 4 + c) * 8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        int m = (int)((g >> (8 * j)) & 0xFF);
                        if (sb & (1 << j)) m = -m;
                        dst[j] = (int8_t)lrintf((float)m * dd * inv_sB);
                    }
                }
            }
            __syncthreads();
        }

        const int kb_groups = 8;
#pragma unroll 1
        for (int grp = 0; grp < kb_groups; ++grp) {
            int32_t b[2];
            const int row_lg = warp * 8 + lg;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int k16 = kb * 16 + grp * 2 + h;
                const int pos = (k16 % 16) * 16;
                const int8_t* pg = &sid[row_lg * 256 + pos + l4 * 4];
                b[h] = *(const int32_t*)pg;
            }
            float sB[2];
            sB[0] = sB_s[(warp * 8 + l4 * 2) * 8 + grp];
            sB[1] = sB_s[(warp * 8 + l4 * 2 + 1) * 8 + grp];
#pragma unroll
            for (int mt = 0; mt < n_mtiles; ++mt) {
                const int64_t xb = xbase + (int64_t)mt * 16 * COLS;
                const int64_t xsb = xsbase + (int64_t)mt * 16 * (COLS / 32);
                int32_t a[2];
                const int k16 = kb * 16 + grp * 2;
                a[0] = *(const int32_t*)(xq + xb + (int64_t)lg * COLS + k16 * 16 + k0);
                a[1] = *(const int32_t*)(xq + xb + (int64_t)(lg + 8) * COLS + k16 * 16 + k0);
                int32_t cg[4] = {0, 0, 0, 0};
                mma16(cg, a, &b[0]);
                a[0] = *(const int32_t*)(xq + xb + (int64_t)lg * COLS + (k16 + 1) * 16 + k0);
                a[1] = *(const int32_t*)(xq + xb + (int64_t)(lg + 8) * COLS + (k16 + 1) * 16 + k0);
                mma16(cg, a, &b[1]);
                const float xs0 = xs[xsb + (int64_t)lg * (COLS / 32) + k16 / 2];
                const float xs1 = xs[xsb + (int64_t)(lg + 8) * (COLS / 32) + k16 / 2];
                facc[mt][0] += (float)cg[0] * sB[0] * xs0;
                facc[mt][1] += (float)cg[1] * sB[1] * xs0;
                facc[mt][2] += (float)cg[2] * sB[0] * xs1;
                facc[mt][3] += (float)cg[3] * sB[1] * xs1;
            }
        }
    }

#pragma unroll 1
    for (int mt = 0; mt < n_mtiles; ++mt) {
        const int64_t obase = (int64_t)e * M_PAD_C * ROWS + (int64_t)mt * 16 * ROWS + row0;
        out[obase + (int64_t)lg * ROWS + l4 * 2] = facc[mt][0];
        out[obase + (int64_t)lg * ROWS + l4 * 2 + 1] = facc[mt][1];
        out[obase + (int64_t)(lg + 8) * ROWS + l4 * 2] = facc[mt][2];
        out[obase + (int64_t)(lg + 8) * ROWS + l4 * 2 + 1] = facc[mt][3];
    }
}

extern "C" QSR_EXPORT void iq2_mma16_tc_launch_single(
    const int8_t* xq, const float* xs, const uint8_t* packed,
    const int64_t* eids, const int64_t* grid,
    const int32_t* ksigns, float* out,
    int E, int ROWS, int COLS, int STRIDE, int M_PAD)
{
    const int smem_bytes = 32 * 256 + 32 * 8 * 4;
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_single_kernel<16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_single_kernel<32>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_single_kernel<48>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaFuncSetAttribute((const void*)iq2_mma16_tc_single_kernel<64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dim3 block(128);
    dim3 gridc(E, ROWS / 32);
    if (M_PAD == 16) {
        iq2_mma16_tc_single_kernel<16><<<gridc, block, smem_bytes>>>(
            xq, xs, packed, eids, grid, ksigns, out, E, ROWS, COLS, STRIDE, M_PAD);
    } else if (M_PAD == 32) {
        iq2_mma16_tc_single_kernel<32><<<gridc, block, smem_bytes>>>(
            xq, xs, packed, eids, grid, ksigns, out, E, ROWS, COLS, STRIDE, M_PAD);
    } else if (M_PAD == 48) {
        iq2_mma16_tc_single_kernel<48><<<gridc, block, smem_bytes>>>(
            xq, xs, packed, eids, grid, ksigns, out, E, ROWS, COLS, STRIDE, M_PAD);
    } else {
        iq2_mma16_tc_single_kernel<64><<<gridc, block, smem_bytes>>>(
            xq, xs, packed, eids, grid, ksigns, out, E, ROWS, COLS, STRIDE, M_PAD);
    }
}
