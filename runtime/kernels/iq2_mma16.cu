// Phase 2: IQ2_XS -> INT8 MMA (m16n8k16) grouped MoE, SM120.
// Each block owns one expert x N32 rows x ALL M tokens (M_PAD).  The 74-byte
// IQ2 blocks are staged into smem per 256-value kblock with coalesced loads,
// decoded to int8 once, then the K-loop reads B fragments from smem and runs
// m16n8k16 mma (4-way ILP), accumulating the exact per-K16 fp32 scale
// (d*(0.5+nibble)*0.25 * xs).  grid = (E, ROWS/32), block = 128 (4 warps,
// each warp = 8 rows of the N32 tile).  Numerics verified 32/32 on SM120
// against dequantize_iq2_xs (cos 1.0).
#define QSR_EXPORT __attribute__((visibility("default")))
#include <cuda_fp16.h>
#include <cstdint>

#define MAX_MTILES 8    // M_PAD <= 128

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
    const int n_mtiles = M_PAD / 16;
    const int n_kblocks = COLS / 256;

    // smem: raw blocks for one kblock (2 mats * 32 rows * 74) + decoded int8
    // (2 mats * 32 rows * 256)
    extern __shared__ uint8_t sraw[];
    uint8_t* sg0 = &sraw[0];
    uint8_t* su0 = &sraw[32 * 74];
    int8_t* sid = (int8_t*)(sraw + 2 * 32 * 74);

    const int64_t xbase = (int64_t)e * M_PAD * COLS;
    const int64_t xsbase = (int64_t)e * M_PAD * (COLS / 32);

    float facc_g[MAX_MTILES][4] = {};
    float facc_u[MAX_MTILES][4] = {};

    const uint8_t* wg_base = packed_gate + (int64_t)eid * ROWS * STRIDE + (int64_t)rowbase * STRIDE;
    const uint8_t* wu_base = packed_up + (int64_t)eid * ROWS * STRIDE + (int64_t)rowbase * STRIDE;

    for (int kb = 0; kb < n_kblocks; ++kb) {
        // Stage 32 rows x 74 bytes for gate and up: coalesced.  74*32 = 2368
        // bytes per matrix.  Each lane loads 74 bytes per matrix = 18.5 uint4.
        {
            const uint8_t* wg = wg_base + (int64_t)kb * 74;
            const uint8_t* wu = wu_base + (int64_t)kb * 74;
            // 32 rows * 74 bytes = 2368; 148 uint4.  Lane loads rows in strides.
            // Simple: each lane handles one row's 74 bytes for gate and up.
            // Row r is handled by lane r%32?  Use 32 lanes, each does row=r's
            // 74 bytes: 74/4 = 18 uint4 + 2 bytes.
            const int r = lane;
            const uint8_t* sr_g = wg + (int64_t)r * STRIDE;
            const uint8_t* sr_u = wu + (int64_t)r * STRIDE;
            uint8_t* dg = sg0 + (int64_t)r * 74;
            uint8_t* du = su0 + (int64_t)r * 74;
#pragma unroll
            for (int i = 0; i < 74; ++i) {
                dg[i] = sr_g[i];
                du[i] = sr_u[i];
            }
        }
        __syncthreads();

        // Decode to int8: 2 mats * 32 rows * 32 codes = 2048 codes / 128 = 16
        // codes per thread.  smem: sid[mat][row][code*8 .. +8]
        {
            for (int t = tid; t < 2 * 32 * 32; t += 128) {
                const int mat = t / (32 * 32);
                const int r = (t / 32) % 32;
                const int ci = t % 32;
                const uint8_t* blk = (mat == 0 ? sg0 : su0) + (int64_t)r * 74;
                const uint16_t cd = (uint16_t)(blk[2 + ci * 2] | (blk[3 + ci * 2] << 8));
                const int64_t g = grid[cd & 511];
                const int32_t sb = ksigns[cd >> 9];
                int8_t* dst = &sid[(mat * 32 + r) * 256 + ci * 8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    int m = (int)((g >> (8 * j)) & 0xFF);
                    if (sb & (1 << j)) m = -m;
                    dst[j] = (int8_t)m;
                }
            }
            __syncthreads();
        }

        // 16 K16 partials in this kblock, processed 4 at a time for ILP.
#pragma unroll 1
        for (int u0 = 0; u0 < 16; u0 += 4) {
            // preload B and scales for 4 K16
            int32_t b_g[4], b_u[4];
            float sg[4][2], su[4][2];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int k16 = kb * 16 + u0 + q;
                // B fragment: lane (lg,l4) needs sid[mat][lg][pos + l4*4 + 0..3]
                // where pos is the within-block K16 position ((k16%16)*16).
                // -- 4 consecutive bytes, load as one int32 from smem.
                const int pos = (k16 % 16) * 16;
                const int row_lg = warp * 8 + lg;
                const int8_t* pg = &sid[(0 * 32 + row_lg) * 256 + pos + l4 * 4];
                const int8_t* pu = &sid[(1 * 32 + row_lg) * 256 + pos + l4 * 4];
                b_g[q] = *(const int32_t*)pg;
                b_u[q] = *(const int32_t*)pu;
                const int cii = (k16 * 2) % 32;
#pragma unroll
                for (int col = 0; col < 2; ++col) {
                    const int nrow = warp * 8 + l4 * 2 + col;
                    {
                        const uint8_t* blk_g = sg0 + (int64_t)nrow * 74;
                        const __half d_g = *(const __half*)blk_g;
                        const uint8_t scv_g = blk_g[66 + cii / 4];
                        const float nib_g = (cii % 4) < 2 ? (float)(scv_g & 0xF) : (float)(scv_g >> 4);
                        sg[q][col] = __half2float(d_g) * (0.5f + nib_g) * 0.25f;
                    }
                    {
                        const uint8_t* blk_u = su0 + (int64_t)nrow * 74;                        const __half d_u = *(const __half*)blk_u;
                        const uint8_t scv_u = blk_u[66 + cii / 4];
                        const float nib_u = (cii % 4) < 2 ? (float)(scv_u & 0xF) : (float)(scv_u >> 4);
                        su[q][col] = __half2float(d_u) * (0.5f + nib_u) * 0.25f;
                    }
                }
            }
            // issue all 4 mma pairs, then convert
            int32_t cg0[4], cu0[4], cg1[4], cu1[4], cg2[4], cu2[4], cg3[4], cu3[4];
            int32_t a0[2], a1[2], a2[2], a3[2];
#pragma unroll 1
            for (int mt = 0; mt < n_mtiles; ++mt) {
                const int64_t xb = xbase + (int64_t)mt * 16 * COLS;
                const int64_t xsb = xsbase + (int64_t)mt * 16 * (COLS / 32);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int k16 = kb * 16 + u0 + q;
                    a0[0] = *(const int32_t*)(xq + xb + (int64_t)lg * COLS + k16 * 16 + k0);
                    a0[1] = *(const int32_t*)(xq + xb + (int64_t)(lg + 8) * COLS + k16 * 16 + k0);
                    int32_t* cg = q == 0 ? cg0 : q == 1 ? cg1 : q == 2 ? cg2 : cg3;
                    int32_t* cu = q == 0 ? cu0 : q == 1 ? cu1 : q == 2 ? cu2 : cu3;
                    int32_t t[4] = {0,0,0,0};
                    int32_t tu[4] = {0,0,0,0};
                    mma16(t, a0, &b_g[q]);
                    mma16(tu, a0, &b_u[q]);
                    for (int r = 0; r < 4; ++r) { cg[r] = t[r]; cu[r] = tu[r]; }
                }
                const float xs0 = xs[xsb + (int64_t)lg * (COLS / 32) + (kb * 16 + u0) / 2];
                const float xs1 = xs[xsb + (int64_t)(lg + 8) * (COLS / 32) + (kb * 16 + u0) / 2];
                const float xs0b = xs[xsb + (int64_t)lg * (COLS / 32) + (kb * 16 + u0 + 2) / 2];
                const float xs1b = xs[xsb + (int64_t)(lg + 8) * (COLS / 32) + (kb * 16 + u0 + 2) / 2];
                // convert+scale: c0/c2 -> (col 0, xs0/xs1), c1/c3 -> (col 1, xs0/xs1)
                facc_g[mt][0] += (float)cg0[0]*sg[0][0]*xs0 + (float)cg1[0]*sg[1][0]*xs0
                               + (float)cg2[0]*sg[2][0]*xs0b + (float)cg3[0]*sg[3][0]*xs0b;
                facc_g[mt][1] += (float)cg0[1]*sg[0][1]*xs0 + (float)cg1[1]*sg[1][1]*xs0
                               + (float)cg2[1]*sg[2][1]*xs0b + (float)cg3[1]*sg[3][1]*xs0b;
                facc_g[mt][2] += (float)cg0[2]*sg[0][0]*xs1 + (float)cg1[2]*sg[1][0]*xs1
                               + (float)cg2[2]*sg[2][0]*xs1b + (float)cg3[2]*sg[3][0]*xs1b;
                facc_g[mt][3] += (float)cg0[3]*sg[0][1]*xs1 + (float)cg1[3]*sg[1][1]*xs1
                               + (float)cg2[3]*sg[2][1]*xs1b + (float)cg3[3]*sg[3][1]*xs1b;
                facc_u[mt][0] += (float)cu0[0]*su[0][0]*xs0 + (float)cu1[0]*su[1][0]*xs0
                               + (float)cu2[0]*su[2][0]*xs0b + (float)cu3[0]*su[3][0]*xs0b;
                facc_u[mt][1] += (float)cu0[1]*su[0][1]*xs0 + (float)cu1[1]*su[1][1]*xs0
                               + (float)cu2[1]*su[2][1]*xs0b + (float)cu3[1]*su[3][1]*xs0b;
                facc_u[mt][2] += (float)cu0[2]*su[0][0]*xs1 + (float)cu1[2]*su[1][0]*xs1
                               + (float)cu2[2]*su[2][0]*xs1b + (float)cu3[2]*su[3][0]*xs1b;
                facc_u[mt][3] += (float)cu0[3]*su[0][1]*xs1 + (float)cu1[3]*su[1][1]*xs1
                               + (float)cu2[3]*su[2][1]*xs1b + (float)cu3[3]*su[3][1]*xs1b;
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

extern "C" QSR_EXPORT void iq2_mma16_launch(
    const int8_t* xq, const float* xs, const uint8_t* packed_gate,
    const uint8_t* packed_up, const int64_t* eids, const int64_t* grid,
    const int32_t* ksigns, float* out_gate, float* out_up,
    int E, int ROWS, int COLS, int STRIDE, int M_PAD)
{
    const int smem_bytes = 2 * 32 * 74 + 2 * 32 * 256;  // 4736 + 16384 = 21120
    cudaFuncSetAttribute(iq2_mma16_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dim3 block(128);
    dim3 gridc(E, ROWS / 32);
    iq2_mma16_kernel<<<gridc, block, smem_bytes>>>(
        xq, xs, packed_gate, packed_up, eids, grid, ksigns,
        out_gate, out_up, E, ROWS, COLS, STRIDE, M_PAD);
}
