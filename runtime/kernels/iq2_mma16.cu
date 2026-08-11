// Phase 2: IQ2_XS in-kernel decode -> INT8 MMA (m16n8k16) grouped MoE, SM120.
// Per K16 partial (2 codes, one nibble value: lo for codes 0-1, hi for 2-3)
// we run mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 and apply the
// exact per-partial scale (d * (0.5+nibble) * 0.25 * activation_scale) in
// FP32 epilogue, so IQ2 semantics are preserved without a resident W8A8.
//
// A fragment layout (m16n8k16, row-major A [16,16] int8, 32 lanes):
//   lane l holds 4 .b32 regs a0..a3; each reg = 4 int8 along K.
//   a_reg[2*(l%4)+0/1] rows m = l/4 + 0/8; k group = (l/4... ) -- see note.
// We decode the two K16 codes and pack into the a_reg bytes.
#include <cuda_fp16.h>
#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define MMA16(a, b, c) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) \
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#else
#define MMA16(a, b, c)
#endif

__global__ void __launch_bounds__(256)
iq2_mma16_kernel(
    const int8_t* __restrict__ xq,       // [E, M_PAD, K] int8 activations
    const float* __restrict__ xs,        // [E, M_PAD, K/32]
    const uint8_t* __restrict__ packed,  // [256, ROWS*STRIDE]
    const int64_t* __restrict__ eids,
    const int64_t* __restrict__ grid,    // [512]
    const int32_t* __restrict__ ksigns,  // [128]
    float* __restrict__ out,             // [E, M_PAD, ROWS]
    int E, int ROWS, int COLS, int STRIDE, int M_PAD)
{
    const int e = blockIdx.x;
    const int eid = (int)eids[e];
    const int row = blockIdx.y * 16 + threadIdx.y * 8;  // N8 rows per warp-group
    const int lane = threadIdx.x;  // 0..31
    const int warp = threadIdx.y;  // 0..3 (4 warps cover 32 output rows / CTA tile)

    // xq row for this expert/token group
    // M16 = token rows; one warp handles M16 tokens x N8 output rows.
    // For the proto: M_PAD = 16, one warp per (expert, 8 output rows).
    int32_t c[4] = {0, 0, 0, 0};
    const int m_off = lane / 4;        // A row group
    const int k_off = (lane % 4) * 4;  // K byte offset within 16
    const int k16_lo = (lane / 4) < 8;  // placeholder

    for (int k16 = 0; k16 < COLS / 16; ++k16) {
        // Two codes (k16 covers 2 codes = 16 values). code index = k16 (within row, flattened).
        // code c = k16 * 2 + half
        const int kb = (k16 * 2) / 32;        // 256-block index
        const int cbase = (k16 * 2) % 32;     // first code in block
        const uint8_t* blk = packed + (int64_t)eid * ROWS * STRIDE
                           + (int64_t)row * STRIDE + kb * 74;
        const int16_t d_h = *(const __half*)blk;
        // A fragment: decode the two K16 codes' int8 magnitudes for this lane's A bytes.
        int32_t a[4];
        for (int reg = 0; reg < 4; ++reg) {
            // A[m, k]: m = m_off + 8*(reg&1), k = k_off + 4*((reg&2)>>1)?  -- placeholder layout
            a[reg] = 0;
        }
        int32_t b[2] = {0, 0};
        MMA16(a, b, c);
        (void)d_h; (void)k16_lo;
    }
    // (proto shell: layouts TBD; see notes)
    out[e * M_PAD * ROWS + 0] = (float)c[0];
}
