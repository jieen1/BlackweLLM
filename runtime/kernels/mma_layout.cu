// Verify m16n8k16 int8 MMA fragment layout: known A/B -> dump C.
#include <cstdint>
#include <cuda_runtime.h>

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

// A[i,j] = i*16+j (int8), B[k,n] = k*8+n. C = A@B. Dump C[16,8].
extern "C" __global__ void __launch_bounds__(32)
mma_layout_kernel(int32_t* out /* [16,8] */) {
    const int lane = threadIdx.x;
    const int l4 = lane % 4, lg = lane / 4;
    // a0 -> A[lg, l4*4 + 0..3], a1 -> A[lg+8, l4*4 + 0..3]
    int32_t a[2] = {0x01010101, 0x01010101};  // A all ones -> C[m,n] = sum_k B[k,n]
    // hypothesis: b0 -> B[k = l4*4 + 0..3, n = lg]
    int32_t b = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        b |= (int32_t)(int8_t)((l4 * 4 + j) * 8 + lg) << (8 * j);
    int32_t c[4] = {0, 0, 0, 0};
    mma16(c, a, &b);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        // map c[r] to (m, n): assume c0=(lg,lg'), c1=(lg+8,...), c2/c3 n+4
        // store all four regs with a lane tag; infer mapping from values.
        out[((int64_t)lane * 4 + r) * 8 + 0] = c[r];
    }
}
extern "C" void mma_layout_launch(int32_t* out) {
    mma_layout_kernel<<<1, 32>>>(out);
}
