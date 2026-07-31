# SM120 NVFP4 GEMM Research — CUTLASS 4.6.1 Dispatch Path Analysis

**Date:** 2026-07-31
**Scope:** How to replace SM80 CUTLASS BF16 GEMM kernels with SM120-native kernels
**Sources:** cutlass-4.6.1 source tree, qwen-sm120-runtime model/backends code

---

## 1. Executive Summary

The 794 SM80 CUTLASS kernel calls (12.9% GPU time) are **not a software
misconfiguration**. SM120 hardware physically lacks BF16/FP16 tensor core MMA
instructions. cuBLAS correctly falls back to SM80-compatible `mma.sync.m16n8k16.bf16`
because that is the only BF16 MMA path available on this GPU.

SM120's native tensor core paths are **FP8 (f8f6f4)** and **NVFP4 (mxf4nvf4
blockscaled)** only. To eliminate SM80 kernels, the dense BF16 projections must
be quantized. The MoE routed experts already use SM120-native sparkinfer NVFP4
kernels and are not part of this problem.

---

## 2. SM120 vs SM100 Architecture

SM120 (Blackwell consumer/RTX) and SM100 (Blackwell datacenter) share the
"Blackwell" marketing name but have fundamentally different MMA subsystems:

| Feature | SM100 (datacenter) | SM120 (consumer) |
|---|---|---|
| MMA instruction class | `tcgen05.mma` | `mma.sync.aligned` |
| Accumulator storage | Tensor Memory (TMEM) | Register file |
| BF16/FP16 tensor core | ✅ Yes | ❌ **No** |
| FP8 (e4m3/e5m2) MMA | ✅ `tcgen05.mma` | ✅ `mma.sync.kind::f8f6f4.m16n8k32` |
| FP6 (e3m2/e2m3) MMA | ✅ | ✅ `mma.sync.kind::f8f6f4.m16n8k32` |
| FP4 (e2m1) MMA | ✅ | ✅ `mma.sync.kind::f8f6f4.m16n8k32` |
| NVFP4 blockscaled MMA | ✅ `tcgen05.mma.blockscaled` | ✅ `mma.sync.kind::mxf4nvf4.m16n8k64` |
| MX blockscaled MMA | ✅ | ✅ `mma.sync.kind::mxf8f6f4.m16n8k32` |
| Cluster multicast | ✅ Yes | ❌ No (cluster always 1×1×1) |
| SMEM capacity | 228 KB | **101,376 bytes** (~99 KB) |
| Warp-specialized design | ✅ (TMEM-decoupled) | ✅ (register-based, TMA+pipeline) |

**Key evidence:**
- `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` line ~97:
  ```cpp
  static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() &&
                detail::is_sm10x_f8f6f4_element<ElementB>(),
                "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA.");
  ```
- `include/cute/arch/mma_sm120.hpp`: All `SM120_16x8x32_TN` specializations
  cover only `{e2m1, e3m2, e2m3, e4m3, e5m2}²` — no `bfloat16_t` or `half_t`.
- `include/cutlass/arch/arch.h` line 45: `sm120_smem_capacity_bytes = 101376`

---

## 3. The Exact Dispatch Path to SM80 Kernels

### 3.1 Call Chain

```
LagunaForCausalLMSelfBuilt.forward()
  → LagunaModelSelfBuilt.forward()
    → LagunaDecoderLayerSelfBuilt.forward()  ×48 layers
      → input_layernorm (TritonRMSNorm)
      → LagunaAttentionSelfBuilt.forward()
        → self.qkv_proj(hidden_states)       ← PlainLinear → F.linear
        → self.q_norm / self.k_norm          ← TritonRMSNorm
        → self.rotary_emb(positions, q, k)   ← custom kernel
        → self.attn(q, k, v)                 ← sparkinfer attention
        → self.g_proj(hidden_states)         ← PlainLinear → F.linear
        → self.o_proj(attn_output)           ← PlainLinear → F.linear
      → post_attention_layernorm (TritonRMSNorm)
      → LagunaMLPSelfBuilt.forward()         ← layer 0 only
        → self.gate_proj(x)                  ← PlainLinear → F.linear
        → self.up_proj(x)                    ← PlainLinear → F.linear
        → silu(gate) * up
        → self.down_proj(x)                  ← PlainLinear → F.linear
      → LagunaMoESelfBuilt.forward()         ← layers 1-47
        → self.gate(hidden_states)           ← PlainLinear → F.linear (3072→256)
        → self.shared_expert.forward()       ← PlainLinear ×3
        → sparkinfer_moe_fp4(...)            ← SM120-native NVFP4 ✅
  → PlainLMHead → F.linear                   ← (3072→100352)
```

### 3.2 Root Cause

```python
# runtime/model/plain_linear.py, line 120
class PlainLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

`F.linear` with BF16 tensors dispatches through:
```
torch._C._nn.linear()
  → at::native::linear()
    → at::native::mm() / at::native::addmm()
      → cuBLAS BF16 GEMM
        → sm80_xmma_gemm_bf16bf16_* (SM80 CUTLASS kernel)
```

cuBLAS on SM120 has no BF16 tensor core path because the hardware lacks it.
It falls back to SM80-compatible `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

### 3.3 Affected GEMM Dimensions

From `tests/fixtures/laguna_configs/target/config.json`:

| Layer | GEMM | M (batch) | N | K | Calls/step |
|---|---|---|---|---|---|
| Attention (full, 12 layers) | qkv_proj | 1-4 | 8192 | 3072 | 12 |
| Attention (sliding, 36 layers) | qkv_proj | 1-4 | 11264 | 3072 | 36 |
| Attention (full) | o_proj | 1-4 | 3072 | 6144 | 12 |
| Attention (sliding) | o_proj | 1-4 | 3072 | 9216 | 36 |
| Attention | g_proj | 1-4 | 3072 | 3072 | 48 |
| Layer 0 MLP | gate_proj | 1-4 | 12288 | 3072 | 1 |
| Layer 0 MLP | up_proj | 1-4 | 12288 | 3072 | 1 |
| Layer 0 MLP | down_proj | 1-4 | 3072 | 12288 | 1 |
| MoE shared expert | gate_proj | 1-4 | 1024 | 3072 | 47 |
| MoE shared expert | up_proj | 1-4 | 1024 | 3072 | 47 |
| MoE shared expert | down_proj | 1-4 | 3072 | 1024 | 47 |
| MoE router | gate | 1-4 | 256 | 3072 | 47 |
| LM head | linear | 1-4 | 100352 | 3072 | 1 |

**Total: ~376 GEMM calls per forward pass** (×2 for MTP draft+verify ≈ 752,
close to the observed 794).

### 3.4 What's Already SM120-Native

- **MoE routed experts:** sparkinfer `sparkinfer_moe_fp4` — NVFP4, ~38μs/layer
- **Attention:** sparkinfer paged attention — FP8 KV cache
- **RMSNorm:** Triton kernel
- **RoPE:** custom kernel

---

## 4. SM120 GEMM Collective Builder API (CUTLASS 4.6.1)

### 4.1 Dense FP8 GEMM (Non-Blockscaled)

**Builder:** `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl`

```cpp
// Template signature
CollectiveBuilder<
    arch::Sm120,                    // ArchTag
    arch::OpClassTensorOp,          // OperatorClass
    ElementA, LayoutA, AlignmentA,  // A: float_e4m3_t, RowMajor, 16
    ElementB, LayoutB, AlignmentB,  // B: float_e4m3_t, ColumnMajor, 16
    ElementAccumulator,             // float
    TileShape_MNK,                  // e.g. Shape<_128,_64,_64>
    ClusterShape_MNK,               // MUST be Shape<_1,_1,_1>
    StageCountType,                 // StageCountAutoCarveout<epilogue_smem>
    BuilderScheduleTag              // KernelScheduleAuto or explicit
>
```

**Constraints:**
- Elements: F8F6F4 only (e4m3, e5m2, e3m2, e2m3, e2m1 and mixes)
- Layout: **TN only** (RowMajor A, ColumnMajor B) — hard assert
- Cluster: **1×1×1 only** — `static_assert(size(ClusterShape) == 1)`
- MMA atom: `SM120_16x8x32_TN<ElementA, ElementB, float>`
- Atom layout: Cooperative `Shape<_4,_2,_1>`, Pingpong `Shape<_2,_2,_1>`
- PermTile: M capped at 128, N capped at 32, K fixed at 32
- SchedulerPipelineStageCount: 2

**Tested tile shapes:**

| TileShape | Element types | Output |
|---|---|---|
| 128×64×64 | e4m3×e4m3 | f32, f16 |
| 128×64×128 | e2m1×e2m1, e2m1×e3m2, etc. | f32, f16 |
| 128×64×128 | e3m2×e3m2 | f32 |

**Kernel schedules:**
```cpp
KernelScheduleAuto                              // → Cooperative
KernelTmaWarpSpecializedCooperative             // Explicit cooperative
KernelTmaWarpSpecializedPingpong                // Explicit pingpong
KernelTmaWarpSpecializedCooperativeSm120<2>     // SM120-specific
KernelTmaWarpSpecializedPingpongSm120<2>        // SM120-specific
KernelScheduleF8f6f4Sm120                       // SM120 F8F6F4-specific
```

**Dispatch policy:**
```cpp
MainloopSm120TmaWarpSpecialized<
    PipelineStages,             // auto-computed from SMEM
    SchedulerPipelineStageCount, // 2
    ClusterShape,               // 1×1×1
    KernelSchedule              // Cooperative or Pingpong
>
```

### 4.2 NVFP4 Block-Scaled GEMM

**Builder:** `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`

```cpp
// Template signature
CollectiveBuilder<
    arch::Sm120,                           // ArchTag
    arch::OpClassBlockScaledTensorOp,      // OperatorClass (NOT OpClassTensorOp)
    ElementPairA, LayoutA, AlignmentA,     // nv_float4_t<float_e2m1_t>, RowMajor, 32
    ElementPairB, LayoutB, AlignmentB,     // nv_float4_t<float_e2m1_t>, ColumnMajor, 32
    ElementAccumulator,                    // float
    TileShape_MNK,                         // e.g. Shape<_128,_128,_256>
    ClusterShape_MNK,                      // MUST be Shape<_1,_1,_1>
    StageCountType,                        // StageCountAutoCarveout<epilogue_smem>
    BuilderScheduleTag                     // KernelScheduleAuto or explicit
>
```

**Element pair types:**
```cpp
// NVFP4: FP4 data + UE4M3 scale factors, SFVectorSize=16
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
// → DataType = float_e2m1_t
// → ScaleFactorType = float_ue4m3_t
// → SfVectorSize = 16

// MXFP4: FP4 data + UE8M0 scale factors, SFVectorSize=32
using ElementPairA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
// → DataType = float_e2m1_t
// → ScaleFactorType = float_ue8m0_t
// → SfVectorSize = 32

// MXFP8: FP8 data + UE8M0 scale factors, SFVectorSize=32
using ElementPairA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
```

**NVFP4 MMA instruction:**
```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```
- Shape: 16×8×64 (M×N×K per warp)
- Scale vector: 4X (16 elements per UE4M3 scale factor)
- FLOPs/instruction: 16×8×64×2 = 16,384

**MX FP8/F6/F4 blockscaled MMA instruction:**
```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```
- Shape: 16×8×32 (M×N×K per warp)
- Scale vector: 1X (32 elements per UE8M0 scale factor)
- FLOPs/instruction: 16×8×32×2 = 8,192

**Constraints:**
- Layout: **TN only** — hard assert
- Cluster: **1×1×1 only**
- N ≥ 8 (hard assert)
- K must be multiple of 64 (NVFP4) or 32 (MX format)
- SFVectorSize: 16 for NVFP4, 32 for MX
- Scale factor layout: interleaved (Sm1xxBlockScaledConfig)
  - Blk_MN = 128 (128 rows/cols per scale factor block)
  - Blk_SF = 4 (4 scale factors per indivisible block)
  - Basic block: 32×4 MN, SFVectorSize×MMA_NSF K

**Tested tile shapes (NVFP4):**

| TileShape | Schedule | Output |
|---|---|---|
| 128×128×256 | Cooperative | bf16, f16, f32 |
| 128×128×256 | Pingpong | bf16 |
| 128×128×256 | Stream-K | f32 |
| 128×64×256 | Cooperative | f32 |
| 128×32×256 | Cooperative | f32 |
| 128×32×128 | Cooperative (group) | bf16 |
| 128×16×256 | Cooperative | f32 |
| 128×16×128 | Cooperative (group) | bf16 |
| 128×8×256 | Cooperative | f32 |

**Tested tile shapes (MX format):**

| TileShape | Format | Output |
|---|---|---|
| 128×128×256 | mxf4×mxf4 | f32 |
| 128×128×128 | mxf6×mxf8 | f32 |
| 128×128×128 | mxf8×mxf4 (group) | f32 |
| 128×64×256 | mxf4×mxf4 | f32 |
| 128×32×256 | mxf4×mxf4 | f32 |

**Kernel schedules:**
```cpp
KernelScheduleAuto                                    // → Cooperative blockscaled
KernelTmaWarpSpecializedCooperative                   // Generic cooperative
KernelTmaWarpSpecializedPingpong                      // Generic pingpong
KernelTmaWarpSpecializedNvf4Sm120                     // NVFP4-specific cooperative
KernelTmaWarpSpecializedPingpongNvf4Sm120             // NVFP4-specific pingpong
KernelTmaWarpSpecializedMxf4Sm120                     // MXFP4-specific cooperative
KernelTmaWarpSpecializedMxf8f6f4Sm120                 // MXF8F6F4-specific cooperative
KernelTmaWarpSpecializedCooperativeBlockScaledSm120<3> // SM120 blockscaled cooperative
KernelTmaWarpSpecializedPingpongBlockScaledSm120<3>    // SM120 blockscaled pingpong
```

**Dispatch policy:**
```cpp
MainloopSm120TmaWarpSpecializedBlockScaled<
    PipelineStages,              // auto-computed
    SchedulerPipelineStageCount, // 3
    ClusterShape,                // 1×1×1
    KernelSchedule
>
```

**Stage computation:**
```
ReducedSmemCapacity = 101376
                    - SchedulerPipelineStorage (CLC fetch async, 3 stages)
                    - CLCResponseStorage (3 × CLCResponse)
                    - TensorMapStorage (0 for non-grouped)
PipelineStages = sm100_compute_stage_count_or_override_blockscaled<
    ReducedSmemCapacity, SmemAllocTypeA, SmemAllocTypeB,
    TileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountAutoCarveout)
```

### 4.3 Complete NVFP4 GEMM Example (from test)

```cpp
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

using namespace cute;

// ── Element types ──
using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue4m3_t;

// ── NVFP4 wrapper types ──
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

// ── Layout (TN only) ──
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

// ── Alignment ──
constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value; // 32
constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value; // 32
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;    // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;    // 8

// ── Tile and cluster ──
using TileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

// ── Epilogue ──
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// ── Mainloop ──
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA, AlignmentA,
    ElementPairB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

// ── Kernel ──
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ── Scale factor layouts (interleaved) ──
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
```

### 4.4 Mainloop Compute Structure

The SM120 blockscaled MMA mainloop (`sm120_blockscaled_mma_tma.hpp`) uses a
warp-specialized producer/consumer design:

**Producer warp (TMA load):**
```
for each k_tile:
    pipeline.producer_acquire(smem_pipe_write)
    TMA load A tile → smem
    TMA load B tile → smem
    TMA load SFA tile → smem
    TMA load SFB tile → smem
    ++smem_pipe_write
```

**Consumer warps (MMA compute):**
```
pipeline.consumer_wait(smem_pipe_read)
copy_kblock(0)  // smem→rmem for A, B, SFA, SFB
for each k_tile:
    for each k_block in K_BLOCK_MAX:
        if last k_block:
            NamedBarrier::sync()
            pipeline.consumer_release(smem_pipe_read)
            pipeline.consumer_wait(next_stage)
        copy_kblock(next)
        fp4_shift_A/B (left-shift FP4 data)
        gemm(tiled_mma, zip(CrA, CrSFA), zip(CrB, CrSFB), accum)
```

The `fp4_shift` step is needed because FP4 data is packed 2 elements per byte
and the MMA instruction expects a specific bit alignment.

### 4.5 Epilogue

SM120 has its own epilogue builder (`sm120_builder.inl`) supporting:
- TMA store (direct smem→gmem)
- Epilogue fusion (bias, activation, scale)
- NVFP4 output (narrow output epilogue)
- BF16/FP16/FP32 output types

---

## 5. Python DSL Support

CUTLASS 4.6.1 includes CuTeDSL Python bindings for SM120:

| File | Classes |
|---|---|
| `python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py` | `MmaSM120BlockScaledOp`, `MmaMXF4Op`, `MmaMXF4NVF4Op`, `MmaMXF8Op` |
| `python/CuTeDSL/cutlass/utils/blockscaled_layout.py` | Scale factor layout helpers |
| `python/CuTeDSL/cutlass/utils/blackwell_helpers.py` | Blackwell helper functions |
| `python/cutlass_library/generator.py` | SM120 kernel generation |

This means a Python-native NVFP4 GEMM kernel is feasible without C++ compilation,
though the C++ path is more mature and better tested.

---

## 6. Migration Strategies

### 6.1 Option A: FP8 Weight-Only Quantization (Recommended First Step)

**Approach:** Quantize BF16 weights → FP8 (e4m3) offline. At runtime, quantize
BF16 activations → FP8 on-the-fly, run SM120 FP8 GEMM, dequantize output.

**Implementation:**

```python
# Step 1: Offline weight quantization
def quantize_weight_fp8(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 weight to FP8 e4m3 with per-channel scale."""
    # weight_bf16: [N, K]
    amax = weight_bf16.abs().amax(dim=1, keepdim=True)  # [N, 1]
    scale = amax / 448.0  # FP8 e4m3 max = 448
    scale = scale.clamp(min=1e-12)
    weight_fp8 = (weight_bf16 / scale).to(torch.float8_e4m3fn)
    return weight_fp8, scale.to(torch.float32)

# Step 2: Runtime forward
class FP8Linear(nn.Module):
    def __init__(self, weight_fp8, weight_scale, bias=None):
        super().__init__()
        self.weight_fp8 = nn.Parameter(weight_fp8, requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activation
        x_amax = x.abs().amax(dim=-1, keepdim=True)
        x_scale = x_amax / 448.0
        x_scale = x_scale.clamp(min=1e-12)
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)

        # FP8 GEMM via torch._scaled_mm or custom CUTLASS kernel
        out = torch._scaled_mm(
            x_fp8, self.weight_fp8.t(),
            scale_a=x_scale, scale_b=self.weight_scale.t(),
            out_dtype=torch.bfloat16
        )
        if self.bias is not None:
            out = out + self.bias
        return out
```

**CUTLASS C++ kernel (if torch._scaled_mm doesn't use SM120):**

```cpp
// SM120 FP8 dense GEMM
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using TileShape = Shape<_128, _64, _64>;
using ClusterShape = Shape<_1, _1, _1>;

// ... (same builder pattern as §4.1)
```

**Expected speedup:** ~1.5-2× on weight-bandwidth-bound decode GEMMs.
**Accuracy risk:** Medium. FP8 e4m3 has 3 mantissa bits (vs BF16's 7).
Attention projections may be sensitive.

**Validation plan:**
1. Quantize one layer's projections → measure cosine similarity vs BF16
2. Run greedy fixed-prompt suite → check token-level agreement
3. Measure MTP acceptance rate delta
4. If acceptance drops >0.5%, exclude attention projections

### 6.2 Option B: NVFP4 Full Quantization (Higher Risk, Higher Reward)

**Approach:** Quantize weights → NVFP4 (FP4 + UE4M3 block scales) offline.
Use SM120 NVFP4 blockscaled GEMM.

**Implementation:**

```python
# Step 1: Offline weight quantization
def quantize_weight_nvfp4(weight_bf16: torch.Tensor, block_size: int = 16):
    """Quantize BF16 weight to NVFP4 with UE4M3 block scales."""
    N, K = weight_bf16.shape
    assert K % block_size == 0

    # Reshape to blocks
    blocks = weight_bf16.view(N, K // block_size, block_size)

    # Per-block scale (UE4M3 range: 2^-9 to 2^8, ~0.002 to 256)
    block_amax = blocks.abs().amax(dim=2, keepdim=True)
    # FP4 e2m1 max = 6.0
    block_scale = block_amax / 6.0
    block_scale = block_scale.clamp(min=2**-9, max=2**8)

    # Quantize to FP4
    blocks_scaled = blocks / block_scale
    blocks_fp4 = blocks_scaled.clamp(-6.0, 6.0)  # FP4 range
    # ... pack to e2m1 format ...

    # Convert scale to UE4M3
    scale_ue4m3 = block_scale.to(torch.float8_e4m3fn)  # unsigned variant

    return packed_fp4, scale_ue4m3
```

**CUTLASS C++ kernel:**
```cpp
// SM120 NVFP4 blockscaled GEMM (see §4.3 for complete example)
using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using TileShape = Shape<_128, _128, _256>;
// ... builder with OpClassBlockScaledTensorOp ...
```

**Expected speedup:** ~2.5-3.5× on weight-bandwidth-bound decode GEMMs.
**Accuracy risk:** HIGH. The checkpoint explicitly excluded attention from NVFP4.
FP4 has only 1 mantissa bit. Block scaling (16 elements) adds overhead.

### 6.3 Option C: Hybrid (Recommended Production Path)

| Component | Current | Proposed | Risk |
|---|---|---|---|
| QKV proj (48 layers) | BF16 → SM80 | **Keep BF16** | — |
| O proj (48 layers) | BF16 → SM80 | **Keep BF16** | — |
| G proj (48 layers) | BF16 → SM80 | FP8 weight-only | Low |
| Layer 0 MLP (3 GEMMs) | BF16 → SM80 | FP8 weight-only | Low |
| Shared expert (47×3) | BF16 → SM80 | FP8 weight-only | Low |
| MoE gate (47×1) | BF16 → SM80 | FP8 weight-only | Low |
| LM head | BF16 → SM80 | FP8 weight-only | Medium |

This eliminates ~283 of 376 GEMM calls from SM80 (75%), keeping the 96
attention QKV+O projections in BF16 for accuracy.

**Expected overall speedup:** ~1.1-1.2× model-level (dense GEMM is 12.9% of
GPU time; eliminating 75% of those calls at ~1.5× gives ~0.129 × 0.75 × 0.5 ≈
4.8% total improvement).

### 6.4 Option D: torch._scaled_mm (Lowest Effort)

Check whether PyTorch's `torch._scaled_mm` dispatches to SM120-native FP8:

```python
# Quick test
a = torch.randn(128, 3072, device='cuda').to(torch.float8_e4m3fn)
b = torch.randn(3072, 3072, device='cuda').to(torch.float8_e4m3fn)
sa = torch.ones(1, device='cuda')
sb = torch.ones(1, device='cuda')
out = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
# Profile with nsys to check which kernel is dispatched
```

If cuBLAS has an SM120 FP8 path, this is zero-effort. If not, a custom
CUTLASS kernel is needed.

---

## 7. Performance Analysis

### 7.1 MMA Throughput Comparison

| Path | MMA instruction | FLOPs/warp/instr | K per instr | Relative |
|---|---|---|---|---|
| SM80 BF16 | `mma.sync.m16n8k16.bf16` | 4,096 | 16 | 1.0× |
| SM120 FP8 | `mma.sync.kind::f8f6f4.m16n8k32` | 8,192 | 32 | 2.0× |
| SM120 NVFP4 | `mma.sync.kind::mxf4nvf4.m16n8k64` | 16,384 | 64 | 4.0× |

### 7.2 Memory Bandwidth Analysis (Decode, M=1)

At decode batch sizes (M=1-4), all dense GEMMs are **weight-bandwidth-bound**:

| Format | Weight bytes/elem | Bandwidth vs BF16 | GEMM speedup (BW-bound) |
|---|---|---|---|
| BF16 | 2.0 | 1.0× | 1.0× |
| FP8 e4m3 | 1.0 | 2.0× | ~1.8× (overhead) |
| NVFP4 | 0.5 + scales | ~3.5× | ~2.5-3.0× (overhead) |

### 7.3 Model-Level Impact

Dense GEMM = 12.9% of GPU time. MoE (already SM120-native) dominates.

| Strategy | SM80 calls eliminated | Dense GEMM speedup | Model-level speedup |
|---|---|---|---|
| FP8 all dense | 376/376 (100%) | ~1.8× | ~1.09× |
| FP8 hybrid (keep attn BF16) | 283/376 (75%) | ~1.5× (weighted) | ~1.06× |
| NVFP4 all dense | 376/376 (100%) | ~2.8× | ~1.13× |
| NVFP4 hybrid | 283/376 (75%) | ~2.2× (weighted) | ~1.09× |

### 7.4 Tile Size / Stage Analysis

SM120's 99 KB SMEM vs SM100's 228 KB means fewer pipeline stages:

For 128×128×256 NVFP4:
- A tile: 128 × 256 × 0.5B = 16 KB
- B tile: 128 × 256 × 0.5B = 16 KB
- SFA: 128/128 × 256/16 × 4 × 1B = 64 B
- SFB: 128/128 × 256/16 × 4 × 1B = 64 B
- Per stage: ~32.1 KB
- Available: ~99 KB - scheduler overhead (~2 KB) ≈ 97 KB
- **Estimated stages: 3**

For 128×64×64 FP8:
- A tile: 128 × 64 × 1B = 8 KB
- B tile: 64 × 64 × 1B = 4 KB
- Per stage: ~12 KB
- **Estimated stages: 7-8**

More stages = better TMA latency hiding. FP8's smaller tiles allow more stages,
partially compensating for lower per-instruction throughput.

---

## 8. Key File Reference

### CUTLASS 4.6.1

| File | Purpose |
|---|---|
| `test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_bf16.cu` | Minimal SM120 NVFP4 GEMM test (128×128×256, 1×1×1) |
| `test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op.cu` | Minimal SM120 FP8 dense GEMM test (128×64×64) |
| `examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu` | Full NVFP4 GEMM example (SM100 target, but API pattern applies) |
| `examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu` | SM100 FP16 GEMM (NOT applicable to SM120 — no BF16/FP16 MMA) |
| `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl` | SM120 blockscaled collective builder |
| `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` | SM120 dense (FP8-only) collective builder |
| `include/cutlass/gemm/collective/builders/sm120_common.inl` | SM120 SMEM/copy atom selectors |
| `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` | SM120 blockscaled MMA mainloop (TMA+pipeline) |
| `include/cutlass/gemm/collective/sm120_mma_tma.hpp` | SM120 dense MMA mainloop |
| `include/cute/arch/mma_sm120.hpp` | SM120 MMA atoms (all PTX instructions) |
| `include/cute/atom/mma_traits_sm120.hpp` | SM120 MMA traits and op selectors |
| `include/cutlass/gemm/dispatch_policy.hpp` | SM120 dispatch policies (lines 585-948) |
| `include/cutlass/epilogue/collective/builders/sm120_builder.inl` | SM120 epilogue builder |
| `include/cutlass/detail/sm100_blockscaled_layout.hpp` | Scale factor interleaved layout (shared SM100/SM120) |
| `include/cutlass/float_subbyte.h` | `nv_float4_t`, `mx_float4_t` type definitions |
| `python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py` | Python DSL SM120 blockscaled MMA ops |

### Runtime

| File | Purpose |
|---|---|
| `runtime/model/plain_linear.py` | Current BF16 dense linear (F.linear → cuBLAS → SM80) |
| `runtime/model/laguna_decoder.py` | Decoder layer with all PlainLinear call sites |
| `runtime/model/laguna_model.py` | Top-level model graph |
| `runtime/model/plain_embedding.py` | Embedding + LM head (F.linear) |
| `runtime/backends/laguna_sparkinfer_moe.py` | Already-SM120-native NVFP4 MoE kernel |
| `tests/fixtures/laguna_configs/target/config.json` | Model dimensions |

---

## 9. Recommendations

1. **Immediate (zero-risk):** Profile `torch._scaled_mm` on SM120 to check if
   cuBLAS dispatches SM120-native FP8. If yes, this is the fastest path.

2. **Short-term (low-risk):** Implement FP8 weight-only quantization for
   non-attention dense layers (shared expert, layer-0 MLP, MoE gate, g_proj).
   Validate with greedy fixed-prompt suite + MTP acceptance rate.

3. **Medium-term (medium-risk):** Extend FP8 to attention projections if
   accuracy validation passes. This eliminates 100% of SM80 calls.

4. **Long-term (high-risk):** Investigate NVFP4 for dense layers if FP8
   accuracy is insufficient for further speedup. Requires careful block-scale
   tuning and extensive acceptance-rate validation.

5. **Do NOT attempt:** BF16 CUTLASS SM120 kernels. The hardware does not
   support them. Any "SM120 BF16 GEMM" effort is a dead end.
