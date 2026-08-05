/*
 * SM120 E4M3 W8A8 scaled GEMM for Qwen3.6's FP8-channel Linears.
 *
 * The scheduling and fused per-token/per-channel scale epilogue are adapted
 * from vLLM's Apache-2.0 SM120 CUTLASS implementation.  This deliberately
 * exposes a small raw-pointer ABI: production must not import or link vLLM.
 *
 * Qwen's affected linears have no bias.  A is row-major E4M3 [M, K], B is
 * column-major E4M3 [K, N] (a transpose view of checkpoint [N, K]), A scales
 * are FP32 [M], B scales are FP32 [N], and D is row-major BF16 [M, N].
 */
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <utility>

using namespace cute;

#if defined(_WIN32)
#define QSR_EXPORT __declspec(dllexport)
#else
#define QSR_EXPORT __attribute__((visibility("default")))
#endif

namespace qsr_fp8_w8a8_detail {

using ElementAB = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassTensorOp;

constexpr int kQuantThreads = 256;
constexpr float kFP8E4M3Max = 448.0f;
constexpr float kMinFP8Scale = 1.0f / (kFP8E4M3Max * 512.0f);

static_assert(sizeof(__nv_fp8_e4m3) == sizeof(ElementAB));

struct FloatMax {
  __device__ float operator()(float left, float right) const { return fmaxf(left, right); }
};

// CUTLASS's SM120 specializations are shared by the 12.x family.  Keep the
// historical wrapper's instantiation boundary without importing its headers:
// on our fixed SM120 target this forwards exactly to GemmUniversal, while an
// accidental launch on another architecture traps instead of producing a
// plausible but unsupported result.
template <typename Kernel>
struct EnableSM120Family : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
    Kernel::operator()(std::forward<Args>(args)...);
#else
    asm("trap;");
#endif
#endif
  }
};

__global__ void dynamic_per_token_e4m3_quant_kernel(
    __nv_fp8_e4m3* out, float* scales, __nv_bfloat16 const* input, int columns) {
  int row = blockIdx.x;
  int thread = threadIdx.x;
  float local_max = 0.0f;
  auto const* row_input = input + static_cast<size_t>(row) * columns;
  auto* row_output = out + static_cast<size_t>(row) * columns;
  for (int column = thread; column < columns; column += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(row_input[column])));
  }

  using BlockReduce = cub::BlockReduce<float, kQuantThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float row_scale;
  float row_max = BlockReduce(reduce_storage).Reduce(local_max, FloatMax{});
  if (thread == 0) {
    row_scale = fmaxf(row_max / kFP8E4M3Max, kMinFP8Scale);
    scales[row] = row_scale;
  }
  __syncthreads();

  for (int column = thread; column < columns; column += blockDim.x) {
    // Preserve the historical W8A8 operation order: it divides by the
    // dynamic token scale before clamping and converting.  A reciprocal
    // multiply differs at E4M3 rounding boundaries and changes tokens.
    float scaled = __bfloat162float(row_input[column]) / row_scale;
    scaled = fminf(kFP8E4M3Max, fmaxf(-kFP8E4M3Max, scaled));
    row_output[column] = __nv_fp8_e4m3(scaled);
  }
}

template <typename ElementAcc_, typename ElementD_, typename TileShape>
struct PerTokenChannelScaledEpilogue {
 private:
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0, TileShape, float, float, Stride<_1, _0, int64_t>>;
  using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, TileShape, float, float, Stride<_0, _1, int64_t>>;
  using ScaleBAccum = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleBAccumEvt = cutlass::epilogue::fusion::Sm90EVT<ScaleBAccum, ScaleB, Accum>;
  using ScaleAOutput = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD_, float, cutlass::FloatRoundStyle::round_to_nearest>;

 public:
  using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<
      ScaleAOutput, ScaleA, ScaleBAccumEvt>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(float const* a_scales, float const* b_scales) {
    typename ScaleA::Arguments a_args{a_scales};
    typename ScaleB::Arguments b_args{b_scales};
    typename ScaleBAccumEvt::Arguments scaled_accum_args{b_args, {}, {}};
    return ArgumentType{a_args, scaled_accum_args, {}};
  }
};

template <typename TileShape, typename KernelSchedule, typename EpilogueTile>
struct BuildGemm {
  using ClusterShape = Shape<_1, _1, _1>;
  using Epilogue = PerTokenChannelScaledEpilogue<ElementAcc, ElementD, TileShape>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, TileShape, ClusterShape, EpilogueTile, ElementAcc,
      float, void, cutlass::layout::RowMajor,
      128 / cutlass::sizeof_bits<ElementD>::value, ElementD,
      cutlass::layout::RowMajor, 128 / cutlass::sizeof_bits<ElementD>::value,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      typename Epilogue::EVTCompute>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementAB, cutlass::layout::RowMajor,
      128 / cutlass::sizeof_bits<ElementAB>::value, ElementAB,
      cutlass::layout::ColumnMajor, 128 / cutlass::sizeof_bits<ElementAB>::value,
      ElementAcc, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using Kernel = EnableSM120Family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using GemmM16 = BuildGemm<
    Shape<_16, _64, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    Shape<_16, _32>>;
using GemmM32 = BuildGemm<
    Shape<_32, _64, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    Shape<_32, _32>>;
using GemmM64 = BuildGemm<
    Shape<_64, _64, _128>, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    cutlass::epilogue::collective::EpilogueTileAuto>;
using GemmDefault = BuildGemm<
    Shape<_128, _128, _128>, cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::epilogue::collective::EpilogueTileAuto>;

template <typename GemmConfig>
int run_gemm(void* out, void const* a, void const* b, float const* a_scales,
             float const* b_scales, void* workspace, size_t workspace_size,
             int m, int n, int k, cudaStream_t stream) {
  using Gemm = typename GemmConfig::Gemm;
  using Kernel = typename Gemm::GemmKernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using Epilogue = typename GemmConfig::Epilogue;

  auto problem_shape = Shape<int, int, int, int>{m, n, k, 1};
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));

  typename Kernel::MainloopArguments mainloop_args{
      static_cast<ElementAB const*>(a), stride_a, static_cast<ElementAB const*>(b), stride_b};
  typename Kernel::EpilogueArguments epilogue_args{
      Epilogue::prepare_args(a_scales, b_scales), static_cast<ElementD*>(out),
      stride_c, static_cast<ElementD*>(out), stride_c};
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  problem_shape, mainloop_args, epilogue_args};
  Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  size_t needed_workspace = gemm.get_workspace_size(args);
  if (workspace_size < needed_workspace || (needed_workspace != 0 && workspace == nullptr)) {
    return -20;
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  status = gemm.run(args, workspace, stream);
  return static_cast<int>(status);
}

template <typename GemmConfig>
size_t workspace_size_for_gemm(int m, int n, int k) {
  using Gemm = typename GemmConfig::Gemm;
  using Kernel = typename Gemm::GemmKernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using Epilogue = typename GemmConfig::Epilogue;

  auto problem_shape = Shape<int, int, int, int>{m, n, k, 1};
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop_args{nullptr, stride_a, nullptr, stride_b};
  typename Kernel::EpilogueArguments epilogue_args{
      Epilogue::prepare_args(nullptr, nullptr), nullptr, stride_c, nullptr,
      stride_c};
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  problem_shape, mainloop_args, epilogue_args};
  Gemm gemm;
  return gemm.get_workspace_size(args);
}

}  // namespace qsr_fp8_w8a8_detail

extern "C" QSR_EXPORT int qsr_fp8_w8a8_abi_version() { return 2; }

extern "C" QSR_EXPORT int qsr_fp8_w8a8_dynamic_per_token_quant_sm120(
    void* out, float* scales, void const* input, int rows, int columns,
    cudaStream_t stream) {
  if (out == nullptr || scales == nullptr || input == nullptr || rows <= 0 || columns <= 0 ||
      columns % 16 != 0) {
    return -10;
  }
  qsr_fp8_w8a8_detail::dynamic_per_token_e4m3_quant_kernel<<<
      rows, qsr_fp8_w8a8_detail::kQuantThreads, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(out), scales,
      static_cast<__nv_bfloat16 const*>(input), columns);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" QSR_EXPORT int qsr_fp8_w8a8_workspace_size_sm120(
    size_t* workspace_size, int m, int n, int k, int batch_invariant) {
  if (workspace_size == nullptr || m <= 0 || n <= 0 || k <= 0 || k % 16 != 0 || n % 16 != 0) {
    return -10;
  }
  if (batch_invariant || m <= 256) {
    if (m <= 16 && !batch_invariant) {
      *workspace_size = qsr_fp8_w8a8_detail::workspace_size_for_gemm<
          qsr_fp8_w8a8_detail::GemmM16>(m, n, k);
    } else if (m <= 32 && !batch_invariant) {
      *workspace_size = qsr_fp8_w8a8_detail::workspace_size_for_gemm<
          qsr_fp8_w8a8_detail::GemmM32>(m, n, k);
    } else {
      *workspace_size = qsr_fp8_w8a8_detail::workspace_size_for_gemm<
          qsr_fp8_w8a8_detail::GemmM64>(m, n, k);
    }
  } else {
    *workspace_size = qsr_fp8_w8a8_detail::workspace_size_for_gemm<
        qsr_fp8_w8a8_detail::GemmDefault>(m, n, k);
  }
  return 0;
}

extern "C" QSR_EXPORT int qsr_fp8_w8a8_scaled_mm_sm120(
    void* out, void const* a, void const* b, float const* a_scales,
    float const* b_scales, void* workspace, size_t workspace_size, int m,
    int n, int k, int batch_invariant, cudaStream_t stream) {
  if (out == nullptr || a == nullptr || b == nullptr || a_scales == nullptr ||
      b_scales == nullptr || m <= 0 || n <= 0 || k <= 0 || k % 16 != 0 ||
      n % 16 != 0) {
    return -10;
  }
  if (batch_invariant || m <= 256) {
    if (m <= 16 && !batch_invariant) return qsr_fp8_w8a8_detail::run_gemm<
        qsr_fp8_w8a8_detail::GemmM16>(
        out, a, b, a_scales, b_scales, workspace, workspace_size, m, n, k, stream);
    if (m <= 32 && !batch_invariant) return qsr_fp8_w8a8_detail::run_gemm<
        qsr_fp8_w8a8_detail::GemmM32>(
        out, a, b, a_scales, b_scales, workspace, workspace_size, m, n, k, stream);
    return qsr_fp8_w8a8_detail::run_gemm<qsr_fp8_w8a8_detail::GemmM64>(
        out, a, b, a_scales, b_scales, workspace, workspace_size, m, n, k, stream);
  }
  return qsr_fp8_w8a8_detail::run_gemm<qsr_fp8_w8a8_detail::GemmDefault>(
      out, a, b, a_scales, b_scales, workspace, workspace_size, m, n, k, stream);
}
