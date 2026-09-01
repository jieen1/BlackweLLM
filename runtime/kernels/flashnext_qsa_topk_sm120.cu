// SM120 radix-select for Flash-Next QSA's fixed block_topk=512.
//
// The selection algorithm is the same radix-select shape used by SGLang's
// QSA fast_topk operator, but this ABI is deliberately standalone: the
// runtime must not import SGLang's Python package or depend on its wheel at
// serving time.  One 1024-thread CTA owns one query row and scans only the
// device-provided valid prefix.  The output order is unspecified; the Python
// adapter performs a cheap 512-wide rerank to retain the runtime's historical
// score order and deterministic attention accumulation.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>

#define QSR_EXPORT __attribute__((visibility("default")))

namespace {

constexpr int kTopK = 512;
constexpr int kThreads = 1024;
constexpr int kRadix = 256;
// The common score distribution narrows to a small candidate list after the
// coarse half-float byte.  Keep that fast path in shared memory, but never
// silently truncate it: rows whose coarse bin exceeds the capacity take the
// exact row-rescan fallback below.  The fallback is rare for real QSA scores
// and makes the ABI safe for quantized/adversarial rows as well.
constexpr int kCandidateCapacity = 4096;
constexpr size_t kDynamicSmemBytes =
    2 * kCandidateCapacity * sizeof(int);

__device__ __forceinline__ uint8_t float_coarse_key(float value) {
  const __half half_value = __float2half_rn(value);
  const uint16_t bits = __half_as_ushort(half_value);
  const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                       : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t float_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int TopK>
__device__ void naive_topk(
    const float* __restrict__ scores,
    int64_t* __restrict__ output,
    int length) {
  for (int index = threadIdx.x; index < TopK; index += kThreads) {
    output[index] = index < length ? index : -1;
  }
}

// Select exactly TopK indices from one row.  This is a direct standalone
// port of the validated SGLang radix selector: one coarse half-float byte
// narrows the candidate set, then four full-float bytes resolve the boundary.
template <int TopK>
__device__ void radix_select_topk(
    const float* __restrict__ scores,
    int64_t* __restrict__ output,
    int length,
    int* __restrict__ candidates) {
  int remaining = TopK;
  __shared__ int histogram[2][kRadix + 128];
  __shared__ int counter;
  __shared__ int threshold_bin;
  __shared__ int thresholds[4];
  __shared__ int candidate_count[2];
  __shared__ int last_remaining;

  const int tid = threadIdx.x;
  if (tid < kRadix + 1) {
    histogram[0][tid] = 0;
  }
  __syncthreads();

  for (int index = tid; index < length; index += kThreads) {
    atomicAdd(&histogram[0][float_coarse_key(scores[index])], 1);
  }
  __syncthreads();

  auto suffix_sum = [&]() {
#pragma unroll
    for (int pass = 0; pass < 8; ++pass) {
      if (tid < kRadix) {
        const int shift = 1 << pass;
        const int source = pass & 1;
        int value = histogram[source][tid];
        if (tid < kRadix - shift) {
          value += histogram[source][tid + shift];
        }
        histogram[source ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  suffix_sum();
  if (tid < kRadix && histogram[0][tid] >= remaining &&
      histogram[0][tid + 1] < remaining) {
    threshold_bin = tid;
    counter = 0;
  }
  __syncthreads();

  const int coarse_threshold = threshold_bin;
  remaining -= histogram[0][coarse_threshold + 1];

  if (remaining == 0) {
    for (int index = tid; index < length; index += kThreads) {
      if (static_cast<int>(float_coarse_key(scores[index])) >
          coarse_threshold) {
        const int slot = atomicAdd(&counter, 1);
        output[slot] = index;
      }
    }
    __syncthreads();
    return;
  }

  // Build the common fast-path candidate list while emitting values strictly
  // above the coarse boundary.  If the boundary is unusually broad, the
  // candidate counter exposes that fact and the exact row-rescan fallback
  // below handles it without truncation.
  if (tid == 0) {
    candidate_count[0] = 0;
  }
  if (tid < kRadix + 1) {
    histogram[0][tid] = 0;
  }
  __syncthreads();
  for (int index = tid; index < length; index += kThreads) {
    const float value = scores[index];
    const int bin = static_cast<int>(float_coarse_key(value));
    if (bin > coarse_threshold) {
      const int slot = atomicAdd(&counter, 1);
      output[slot] = index;
    } else if (bin == coarse_threshold) {
      const int slot = atomicAdd(&candidate_count[0], 1);
      if (slot < kCandidateCapacity) {
        candidates[slot] = index;
        atomicAdd(&histogram[0][(float_key(value) >> 24) & 0xFFu], 1);
      }
    }
  }
  __syncthreads();

  if (candidate_count[0] > kCandidateCapacity) {
  // Refine the coarse boundary with all four bytes of the monotonic float
  // key.  Each pass scans the original row and keeps only values matching
  // the already-selected high-byte prefix.  This branch is exact even when
  // an entire coarse bin exceeds the shared candidate capacity.
#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    __syncthreads();
    if (tid < kRadix + 1) {
      histogram[0][tid] = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    for (int index = tid; index < length; index += kThreads) {
      const float value = scores[index];
      if (static_cast<int>(float_coarse_key(value)) != coarse_threshold) {
        continue;
      }
      const uint32_t key = float_key(value);
      bool prefix_match = true;
#pragma unroll
      for (int prior = 0; prior < pass; ++prior) {
        if (((key >> (24 - prior * 8)) & 0xFFu) !=
            static_cast<uint32_t>(thresholds[prior])) {
          prefix_match = false;
        }
      }
      if (prefix_match) {
        atomicAdd(&histogram[0][(key >> shift) & 0xFFu], 1);
      }
    }
    __syncthreads();

    suffix_sum();
    if (tid < kRadix && histogram[0][tid] >= remaining &&
        histogram[0][tid + 1] < remaining) {
      threshold_bin = tid;
      thresholds[pass] = tid;
    }
    __syncthreads();

    const int pass_threshold = threshold_bin;
    remaining -= histogram[0][pass_threshold + 1];
    if (remaining == 0) {
      for (int index = tid; index < length; index += kThreads) {
        const float value = scores[index];
        if (static_cast<int>(float_coarse_key(value)) != coarse_threshold) {
          continue;
        }
        const uint32_t key = float_key(value);
        bool prefix_match = true;
#pragma unroll
        for (int prior = 0; prior < pass; ++prior) {
          if (((key >> (24 - prior * 8)) & 0xFFu) !=
              static_cast<uint32_t>(thresholds[prior])) {
            prefix_match = false;
          }
        }
        if (prefix_match && ((key >> shift) & 0xFFu) >
                                static_cast<uint32_t>(pass_threshold)) {
          const int output_slot = atomicAdd(&counter, 1);
          output[output_slot] = index;
        }
      }
      __syncthreads();
      break;
    }

    if (pass == 3) {
      // The final byte still has a strict-higher portion that belongs in the
      // result before the equal-byte tie set is filled.
      for (int index = tid; index < length; index += kThreads) {
        const float value = scores[index];
        if (static_cast<int>(float_coarse_key(value)) != coarse_threshold) {
          continue;
        }
        const uint32_t key = float_key(value);
        bool prefix_match = true;
#pragma unroll
        for (int prior = 0; prior < pass; ++prior) {
          if (((key >> (24 - prior * 8)) & 0xFFu) !=
              static_cast<uint32_t>(thresholds[prior])) {
            prefix_match = false;
          }
        }
        if (prefix_match && ((key >> shift) & 0xFFu) >
                                static_cast<uint32_t>(pass_threshold)) {
          const int output_slot = atomicAdd(&counter, 1);
          output[output_slot] = index;
        }
      }
      __syncthreads();
      if (tid == 0) {
        last_remaining = remaining;
      }
      __syncthreads();
      for (int index = tid; index < length; index += kThreads) {
        const float value = scores[index];
        if (static_cast<int>(float_coarse_key(value)) != coarse_threshold) {
          continue;
        }
        const uint32_t key = float_key(value);
        bool prefix_match = true;
#pragma unroll
        for (int prior = 0; prior < pass; ++prior) {
          if (((key >> (24 - prior * 8)) & 0xFFu) !=
              static_cast<uint32_t>(thresholds[prior])) {
            prefix_match = false;
          }
        }
        if (!prefix_match ||
            ((key >> shift) & 0xFFu) != static_cast<uint32_t>(pass_threshold)) {
          continue;
        }
        const int output_slot = atomicAdd(&last_remaining, -1);
        if (output_slot > 0) {
          output[TopK - output_slot] = index;
        }
      }
      __syncthreads();
      break;
    }

    __syncthreads();
    for (int index = tid; index < length; index += kThreads) {
      const float value = scores[index];
      if (static_cast<int>(float_coarse_key(value)) != coarse_threshold) {
        continue;
      }
      const uint32_t key = float_key(value);
      bool prefix_match = true;
#pragma unroll
      for (int prior = 0; prior < pass; ++prior) {
        if (((key >> (24 - prior * 8)) & 0xFFu) !=
            static_cast<uint32_t>(thresholds[prior])) {
          prefix_match = false;
        }
      }
      if (prefix_match && ((key >> shift) & 0xFFu) >
                              static_cast<uint32_t>(pass_threshold)) {
        const int output_slot = atomicAdd(&counter, 1);
        output[output_slot] = index;
      }
    }
    __syncthreads();
  }
  } else {
    // Fast path: the coarse boundary fits in shared memory, so refine the
    // candidate list instead of rescanning the full pooled-key row.
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      const int source = pass & 1;
      const int next = source ^ 1;
      const int count = candidate_count[source];
      const int shift = 24 - pass * 8;

      suffix_sum();
      if (tid < kRadix && histogram[0][tid] >= remaining &&
          histogram[0][tid + 1] < remaining) {
        threshold_bin = tid;
        thresholds[pass] = tid;
      }
      __syncthreads();

      const int pass_threshold = threshold_bin;
      remaining -= histogram[0][pass_threshold + 1];
      if (remaining == 0) {
        for (int slot = tid; slot < count; slot += kThreads) {
          const int index = candidates[source * kCandidateCapacity + slot];
          const int bin = static_cast<int>(
              (float_key(scores[index]) >> shift) & 0xFFu);
          if (bin > pass_threshold) {
            const int output_slot = atomicAdd(&counter, 1);
            output[output_slot] = index;
          }
        }
        __syncthreads();
        break;
      }

      if (pass == 3) {
        // Emit the strict-higher final-byte values, then fill the remaining
        // slots from the equal-byte tie set.
        for (int slot = tid; slot < count; slot += kThreads) {
          const int index = candidates[source * kCandidateCapacity + slot];
          const int bin = static_cast<int>(
              (float_key(scores[index]) >> shift) & 0xFFu);
          if (bin > pass_threshold) {
            const int output_slot = atomicAdd(&counter, 1);
            output[output_slot] = index;
          }
        }
        __syncthreads();
        if (tid == 0) {
          last_remaining = remaining;
        }
        __syncthreads();
        for (int slot = tid; slot < count; slot += kThreads) {
          const int index = candidates[source * kCandidateCapacity + slot];
          const int bin = static_cast<int>(
              (float_key(scores[index]) >> shift) & 0xFFu);
          if (bin == pass_threshold) {
            const int output_slot = atomicAdd(&last_remaining, -1);
            if (output_slot > 0) {
              output[TopK - output_slot] = index;
            }
          }
        }
        __syncthreads();
        break;
      }

      __syncthreads();
      if (tid < kRadix + 1) {
        histogram[0][tid] = 0;
      }
      if (tid == 0) {
        candidate_count[next] = 0;
      }
      __syncthreads();
      for (int slot = tid; slot < count; slot += kThreads) {
        const int index = candidates[source * kCandidateCapacity + slot];
        const uint32_t key = float_key(scores[index]);
        const int bin = static_cast<int>((key >> shift) & 0xFFu);
        if (bin > pass_threshold) {
          const int output_slot = atomicAdd(&counter, 1);
          output[output_slot] = index;
        } else if (bin == pass_threshold) {
          const int next_slot = atomicAdd(&candidate_count[next], 1);
          // The initial candidate list is bounded, so every refinement set
          // is bounded by the same capacity.
          candidates[next * kCandidateCapacity + next_slot] = index;
          atomicAdd(&histogram[0][(key >> (shift - 8)) & 0xFFu], 1);
        }
      }
      __syncthreads();
    }
  }
}

struct QsaTopKParams {
  const float* scores;
  const int64_t* lengths;
  int64_t* output;
  int64_t stride;
};

__global__ __launch_bounds__(kThreads) void qsa_topk_kernel(QsaTopKParams params) {
  extern __shared__ int candidates[];
  const int row = static_cast<int>(blockIdx.x);
  const int64_t raw_length = params.lengths[row];
  const int length = raw_length < 0 ? 0 : static_cast<int>(raw_length);
  const float* row_scores = params.scores + static_cast<int64_t>(row) * params.stride;
  int64_t* row_output = params.output + static_cast<int64_t>(row) * kTopK;

  if (length <= kTopK) {
    naive_topk<kTopK>(row_scores, row_output, length);
  } else {
    radix_select_topk<kTopK>(row_scores, row_output, length, candidates);
  }
}

}  // namespace

extern "C" QSR_EXPORT int qsr_flashnext_qsa_topk_abi_version() { return 2; }

extern "C" QSR_EXPORT int qsr_flashnext_qsa_topk_sm120(
    const void* scores,
    const void* lengths,
    void* output,
    int rows,
    int64_t stride,
    void* stream) {
  if (scores == nullptr || lengths == nullptr || output == nullptr || rows < 0 ||
      stride < kTopK) {
    return 1;
  }
  qsa_topk_kernel<<<rows, kThreads, kDynamicSmemBytes,
                    static_cast<cudaStream_t>(stream)>>>(
      QsaTopKParams{
          static_cast<const float*>(scores),
          static_cast<const int64_t*>(lengths),
          static_cast<int64_t*>(output),
          stride,
      });
  return static_cast<int>(cudaGetLastError());
}
