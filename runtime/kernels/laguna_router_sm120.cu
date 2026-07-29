/*
 * Fixed Laguna MoE router for SM120.
 *
 * This is an Apache-2.0 specialization of the executed 256-expert FP32
 * sigmoid path in vLLM's csrc/libtorch_stable/moe/topk_softmax_kernels.cu,
 * itself adapted from TensorRT-LLM.  It deliberately preserves the source
 * operation order for bit-level oracle parity; do not refactor the math or
 * tie-breaking without renewing the full router oracle evidence.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kExperts = 256;
constexpr int kTopK = 10;
constexpr int kWarpsPerBlock = 4;
constexpr int kValuesPerLane = 8;

__global__ __launch_bounds__(32 * kWarpsPerBlock) void laguna_router_f32_kernel(
    const float* logits,
    const float* correction_bias,
    float* topk_weights,
    int32_t* topk_ids,
    int32_t num_rows) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x * kWarpsPerBlock + threadIdx.y;
    if (row >= num_rows) {
        return;
    }

    const float* row_logits = logits + static_cast<size_t>(row) * kExperts;
    const float4 first = *reinterpret_cast<const float4*>(row_logits + lane * 4);
    const float4 second = *reinterpret_cast<const float4*>(row_logits + lane * 4 + 128);
    float scores[kValuesPerLane] = {
        first.x, first.y, first.z, first.w, second.x, second.y, second.z, second.w,
    };
    float scores_for_choice[kValuesPerLane];

#pragma unroll
    for (int value_idx = 0; value_idx < kValuesPerLane; ++value_idx) {
        float score = 1.0f / (1.0f + __expf(-scores[value_idx]));
        if (isnan(score) || isinf(score)) {
            score = 0.0f;
        }
        scores[value_idx] = score;

        const int expert = lane * 4 + (value_idx < 4 ? value_idx : value_idx + 124);
        scores_for_choice[value_idx] = score + correction_bias[expert];
    }

    float selected_sum = 0.0f;
#pragma unroll
    for (int topk_idx = 0; topk_idx < kTopK; ++topk_idx) {
        float max_for_choice = scores_for_choice[0];
        float max_score = scores[0];
        int expert = lane * 4;

#pragma unroll
        for (int value_idx = 0; value_idx < kValuesPerLane; ++value_idx) {
            const float value_for_choice = scores_for_choice[value_idx];
            if (value_for_choice > max_for_choice) {
                max_for_choice = value_for_choice;
                max_score = scores[value_idx];
                expert = lane * 4 + (value_idx < 4 ? value_idx : value_idx + 124);
            }
        }

#pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            const float other_max_for_choice = __shfl_xor_sync(0xffffffff, max_for_choice, mask, 32);
            const float other_max_score = __shfl_xor_sync(0xffffffff, max_score, mask, 32);
            const int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, 32);
            if (other_max_for_choice > max_for_choice ||
                (other_max_for_choice == max_for_choice && other_expert < expert)) {
                max_for_choice = other_max_for_choice;
                max_score = other_max_score;
                expert = other_expert;
            }
        }

        const int output_idx = row * kTopK + topk_idx;
        if (lane == 0) {
            topk_weights[output_idx] = max_score;
            topk_ids[output_idx] = expert;
            selected_sum += max_score;
        }

        if (topk_idx + 1 < kTopK && lane == (expert / 4) % 32) {
            const int value_idx = (expert / 128) * 4 + expert % 4;
            scores_for_choice[value_idx] = -10000.0f;
        }
    }

    if (lane == 0) {
        const float denom = selected_sum > 0.0f ? selected_sum : 1.0f;
        float scale = 1.0f;
        scale /= denom;
#pragma unroll
        for (int topk_idx = 0; topk_idx < kTopK; ++topk_idx) {
            const int output_idx = row * kTopK + topk_idx;
            topk_weights[output_idx] = topk_weights[output_idx] * scale;
        }
    }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int qsr_laguna_router_abi_version() {
    return 1;
}

extern "C" __attribute__((visibility("default"))) int qsr_laguna_router_f32(
    const float* logits,
    const float* correction_bias,
    float* topk_weights,
    int32_t* topk_ids,
    int32_t num_rows,
    cudaStream_t stream) {
    if (logits == nullptr || correction_bias == nullptr || topk_weights == nullptr || topk_ids == nullptr) {
        return -1;
    }
    if (num_rows < 0) {
        return -2;
    }
    if (num_rows == 0) {
        return 0;
    }

    const dim3 block(32, kWarpsPerBlock);
    const dim3 grid((num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    laguna_router_f32_kernel<<<grid, block, 0, stream>>>(
        logits, correction_bias, topk_weights, topk_ids, num_rows);
    const cudaError_t status = cudaPeekAtLastError();
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}
