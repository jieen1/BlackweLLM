/* decode_loop.cu — Zero-Python-overhead decode loop.
 *
 * Pre-compute all per-step metadata on GPU, then a tight C++ loop
 * does GPU→GPU copies + cudaGraphLaunch. Eliminates 2.7ms Python
 * overhead per step (PyTorch dispatcher + _run_plan + H2D copies).
 *
 * Build: python setup_decode_loop.py build_ext --inplace
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

/*
 * decode_loop_cuda: run N decode steps with zero Python per-step overhead.
 *
 * Args:
 *   graph_exec_handle: int64 pointer to cudaGraphExec_t
 *   input_ids: [1] GPU tensor (argmax feedback, updated in-place by graph)
 *   out_tokens: [max_steps+1] GPU tensor (accumulates output)
 *   positions: [max_steps] GPU pre-computed positions
 *   slot_mappings: [max_steps] GPU pre-computed slot mappings
 *   swa_slot_mappings: [max_steps] GPU pre-computed SWA slot mappings
 *   lpls: [max_steps] GPU pre-computed last_page_lens (int32)
 *   swa_lpls: [max_steps] GPU pre-computed SWA last_page_lens (int32)
 *   target_positions: [1] GPU - where to write position each step
 *   target_slot_mapping: [1] GPU - where to write slot_mapping
 *   target_swa_slot_mapping: [1] GPU
 *   target_lpl: [1] GPU - where to write last_page_len
 *   target_swa_lpl: [1] GPU
 *   max_steps: number of decode steps
 *
 * The graph already contains: forward + argmax + write input_ids[0].
 * We just need to update metadata buffers before each replay.
 */
void decode_loop_cuda(
    int64_t graph_exec_handle,
    torch::Tensor input_ids,
    torch::Tensor out_tokens,
    torch::Tensor positions,
    torch::Tensor slot_mappings,
    torch::Tensor swa_slot_mappings,
    torch::Tensor lpls,
    torch::Tensor swa_lpls,
    torch::Tensor target_positions,
    torch::Tensor target_slot_mapping,
    torch::Tensor target_swa_slot_mapping,
    torch::Tensor target_lpl,
    torch::Tensor target_swa_lpl,
    int64_t max_steps
) {
    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(graph_exec_handle);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get raw pointers for GPU→GPU copies
    int64_t* pos_src = positions.data_ptr<int64_t>();
    int64_t* sm_src = slot_mappings.data_ptr<int64_t>();
    int64_t* swa_sm_src = swa_slot_mappings.data_ptr<int64_t>();
    int32_t* lpl_src = lpls.data_ptr<int32_t>();
    int32_t* swa_lpl_src = swa_lpls.data_ptr<int32_t>();

    int64_t* pos_dst = target_positions.data_ptr<int64_t>();
    int64_t* sm_dst = target_slot_mapping.data_ptr<int64_t>();
    int64_t* swa_sm_dst = target_swa_slot_mapping.data_ptr<int64_t>();
    int32_t* lpl_dst = target_lpl.data_ptr<int32_t>();
    int32_t* swa_lpl_dst = target_swa_lpl.data_ptr<int32_t>();

    int64_t* out_ptr = out_tokens.data_ptr<int64_t>();
    int64_t* in_ptr = input_ids.data_ptr<int64_t>();

    for (int64_t step = 0; step < max_steps; step++) {
        // GPU→GPU copies (async, same stream, ~1μs each)
        cudaMemcpyAsync(pos_dst, &pos_src[step], sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(sm_dst, &sm_src[step], sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(swa_sm_dst, &swa_sm_src[step], sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(lpl_dst, &lpl_src[step], sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(swa_lpl_dst, &swa_lpl_src[step], sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);

        // Launch the captured graph
        cudaGraphLaunch(exec, stream);

        // Copy argmax result to output accumulator (GPU→GPU)
        cudaMemcpyAsync(&out_ptr[step + 1], in_ptr, sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_loop_cuda", &decode_loop_cuda,
          "Zero-overhead C++ decode loop (pre-computed metadata + graph launch)");
}
