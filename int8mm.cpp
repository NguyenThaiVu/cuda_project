
#include <torch/extension.h>
#include <cuda_runtime.h>             // <-- for cudaStream_t, cudaGetLastError, cudaSuccess
#include <ATen/cuda/CUDAContext.h>    // <-- for at::cuda::getCurrentCUDAStream()

#include <cstdint>

// Declaration of the launcher implemented in .cu:
void int8_gemm_tiled_launcher(const int8_t* A, const int8_t* B, int32_t* C,
                              int M, int N, int K, cudaStream_t stream);

torch::Tensor int8_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "A.size(1) must equal B.size(0)");
    const int64_t N = B.size(1);

    A = A.contiguous();
    B = B.contiguous();

    auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

    // Get the current CUDA stream from PyTorch and pass its raw cudaStream_t
    auto torch_stream = at::cuda::getCurrentCUDAStream();
    int8_gemm_tiled_launcher(
        reinterpret_cast<const int8_t*>(A.data_ptr<int8_t>()),
        reinterpret_cast<const int8_t*>(B.data_ptr<int8_t>()),
        C.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        torch_stream.stream()   // <-- cudaStream_t
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm", &int8_gemm, "Int8 GEMM -> Int32 (tiled shared-memory)");
}
