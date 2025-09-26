// int8mm_kernel.cu
#include <cuda_runtime.h>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Simple tiled int8 GEMM: C = A (M x K) * B (K x N), int8 inputs, int32 accum/output.
// Row-major, contiguous tensors assumed.
__global__ void int8_gemm_tiled_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C,
    int M, int N, int K)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    __shared__ int8_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int8_t BsT[BLOCK_SIZE][BLOCK_SIZE + 1];  // transpose + pad to avoid bank conflicts

    int32_t acc = 0;

    const int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int a_col = t * BLOCK_SIZE + tx;
        const int b_row = t * BLOCK_SIZE + ty;

        // Guarded loads with zero-padding
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        int8_t bval = (b_row < K && col < N) ? B[b_row * N + col] : 0;
        BsT[tx][ty] = bval;  // store B tile transposed

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            // (int32) promote before multiply-add
            acc += static_cast<int32_t>(As[ty][k]) * static_cast<int32_t>(BsT[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Launcher
void int8_gemm_tiled_launcher(const int8_t* A, const int8_t* B, int32_t* C,
                              int M, int N, int K, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int8_gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
