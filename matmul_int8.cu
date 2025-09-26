#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#include <stdio.h>
#include </home/thaiv7/Desktop/cuda_project/utils/in_out_helper.h>
#include </home/thaiv7/Desktop/cuda_project/utils/matrix_utils.cu>
#include </home/thaiv7/Desktop/cuda_project/utils/gpu_helper.cu>

#define BLOCK_SIZE 32

inline double gflops_from_ms(long long M, long long N, long long K,
                             double elapsed_ms, int repeats = 1)
{
    // Total floating-point operations for GEMM
    long double ops = 2.0L * (long double)M * (long double)N * (long double)K * (long double)repeats;
    long double seconds = elapsed_ms / 1000.0L;
    long double gflops = ops / (seconds * 1.0e9L);
    return static_cast<double>(gflops);
}

__global__ void matmul_int8(const int8_t *A, const int8_t *B, int32_t *C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int32_t sum = 0;
        for (int k = 0; k < K; ++k)
        {
            int8_t a = A[row * K + k];
            int8_t b = B[k * N + col];
            sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_int8_tiled(const int8_t *__restrict__ A,
                                  const int8_t *__restrict__ B,
                                  int32_t *__restrict__ C,
                                  int M, int N, int K)
{
    // Block coordinates map to a TILE x TILE tile of C
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Shared memory tiles
    __shared__ int8_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int8_t Bs[BLOCK_SIZE][BLOCK_SIZE];

    int32_t acc = 0;

    // Sweep over K dimension in TILE-sized chunks
    for (int kt = 0; kt < K; kt += BLOCK_SIZE)
    {
        // Global indices we want to load
        int a_col = kt + threadIdx.x; // column in A
        int b_row = kt + threadIdx.y; // row in B

        // Guarded loads into shared memory (zero-pad out-of-range)
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : (int8_t)0;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : (int8_t)0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write back
    if (row < M && col < N)
    {
        C[row * N + col] = acc;
    }
}

int main()
{
    int M = 512, N = 512, K = 512;

    int8_t hA[M * K], hB[K * N];
    int32_t hC[M * N];

    // Fill small test matrices
    for (int i = 0; i < M * K; ++i)
        hA[i] = 1;
    for (int i = 0; i < K * N; ++i)
        hB[i] = 1;

    int8_t *dA, *dB;
    int32_t *dC;
    cudaMalloc(&dA, M * K * sizeof(int8_t));
    cudaMalloc(&dB, K * N * sizeof(int8_t));
    cudaMalloc(&dC, M * N * sizeof(int32_t));

    cudaMemcpy(dA, hA, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warm up
    for (int i = 0; i < 10; ++i)
    {
        matmul_int8_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
    }

    float total_time = 0.0f;
    GpuTimer timer;
    for (int i = 0; i < 10; ++i)
    {
        timer.Start();
        // matmul_int8<<<grid, block>>>(dA, dB, dC, M, N, K);
        matmul_int8_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();
        timer.Stop();
        float elapsed = timer.Elapsed();
        total_time += elapsed;
    }
    float elapsed = total_time / 10.0f;
    printf("Time: %f ms\n", elapsed);
    double gflops = gflops_from_ms(M, N, K, elapsed);
    printf("Performance: %f GFLOPS\n", gflops);

    cudaMemcpy(hC, dC, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
