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
    // Total floating-point operation for GEMM
    long double ops = 2.0L * (long double)M * (long double)N * (long double)K * (long double)repeats;
    long double seconds = elapsed_ms / 1000.0L;
    long double gflops = ops / (seconds * 1.0e9L);
    return static_cast<double>(gflops);
}

__global__ void matmulDeviceKernel(Matrix d_A, Matrix d_B, Matrix d_C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < d_C.height) && (col < d_C.width))
    {
        float value = 0;

        for (int k = 0; k < d_A.width; k++)
        {
            value += getElementMatrix(d_A, row, k) * getElementMatrix(d_B, k, col);
        }
        setElementMatrix(d_C, row, col, value);
    }
}

// TODO: implement for matrix size not multiple of BLOCK_SIZE
__global__ void matmulDeviceKernelSharedMem(Matrix d_A, Matrix d_B, Matrix d_C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float subMatrixA_share[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subMatrixB_share[BLOCK_SIZE][BLOCK_SIZE];

    int numTile = (d_A.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float value = 0.0;
    for (int idxTile = 0; idxTile < numTile; idxTile++)
    {
        int sx = threadIdx.x;
        int sy = threadIdx.y;

        if ((row < d_A.height) && (idxTile * BLOCK_SIZE + sx < d_A.width))
            subMatrixA_share[sy][sx] = getElementMatrix(d_A, row, idxTile * BLOCK_SIZE + sx);
        else
            subMatrixA_share[sy][sx] = 0.0;

        if ((idxTile * BLOCK_SIZE + sy < d_B.height) && (col < d_B.width))
            subMatrixB_share[sy][sx] = getElementMatrix(d_B, idxTile * BLOCK_SIZE + sy, col);
        else
            subMatrixB_share[sy][sx] = 0.0;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            value += subMatrixA_share[sy][k] * subMatrixB_share[k][sx];
        }
        __syncthreads();
    }
    if ((row < d_C.height) && (col < d_C.width))
    {
        setElementMatrix(d_C, row, col, value);
    }
}

void matmulDevice(Matrix A, Matrix B, Matrix C)
{
    Matrix d_A, d_B, d_C;

    d_A.height = A.height;
    d_A.width = A.width;
    d_A.stride = A.stride;
    cudaMalloc((void **)&d_A.arr, d_A.height * d_A.width * sizeof(float));
    cudaMemcpy(d_A.arr, A.arr, d_A.height * d_A.width * sizeof(float), cudaMemcpyHostToDevice);

    d_B.height = B.height;
    d_B.width = B.width;
    d_B.stride = B.stride;
    cudaMalloc((void **)&d_B.arr, d_B.height * d_B.width * sizeof(float));
    cudaMemcpy(d_B.arr, B.arr, d_B.height * d_B.width * sizeof(float), cudaMemcpyHostToDevice);

    d_C.height = d_A.height;
    d_C.width = d_B.width;
    d_C.stride = C.stride;
    cudaMalloc((void **)&d_C.arr, d_C.height * d_C.width * sizeof(float));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_C.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_C.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm up
    // matmulDeviceKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    for (int i = 0; i < 10; ++i)
    {
        matmulDeviceKernelSharedMem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
    }

    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < 10; ++i)
    {
        timer.Start();
        // matmulDeviceKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        matmulDeviceKernelSharedMem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize(); // wait to device computation is finished.
        timer.Stop();
        float ms = timer.Elapsed();
        totalTime += ms;
    }
    float ms = totalTime / 10.0f;
    printf("Kernel execution time: %f ms\n", ms);

    double gflops = gflops_from_ms(d_A.height, B.width, B.height, ms);
    printf("Performance: %f GFLOPS\n", gflops);

    cudaMemcpy(C.arr, d_C.arr, d_C.height * d_C.width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaFree(d_A.arr);
    cudaFree(d_B.arr);
    cudaFree(d_C.arr);
}

int main()
{
    int m = 512;
    int n = 512;
    int p = 512;

    Matrix A, B, C, C_host;
    initMatrixHost(&A, m, p, 1.0);
    initMatrixHost(&B, p, n, 1.0);
    initMatrixHost(&C_host, m, n, 0.0);
    initMatrixHost(&C, m, n, 0.0);

    matmulHost(A, B, C_host);
    matmulDevice(A, B, C);

    bool isSimilar = compare2Matrix(C, C_host);
    if (isSimilar == true)
    {
        cout << "CUDA implementation is correct" << endl;
    }
    else
    {
        cout << "CUDA is IN-CORRECT !" << endl;
    }

    // printMatrix(A);
    // printMatrix(B);
    // printMatrix(C_host);
    // printMatrix(C);

    free(A.arr);
    free(B.arr);
    free(C_host.arr);
    free(C.arr);
    return 0;
}