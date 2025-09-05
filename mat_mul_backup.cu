#include <iostream>
#include <fstream>
using namespace std;
#include <stdio.h>
#include "utils/gpu_helper.cu"


#define BLOCK_SIZE 32


/*
Define helper functions for matrix
*/
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ float GetElement(Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

/*
Get the submatrix of A, which starts from (row, col) with height and width
*/
__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int height, int width)
{
    Matrix Asub;
    Asub.width = width;
    Asub.height = height;
    Asub.elements = &A.elements[A.width * row + col];
    return Asub;
}


/*
This function performa matrix multiplication C = A*B, where each thread computes one element of matrix C.
*/
__global__ void mat_mul(Matrix A, Matrix B, Matrix C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int output_width = C.width;
    int output_height = C.height;

    if (row < output_height && col < output_width) 
    {
        float value = 0.0f;
        for (int w = 0; w < A.width; w++) 
        {
            value += A.elements[row * A.width + w] * B.elements[w * B.width + col];
        }

        SetElement(C, row, col, value);
    }
}

/*
This function perform matrix multiplication C = A*B.
This version uses shared memory, where each block compute each C_sub.
*/
__global__ void mat_mul_shared_mem(Matrix A, Matrix B, Matrix C)
{
    int block_row_idx = blockIdx.y;
    int block_col_idx = blockIdx.x;

    // Each block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, block_row_idx * BLOCK_SIZE, block_col_idx * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    float value = 0.0f;

    // Thread row and col
    int row = threadIdx.y;
    int col = threadIdx.x;

    int n_sub_matrix = (A.width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int idx_sub_mat = 0; idx_sub_mat < n_sub_matrix; idx_sub_mat++)
    {
        Matrix Asub = GetSubMatrix(A, block_row_idx * BLOCK_SIZE, idx_sub_mat * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrix(B, idx_sub_mat * BLOCK_SIZE, block_col_idx * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();    // Synchronize to make sure the sub-matrices are loaded

        // Multiply Asub and Bsub together, where each thread computes one element of Csub
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            value += As[row][i] * Bs[i][col];
        }
        __syncthreads();
    }
    SetElement(Csub, row, col, value);
}

/*
Create a matrix multiplication on host code. CPU will call this function to perform matrix multiplication.
*/
void MatMul(Matrix A, Matrix B, Matrix C)
{
    // Device pointers for vectors A, B, and C
    Matrix d_A, d_B, d_C; 

    // Allocate memory on the device
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = d_A.width;
    size_t size_a = d_A.height * d_A.width * sizeof(float);
    cudaMalloc((void **)&d_A.elements, size_a * sizeof(float));
    cudaMemcpy(d_A.elements, A.elements, size_a, cudaMemcpyHostToDevice);

    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = d_B.width;
    size_t size_b = d_B.height * d_B.width * sizeof(float);
    cudaMalloc((void **)&d_B.elements, size_b);
    cudaMemcpy(d_B.elements, B.elements, size_b, cudaMemcpyHostToDevice);

    d_C.width = C.width;
    d_C.height = C.height;
    d_C.stride = d_C.width;
    size_t size_c = d_C.height * d_C.width * sizeof(float);
    cudaMalloc((void **)&d_C.elements, size_c);

    // Call kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((C.width + BLOCK_SIZE) / BLOCK_SIZE,  (C.height + BLOCK_SIZE) / BLOCK_SIZE);

    GpuTimer timer;
    timer.Start();
    mat_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // mat_mul_shared_mem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    timer.Stop();
    printf("GPU time: %.3f ms\n", timer.Elapsed());

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // Copy data from device to host
    cudaMemcpy(C.elements, d_C.elements, size_c, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}




int main(void) 
{
    int height_a = 5000;
    int width_a = 2000;
    int height_b = 2000;
    int width_b = 3000;
    
    Matrix A, B, C;
    A.width = width_a;
    A.height = height_a;
    A.stride = A.width;
    A.elements = new float[height_a * width_a];
    for (int i = 0; i < height_a*width_a; i++) {
        A.elements[i] = i;
    }

    B.width = width_b;
    B.height = height_b;
    B.stride = B.width;
    B.elements = new float[height_b * width_b];
    for (int i = 0; i < height_b*width_b; i++) {
        B.elements[i] = i;
    }

    C.height = height_a;
    C.width = width_b;
    C.stride = C.width;
    C.elements = new float[C.height * C.width];

    MatMul(A, B, C);

    // save_to_file(C.elements, C.height * C.width, "output_matrix_C.txt");

    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;

    return 0;
}








