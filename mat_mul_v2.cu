#include <iostream>
#include <fstream>
using namespace std;
#include <stdio.h>
#include "utils/helper.cu"


# define BLOCK_SIZE 32


typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix_Row_Major;


typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix_Colunm_Major;


__device__ float GetElement_Row_Major(Matrix_Row_Major A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement_Row_Major(Matrix_Row_Major A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ float GetElement_Column_Major(Matrix_Colunm_Major A, int row, int col)
{
    return A.elements[col * A.stride + row];
}

__device__ void SetElement_Column_Major(Matrix_Colunm_Major A, int row, int col, float value)
{
    A.elements[col * A.stride + row] = value;
}


/*
This function convert a matrix from row-major format to column-major format.
*/

__global__ void row_to_col_major(Matrix_Row_Major A_row, Matrix_Colunm_Major A_col)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_row.height && col < A_row.width)
    {
        float value = GetElement_Row_Major(A_row, row, col);

        SetElement_Column_Major(A_col, row, col, value);

    }
}

/*
This function performa matrix multiplication C = A*B, where each thread computes one element of matrix C.
NOTE: The second matrix (matrix B) is store in the column-major format, which will speed up the computation via coalesced memory access.
*/
__global__ void mat_mul_v2(Matrix_Row_Major A, Matrix_Colunm_Major B, Matrix_Row_Major C)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < C.height && col < C.width) 
    {
        float value = 0.0f;
        for (int k = 0; k < A.width; k++) 
        {
            value += GetElement_Row_Major(A, row, k) * GetElement_Column_Major(B, k, col);
        }

        SetElement_Row_Major(C, row, col, value);
    }
}

float GetElement_Row_Major_Host(Matrix_Row_Major A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

float GetElement_Column_Major_Host(Matrix_Colunm_Major A, int row, int col)
{
    return A.elements[col * A.stride + row];
}



/*
Create a matrix multiplication on host code. CPU will call this function to perform matrix multiplication.
*/
void Mat_Mul_v2(Matrix_Row_Major A, Matrix_Row_Major B, Matrix_Row_Major C)
{
    // Declare matrix on DEVICE
    Matrix_Row_Major d_A, d_B, d_C;
    Matrix_Colunm_Major d_B_col;

    // Matrix d_A (row-major)
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = A.stride;
    size_t size_d_A = d_A.width * d_A.height;
    cudaMalloc(&d_A.elements, size_d_A * sizeof(float));
    cudaMemcpy(d_A.elements, A.elements, size_d_A * sizeof(float), cudaMemcpyHostToDevice);

    // Matrix d_B (row-major)
    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = B.stride;
    size_t size_d_B = d_B.width * d_B.height;
    cudaMalloc(&d_B.elements, size_d_B * sizeof(float));
    cudaMemcpy(d_B.elements, B.elements, size_d_B * sizeof(float), cudaMemcpyHostToDevice);

    // Matrix d_B_col (column-major)
    d_B_col.width = B.width;
    d_B_col.height = B.height;
    d_B_col.stride = d_B_col.height;
    size_t size_d_B_col = d_B_col.width * d_B_col.height;
    cudaMalloc(&d_B_col.elements, size_d_B_col * sizeof(float));

    // Matrix d_C (row-major)
    d_C.width = C.width;
    d_C.height = C.height;
    d_C.stride = C.stride;
    size_t size_d_C = d_C.width * d_C.height;
    cudaMalloc(&d_C.elements, size_d_C * sizeof(float));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_B_col.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_B_col.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimGrid_mat_mul((C.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (C.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    GpuTimer timer;
    timer.Start();

    // 1. Convert matrix B from row-major to column-major format
    row_to_col_major<<<dimGrid, dimBlock>>>(d_B, d_B_col);
    cudaDeviceSynchronize();

    // 2. Call kernel for Matrix Multiplication
    mat_mul_v2<<<dimGrid_mat_mul, dimBlock>>>(d_A, d_B_col, d_C);

    timer.Stop();
    printf("GPU time: %.3f ms\n", timer.Elapsed());

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Copy data from DEVICE to HOST
    cudaMemcpy(C.elements, d_C.elements, size_d_C * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_B_col.elements);
    cudaFree(d_C.elements);

}


int main()
{
    // Initialize the matrix on HOST
    Matrix_Row_Major A, B, C;
    int height_a = 5000;
    int width_a = 2000;
    int height_b = 2000;
    int width_b = 3000;

    A.height = height_a;
    A.width = width_a;
    A.stride = A.width;
    size_t size_A = A.width * A.height;
    A.elements = (float*)malloc(size_A * sizeof(float));
    for (int i = 0; i < size_A; i++)
    {
        A.elements[i] = i;
    }

    B.height = height_b;
    B.width = width_b;
    B.stride = B.width;
    size_t size_B = B.width * B.height;
    B.elements = (float*)malloc(size_B * sizeof(float));
    for (int i = 0; i < size_B; i++)
    {
        B.elements[i] = i;
    }

    C.height = A.height;
    C.width = B.width;
    C.stride = C.width;
    size_t size_C = C.width * C.height;
    C.elements = (float*)malloc(size_C * sizeof(float));


    Mat_Mul_v2(A, B, C);

    // save_to_file(C.elements, C.height * C.width, "output_matrix_C_v2.txt");

    // Free memory
    free(A.elements);
    free(B.elements);
    free(C.elements);
}

