#include <iostream>
using namespace std;
#include <stdio.h>
#include </home/thaiv7/Desktop/cuda_project/utils/in_out_helper.h>

__global__ void addTwoVectorKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void addTwoVectorDevice(float *A, float *B, float *C, int n)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, n * sizeof(float));
    cudaMalloc((void **)&d_B, n * sizeof(float));
    cudaMalloc((void **)&d_C, n * sizeof(float));

    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    int GRID_SIZE = int(n / BLOCK_SIZE) + 1;
    addTwoVectorKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize(); // wait to device computation is finished.

    cudaMemcpy(C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void addTwoVectorHost(float A[], float B[], float C[], int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    int N = 1000000;
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    float *trueC = new float[N];

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i;
    }
    addTwoVectorHost(A, B, trueC, N);

    addTwoVectorDevice(A, B, C, N);

    bool isSimilar = compare2Vector(C, trueC, N);
    if (isSimilar == true)
    {
        cout << "CUDA implementation is correct" << endl;
    }
    else
    {
        cout << "CUDA is incorrect" << endl;
    }

    return 0;
}
