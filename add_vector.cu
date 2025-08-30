#include <iostream>
using namespace std;
#include <stdio.h>
#include </home/thaiv7/Desktop/cuda_project/utils/in_out_helper.h>
#include </home/thaiv7/Desktop/cuda_project/utils/gpu_helper.cu>

__global__ void addTwoVectorsDevice(float A[], float B[], float C[], int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
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

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    int NUM_GRID = int(N / BLOCK_SIZE) + 1;

    GpuTimer timer;
    timer.Start();
    addTwoVectorsDevice<<<NUM_GRID, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // wait to device computation is finished.
    timer.Stop();
    printf("GPU Time: %.3f ms\n", timer.Elapsed());

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    bool isSimilar = compare2Vector(C, trueC, N);
    if (isSimilar == true)
    {
        cout << "CUDA implementation is correct" << endl;
    } else {
        cout << "CUDA is incorrect" << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
