#include <stdio.h>

__global__ void AddTwoVectors(float A[], float B[], float C[], int N) 
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(void) 
{
    int N = 1000;
    float A[N], B[N], C[N]; // Arrays for vectors A, B, and C

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i+1;
    }

    // Device pointers for vectors A, B, and C
    float *d_A, *d_B, *d_C; 

    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);    

    AddTwoVectors<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy data from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // Waits untill all CUDA threads are executed
    cudaDeviceSynchronize();
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < N; ++i) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    return 0;
}
