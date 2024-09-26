#include <stdio.h>

__global__ void matrix_multiplication(float* A, float* B, float* C, int width, int height)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    // printf("Thread x: %d, Thread y: %d\n", threadIdx.x, threadIdx.y);

    if (x < width && y < height) 
    {
        int idx = (y * width + x);  
        float value = 0.0f;
        for (int w = 0; w < width; ++w) 
        {
            value += A[y * width + w] * B[w * width + x];
        }
        C[idx] = value;
    }
}

int main(void) 
{
    int height = 2;
    int width = 2;
    int N = height * width;

    float A[N] = {1, 2, 3, 4};
    float B[N] = {1, 2, 3, 4};
    float C[N];

    // Device pointers for vectors A, B, and C
    float *d_A, *d_B, *d_C; 

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);    

    dim3 blockDim(height, width);
    matrix_multiplication<<<1, blockDim>>>(d_A, d_B, d_C, height, width);

    // Copy data from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
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
