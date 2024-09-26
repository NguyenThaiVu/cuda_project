#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements in the input array
#define BLOCK_SIZE 32  // Number of threads per block


// CUDA Kernel for parallel reduction (sum)
__global__ void parallelReductionSum(float* input, float* output, int size) 
{
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;  
    }
    __syncthreads();  // Synchronize threads to ensure all data is loaded

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();  // Ensure all additions are done before the next step
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() 
{
    // Initialize input array
    float h_input[N], h_output[N / BLOCK_SIZE];
    for (int i = 0; i < N; ++i) {
        h_input[i] = i;  
    }

    float* d_input;
    float* d_output;

    int num_blocks = N / BLOCK_SIZE;  // grid size


    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, num_blocks * sizeof(float));

    // Copy input data to device and run kernel
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    parallelReductionSum<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform final reduction on the CPU
    float total_sum = 0.0f;
    for (int i = 0; i < N / BLOCK_SIZE; ++i) {
        total_sum += h_output[i];
    }
    std::cout << "Sum: " << total_sum << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
