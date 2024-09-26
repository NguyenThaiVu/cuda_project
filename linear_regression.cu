#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
using namespace std;

/*
This function perform a single step of SGD
d_x: Input data points.
d_y: Output data points.
d_w: Weight.
d_b: Bias.
N: Number of data points.
alpha: Learning rate.
*/
__global__ void sgdStep(float* d_x, float* d_y, float* d_w, float* d_b, int N, float alpha) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handle edge case
    if (idx < N) 
    {
        float x = d_x[idx];
        float y = d_y[idx];

        float y_pred = (*d_w) * x + (*d_b);

        // Compute gradient for w and b
        float error = y_pred - y;
        float grad_w = 2 * error * x;
        float grad_b = 2 * error;

        // Update the weights, atomicAdd to avoid error
        atomicAdd(d_w, -alpha * grad_w); 
        atomicAdd(d_b, -alpha * grad_b);
    }
}

float random_float(float min=-0.05f, float max=0.05f) 
{
    random_device rd;  
    mt19937 generator(rd());  
    uniform_real_distribution<float> distribution(min, max);  
	return distribution(generator);
}

void create_toy_dataset(float* x, float* y, int N) 
{
    for (int i = 0; i < N; ++i) 
    {
        x[i] = random_float(-2.0f, 2.0f);
        y[i] = 2.0f * x[i] + 0.1f + random_float(-0.05f, 0.05f);
    }
}


int main() 
{
    // Create toy data
    const int N = 100;  
    float h_x[N];
    float h_y[N];
    create_toy_dataset(h_x, h_y, N);

    // Initial values of w and b
    float h_w = 0.0f; 
    float h_b = 0.0f; 
    float alpha = 0.005;
    int num_epochs = 100;

    // Allocate memory on device 
    float *d_x, *d_y, *d_w, *d_b;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Perform SGD steps
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        sgdStep<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_w, d_b, N, alpha);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(&h_w, d_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Trained weight (w): " << h_w << std::endl;
    std::cout << "Trained bias (b): " << h_b << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_b);

    return 0;
}
