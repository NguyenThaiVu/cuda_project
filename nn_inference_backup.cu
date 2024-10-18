#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <fstream>
using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

#include "utils/gpu_helper.cu"
#include "utils/math_helper.cu"


#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define BLOCK_SIZE 16


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


__device__ float relu(float x) 
{
    return x > 0.0f ? x : 0.0f;
}

__global__ void applyReLU_dense(float* data, int size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = relu(data[tid]);
    }
}


// Kernel to compute the fully connected layer output: Y = X * W + B
__global__ void fullyConnectedLayer(float* input, float* weights, float* bias, float* output, int input_size, int output_size) 
{
    // weights: input_size x output_size

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < output_size) {
        float sum = 0.0f
        ;
        for (int in_idx = 0; in_idx < input_size; in_idx++) 
        {
            sum += input[in_idx] * weights[in_idx * output_size + out_idx];  
        }
        output[out_idx] = sum + bias[out_idx];
    }
}


void softmax(float* input, float* output, int size) 
{
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i]);  
        sum += output[i];  
    }
    
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;  
    }
}


// Function to load weights/bias from a .txt file
void load_weights(const string& filename, float* weight, int size) 
{
    ifstream infile(filename);

    if (infile.is_open()) {
        for (int i = 0; i < size; ++i) {
            infile >> weight[i];
        }
    } else {
        std::cerr << "Could not open file: " << filename << std::endl;
    }

    infile.close();
}


int main() 
{

    // Load input image
    const char* inputImagePath = "image/digit_gray.jpg";  

    int INPUT_WIDTH = 28;
    int INPUT_HEIGHT = 28;
    int INPUT_CHANNELS = 1;
    int INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    
    unsigned char* input_image = stbi_load(inputImagePath, &INPUT_WIDTH, &INPUT_HEIGHT, &INPUT_CHANNELS, 0);
    if (!input_image) {
        printf("Error: Failed to load image!\n");
        return -1;
    }

    // Allocate memory for float* image (normalized in range 0.0 to 1.0)
    float* h_input = new float[INPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++) {
        h_input[i] = input_image[i] / 255.0f;  // Normalize to [0.0, 1.0]
    }

    // Initialize and load weights and biases
    float h_weights_hidden[HIDDEN_SIZE * INPUT_SIZE];  // Weights between input and hidden layer
    float h_bias_hidden[HIDDEN_SIZE];                  // Biases for the hidden layer
    float h_weights_output[OUTPUT_SIZE * HIDDEN_SIZE]; // Weights between hidden and output layer
    float h_bias_output[OUTPUT_SIZE];                  // Biases for the output layer

    load_weights("weight/hidden_weights.txt", h_weights_hidden, HIDDEN_SIZE * INPUT_SIZE);
    load_weights("weight/hidden_bias.txt", h_bias_hidden, HIDDEN_SIZE);
    load_weights("weight/output_weights.txt", h_weights_output, OUTPUT_SIZE * HIDDEN_SIZE);
    load_weights("weight/output_bias.txt", h_bias_output, OUTPUT_SIZE);


    // Allocate device memory
    float *d_input, *d_hidden_output, *d_final_output;
    float *d_weights_hidden, *d_bias_hidden;
    float *d_weights_output, *d_bias_output;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_output, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    cudaMalloc(&d_weights_hidden, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_bias_hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_weights_output, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_bias_output, OUTPUT_SIZE * sizeof(float));

    // Copy input, weights, and biases to device memory
    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_hidden, h_weights_hidden, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_hidden, h_bias_hidden, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_output, h_weights_output, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_output, h_bias_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int dimBlock = BLOCK_SIZE;
    int dimGrid = (HIDDEN_SIZE + BLOCK_SIZE) / BLOCK_SIZE;

    GpuTimer timer;
    timer.Start();

    // 1. Kernel for hidden layer: Y_hidden = X * W_hidden + B_hidden
    fullyConnectedLayer<<<dimBlock, dimGrid>>>(d_input, d_weights_hidden, d_bias_hidden, d_hidden_output, INPUT_SIZE, HIDDEN_SIZE);
    cudaDeviceSynchronize();  

    applyReLU_dense<<<dimBlock, dimGrid>>>(d_hidden_output, HIDDEN_SIZE);
    cudaDeviceSynchronize();  

    // 2. Kernel for output layer: Y_output = Y_hidden * W_output + B_output
    int dimGrid_output = (OUTPUT_SIZE + BLOCK_SIZE) / BLOCK_SIZE;
    fullyConnectedLayer<<<dimBlock, dimGrid_output>>>(d_hidden_output, d_weights_output, d_bias_output, d_final_output, HIDDEN_SIZE, OUTPUT_SIZE);
    cudaDeviceSynchronize();  // Ensure all threads are finished

    timer.Stop();
    printf("\nGPU time: %.3f ms\n", timer.Elapsed());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy output back to host memory
    float h_final_output[OUTPUT_SIZE];
    cudaMemcpy(h_final_output, d_final_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
 
    // Apply softmax to the final output
    float h_softmax_output[OUTPUT_SIZE];
    softmax(h_final_output, h_softmax_output, OUTPUT_SIZE);

    cout << "Softmax output:" << endl;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cout << h_softmax_output[i] << endl;
    }

    cudaFree(d_input);
    cudaFree(d_hidden_output);
    cudaFree(d_final_output);
    cudaFree(d_weights_hidden);
    cudaFree(d_bias_hidden);
    cudaFree(d_weights_output);
    cudaFree(d_bias_output);

    return 0;
}
