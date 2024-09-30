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


#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define BLOCK_SIZE 128


__device__ float relu(float x) 
{
    return x > 0.0f ? x : 0.0f;
}

// Kernel apply ReLU
__global__ void applyReLU(float* data, int size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = relu(data[tid]);
    }
}

// Kernel to compute the fully connected layer output: Y = X * W + B
__global__ void fullyConnectedLayer(float* input, float* weights, float* bias, float* output, int input_size, int output_size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i) {
            sum += input[i] * weights[tid * input_size + i];
        }
        output[tid] = sum + bias[tid];
    }
}


float random_float(float min=-0.5f, float max=0.5f) 
{
    random_device rd;  
    mt19937 generator(rd());  
    uniform_real_distribution<float> distribution(min, max);  
	return distribution(generator);
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


int main() {
    // Initialize input data
    // float h_input[INPUT_SIZE];
    // for (int i = 0; i < INPUT_SIZE; ++i) {
    //     h_input[i] = random_float();
    // }

    const char* inputImagePath = "image/digit_gray.jpg";  

    int width, height, channels;
    
    unsigned char* input_image = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!input_image) {
        printf("Error: Failed to load image!\n");
        return -1;
    }

    // Allocate memory for float* image (normalized in range 0.0 to 1.0)
    float* h_input = new float[width * height * channels];

    for (int i = 0; i < width * height * channels; ++i) {
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

    // Kernel for hidden layer: Y_hidden = X * W_hidden + B_hidden
    int hidden_blocks = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fullyConnectedLayer<<<hidden_blocks, BLOCK_SIZE>>>(d_input, d_weights_hidden, d_bias_hidden, d_hidden_output, INPUT_SIZE, HIDDEN_SIZE);

    // Apply ReLU to hidden layer
    applyReLU<<<hidden_blocks, BLOCK_SIZE>>>(d_hidden_output, HIDDEN_SIZE);

    // Launch kernel for output layer: Y_output = Y_hidden * W_output + B_output
    int output_blocks = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fullyConnectedLayer<<<output_blocks, BLOCK_SIZE>>>(d_hidden_output, d_weights_output, d_bias_output, d_final_output, HIDDEN_SIZE, OUTPUT_SIZE);

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
