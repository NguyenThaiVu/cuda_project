#include <iostream>
#include <stdio.h>
#include <cmath>
#include <random>
#include <fstream>
using namespace std;


#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

// Fully connected layer kernel
__global__ void fullyConnectedLayer(float* input, float* weights, float* bias, float* output, int input_size, int output_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            // sum += input[i] * weights[tid * input_size + i];
            sum += input[i] * weights[i * output_size + tid];
        }
        output[tid] = sum + bias[tid];
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
        cerr << "Could not open file: " << filename << endl;
    }

    infile.close();
}


void saveArrayToFile(float* d_array, int size, const std::string& filename) {
    // Allocate host memory for the array
    float* h_array = new float[size];

    // Copy the array from the device (GPU) to the host (CPU)
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Open the file for writing
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        delete[] h_array;  // Free host memory before returning
        return;
    }

    // Write the array elements to the file
    for (int i = 0; i < size; ++i) {
        outfile << h_array[i] << std::endl;  // Each element on a new line
    }

    // Close the file
    outfile.close();

    // Free the host memory
    delete[] h_array;

    std::cout << "Array successfully saved to " << filename << std::endl;
}


// Host code to test the fully connected layer kernel
int main() {
    // Sample input, weights, and bias for a simple test
    const int input_size = 21632;
    const int output_size = 10;
    float h_input[input_size];  // Input vector
    float h_weights[output_size * input_size];
    float h_bias[output_size] = {0.1f, -0.2f};  // Bias for each output neuron

    load_weights("z_output_conv.txt", h_input, input_size);
    load_weights("weight/dense_weights.txt", h_weights, output_size * input_size);
    load_weights("weight/dense_bias.txt", h_bias, output_size);

    // Allocate memory on the device
    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_weights, output_size * input_size * sizeof(float));
    cudaMalloc(&d_bias, output_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid size
    int blockSize = 256;  // Number of threads per block
    int gridSize = (output_size + blockSize - 1) / blockSize;

    // Launch the fully connected layer kernel
    fullyConnectedLayer<<<gridSize, blockSize>>>(d_input, d_weights, d_bias, d_output, input_size, output_size);

    // Synchronize to wait for kernel to finish
    cudaDeviceSynchronize();

    saveArrayToFile(d_output, output_size, "z_output_matmul.txt");

    // Copy result back to host
    float h_output[output_size];
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output
    std::cout << "Fully Connected Layer Output:" << std::endl;
    for (int i = 0; i < output_size; ++i) {
        std::cout << "Output[" << i << "]: " << h_output[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}
