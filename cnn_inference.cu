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

const float FLT_MAX = 3.402823466e+38F;


#define CUDA_CHECK(call) \
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

void printDeviceInfo() 
{
    cudaDeviceProp devProv;
    CUDA_CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
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

    outfile.close();

    delete[] h_array;

    std::cout << "Array successfully saved to " << filename << std::endl;
}


/*
This project demonstrade the 2D convolution operation using CUDA.
We will use the Channel-last format in the tensors computation. This make consistent with TensorFlow library.
*/

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

__global__ void applyReLU_2D(float* data, int outputChannels, int outputHeight, int outputWidth) 
{
    // Calculate the row and column index that this thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Apply ReLU to each channel for the current (row, col) position
    if (row < outputHeight && col < outputWidth) 
    {
        for (int c = 0; c < outputChannels; c++) 
        {
            int index = c * (outputHeight * outputWidth) + row * outputWidth + col;
            data[index] = relu(data[index]);
        }
    }
}

__global__ void fullyConnectedLayer(float* input, float* weights, float* bias, float* output, int input_size, int output_size) 
{
    // weights: input_size x output_size

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < output_size) {
        float sum = 0.0f
        ;
        for (int in_idx = 0; in_idx < input_size; in_idx++) 
        {
            // sum += input[i] * weights[tid * input_size + i];  This is WX
            sum += input[in_idx] * weights[in_idx * output_size + out_idx];  // This is XW 
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
        cerr << "Could not open file: " << filename << endl;
    }

    infile.close();
}



// CUDA kernel for 2D convolution
/*
This function implement the 2D convolution operation.
The input tensor has dimensions (CxHxW)
The kernel has dimensions (outputChannels x  inputChannels x K x K)
The output tensor has dimensions (H-K+1 x W-K+1 x outputChannels)
*/
__global__ void conv2d(float *input, float *kernel, float *output, 
                       int inputHeight, int inputWidth, int inputChannels, 
                       int kernelSize, int outputChannels) 
{
    // Calculate the row and column index of the element the thread is handling
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int outputHeight = inputHeight - kernelSize + 1;
    int outputWidth = inputWidth - kernelSize + 1;

    // Ensure thread is within the bounds of the output tensor
    if (row < outputHeight && col < outputWidth) 
    {
        // Loop over all output channels
        for (int oc = 0; oc < outputChannels; oc++) 
        {
            float outputValue = 0.0f;

            // Loop over all input channels (depth)
            for (int ic = 0; ic < inputChannels; ic++) 
            {
                // Perform convolution for the (row, col) position
                for (int kh = 0; kh < kernelSize; kh++) 
                {
                    for (int kw = 0; kw < kernelSize; kw++) 
                    {
                        int inputRow = row + kh;
                        int inputCol = col + kw;

                        if (inputRow < inputHeight && inputCol < inputWidth)  // Boundary check
                        {  
                            // 3D index for input tensor (C x H x W) flattened
                            int inputIndex = ic * inputHeight * inputWidth + inputRow * inputWidth + inputCol;

                            // 4D index for kernel tensor (outputChannels x inputChannels x K x K)
                            int kernelIndex = oc * inputChannels * kernelSize * kernelSize + ic * kernelSize * kernelSize + kh * kernelSize + kw;

                            // Accumulate the convolution result
                            outputValue += input[inputIndex] * kernel[kernelIndex];
                        }
                    }
                }
            }

            // Store the result in the output tensor (flattened 3D array H_out x W_out x C_out)
            int outputIndex = oc * outputHeight * outputWidth + row * outputWidth + col;
            output[outputIndex] = outputValue;
        }
    }
}


__global__ void maxPooling2D(float* input, float* output, 
                             int inputHeight, int inputWidth, int inputChannels, 
                             int poolHeight, int poolWidth, int stride) 
{
    // Calculate the row and column index of the element the thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Output dimensions
    int outputHeight = (inputHeight - poolHeight) / stride + 1;
    int outputWidth = (inputWidth - poolWidth) / stride + 1;

    // Ensure thread is within bounds of the output tensor
    if (row < outputHeight && col < outputWidth) {
        // Loop over all input channels
        for (int c = 0; c < inputChannels; c++) {
            float maxValue = -FLT_MAX;  // Initialize to the smallest possible float

            // Find the maximum value in the pooling window
            for (int ph = 0; ph < poolHeight; ph++) {
                for (int pw = 0; pw < poolWidth; pw++) {
                    int inputRow = row * stride + ph;
                    int inputCol = col * stride + pw;

                    // Compute the index for input tensor
                    int inputIndex = c * inputHeight * inputWidth + inputRow * inputWidth + inputCol;

                    if (input[inputIndex] > maxValue) {
                        maxValue = input[inputIndex];
                    }
                }
            }

            // Store the result in the output tensor
            int outputIndex = c * outputHeight * outputWidth + row * outputWidth + col;
            output[outputIndex] = maxValue;
        }
    }
}


int main() 
{

    // printDeviceInfo();
    
    // Load input image
    const char* inputImagePath = "image/digit_gray.jpg";  

    int INPUT_WIDTH = 28;
    int INPUT_HEIGHT = 28;
    int INPUT_CHANNELS = 1;
    
    unsigned char* image = stbi_load(inputImagePath, &INPUT_WIDTH, &INPUT_HEIGHT, &INPUT_CHANNELS, 0);
    if (!image) {
        printf("Error: Failed to load image!\n");
        return -1;
    }

    // Allocate space for "channels first" format
    unsigned char input_image [INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    for (int c = 0; c < INPUT_CHANNELS; c++) {
        for (int h = 0; h < INPUT_HEIGHT; h++) {
            for (int w = 0; w < INPUT_WIDTH; w++) {
                // From channels last: image[h * width * channels + w * channels + c]
                // To channels first:  image_channel_first[c * height * width + h * width + w]
                input_image[c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w] = image[h * INPUT_WIDTH * INPUT_CHANNELS + w * INPUT_CHANNELS + c];
            }
        }
    }

    // Free the original image memory
    stbi_image_free(image);

    const int INPUT_SIZE = INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;

    // Convolutional layer
    const int KERNEL_SIZE = 3;
    const int OUTPUT_CONV_CHANNELS = 32;
    const int OUTPUT_CONV_WIDTH = INPUT_WIDTH - KERNEL_SIZE + 1;
    const int OUTPUT_CONV_HEIGHT = INPUT_HEIGHT - KERNEL_SIZE + 1;

    // Max pooling layer
    const int POOL_HEIGHT = 2;
    const int POOL_WIDTH = 2;
    const int POOL_STRIDE = 2;
    int OUTPUT_POOL_WIDTH = (OUTPUT_CONV_WIDTH - POOL_WIDTH) / POOL_STRIDE + 1;
    int OUTPUT_POOL_HEIGHT = (OUTPUT_CONV_HEIGHT - POOL_HEIGHT) / POOL_STRIDE + 1;
    
    // Convolutional layer
    int KERNEL_CONV_1_SIZE = 3;
    int OUTPUT_CONV_1_CHANNELS = 64;
    int OUTPUT_CONV_1_WIDTH = OUTPUT_POOL_WIDTH - KERNEL_CONV_1_SIZE + 1;
    int OUTPUT_CONV_1_HEIGHT = OUTPUT_POOL_HEIGHT - KERNEL_CONV_1_SIZE + 1;

    // Max pooling layer
    int POOL_1_WIDTH = 2;
    int POOL_1_HEIGHT = 2;
    int POOL_1_STRIDE = 2;
    int OUTPUT_POOL_1_WIDTH = (OUTPUT_CONV_1_WIDTH - POOL_1_WIDTH) / POOL_1_STRIDE + 1;
    int OUTPUT_POOL_1_HEIGHT = (OUTPUT_CONV_1_HEIGHT - POOL_1_HEIGHT) / POOL_1_STRIDE + 1;

    // Fully connected layer
    const int HIDDEN_SIZE = 64*5*5;
    const int OUTPUT_SIZE = 10;

    
    // Allocate memory for float* image (normalized in range 0.0 to 1.0)
    float* h_input = new float[INPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++) 
    {
        h_input[i] = input_image[i] / 255.0f;  // Normalize to [0.0, 1.0]
    }

    // Initialize and load weights and biases
    float h_conv_weights[KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * OUTPUT_CONV_CHANNELS];  
    float h_conv_1_weights[KERNEL_CONV_1_SIZE * KERNEL_CONV_1_SIZE * OUTPUT_CONV_CHANNELS * OUTPUT_CONV_1_CHANNELS];
    float h_weights_output[OUTPUT_SIZE * HIDDEN_SIZE]; 
    float h_bias_output[OUTPUT_SIZE];                  

    load_weights("weight/conv2d_weights.txt", h_conv_weights, KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * OUTPUT_CONV_CHANNELS);
    load_weights("weight/conv2d_1_weights.txt", h_conv_1_weights, KERNEL_CONV_1_SIZE * KERNEL_CONV_1_SIZE * OUTPUT_CONV_CHANNELS * OUTPUT_CONV_1_CHANNELS);
    load_weights("weight/dense_weights.txt", h_weights_output, OUTPUT_SIZE * HIDDEN_SIZE);
    load_weights("weight/dense_bias.txt", h_bias_output, OUTPUT_SIZE);
  

    // Allocate device memory for input and output tensors
    float *d_input, *d_conv_output, *d_pool_output, *d_conv_1_output, *d_pool_1_output, *d_final_output;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_conv_output, OUTPUT_CONV_CHANNELS * OUTPUT_CONV_WIDTH * OUTPUT_CONV_HEIGHT * sizeof(float));
    cudaMalloc(&d_pool_output, OUTPUT_CONV_CHANNELS * OUTPUT_POOL_WIDTH * OUTPUT_POOL_HEIGHT * sizeof(float));
    cudaMalloc(&d_conv_1_output, OUTPUT_CONV_1_CHANNELS * OUTPUT_CONV_1_WIDTH * OUTPUT_CONV_1_HEIGHT * sizeof(float));
    cudaMalloc(&d_pool_1_output, OUTPUT_CONV_1_CHANNELS * OUTPUT_POOL_1_WIDTH * OUTPUT_POOL_1_HEIGHT * sizeof(float));
    cudaMalloc(&d_final_output, OUTPUT_SIZE * sizeof(float));

    // Allocate device memory for weights and biases
    float *d_conv_weight, *d_conv_1_weight;;
    float *d_weights_output, *d_bias_output;
    cudaMalloc(&d_conv_weight, KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * OUTPUT_CONV_CHANNELS * sizeof(float));
    cudaMalloc(&d_conv_1_weight, OUTPUT_CONV_1_CHANNELS * OUTPUT_CONV_CHANNELS * KERNEL_CONV_1_SIZE * KERNEL_CONV_1_SIZE * sizeof(float));
    cudaMalloc(&d_weights_output, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_bias_output, OUTPUT_SIZE * sizeof(float));

    // Copy input, weights, and biases to device memory
    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_weight, h_conv_weights, KERNEL_SIZE * KERNEL_SIZE * INPUT_CHANNELS * OUTPUT_CONV_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_1_weight, h_conv_1_weights, OUTPUT_CONV_1_CHANNELS * OUTPUT_CONV_CHANNELS * KERNEL_CONV_1_SIZE * KERNEL_CONV_1_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_output, h_weights_output, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_output, h_bias_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);


    // Define block size
    int BLOCK_SIZE = 8;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridSize((OUTPUT_CONV_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_CONV_HEIGHT + blockSize.y - 1) / blockSize.y);
    conv2d<<<gridSize, blockSize>>>(d_input, d_conv_weight, d_conv_output, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, KERNEL_SIZE, OUTPUT_CONV_CHANNELS);
    CUDA_CHECK(cudaDeviceSynchronize());

    applyReLU_2D<<<gridSize, blockSize>>>(d_conv_output, OUTPUT_CONV_CHANNELS, OUTPUT_CONV_HEIGHT, OUTPUT_CONV_WIDTH);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch max pooling kernel
    dim3 poolGridSize((OUTPUT_POOL_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_POOL_HEIGHT + blockSize.y - 1) / blockSize.y);
    maxPooling2D<<<poolGridSize, blockSize>>>(d_conv_output, d_pool_output, OUTPUT_CONV_HEIGHT, OUTPUT_CONV_WIDTH, OUTPUT_CONV_CHANNELS,
                                                                            POOL_HEIGHT, POOL_WIDTH, POOL_STRIDE);
    cudaDeviceSynchronize();  // Ensure all threads are finished

    // Convolutional layer 2
    dim3 gridSize_1((OUTPUT_POOL_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_POOL_HEIGHT + blockSize.y - 1) / blockSize.y);
    conv2d<<<gridSize_1, blockSize>>>(d_pool_output, d_conv_1_weight, d_conv_1_output,
                                     OUTPUT_POOL_HEIGHT, OUTPUT_POOL_WIDTH, OUTPUT_CONV_CHANNELS, KERNEL_CONV_1_SIZE, OUTPUT_CONV_1_CHANNELS);
    CUDA_CHECK(cudaDeviceSynchronize());

    applyReLU_2D<<<gridSize_1, blockSize>>>(d_conv_1_output, OUTPUT_CONV_1_CHANNELS, OUTPUT_CONV_1_HEIGHT, OUTPUT_CONV_1_WIDTH);

    // Launch max pooling kernel
    dim3 poolGridSize_1((OUTPUT_POOL_1_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_POOL_1_HEIGHT + blockSize.y - 1) / blockSize.y);
    maxPooling2D<<<poolGridSize_1, blockSize>>>(d_conv_1_output, d_pool_1_output, OUTPUT_CONV_1_HEIGHT, OUTPUT_CONV_1_WIDTH, OUTPUT_CONV_1_CHANNELS,
                                                                            POOL_1_HEIGHT, POOL_1_WIDTH, POOL_1_STRIDE);

    // Launch FC
    int output_blocks = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // fullyConnectedLayer<<<output_blocks, BLOCK_SIZE>>>(d_conv_output, d_weights_output, d_bias_output, d_final_output, HIDDEN_SIZE, OUTPUT_SIZE);
    fullyConnectedLayer<<<output_blocks, BLOCK_SIZE>>>(d_pool_1_output, d_weights_output, d_bias_output, d_final_output, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy output back to host memory
    float h_final_output[OUTPUT_SIZE];
    cudaMemcpy(h_final_output, d_final_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
 
    // // Apply softmax to the final output
    float h_softmax_output[OUTPUT_SIZE];
    softmax(h_final_output, h_softmax_output, OUTPUT_SIZE);

    cout << "Softmax output:" << endl;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cout << h_softmax_output[i] << endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_conv_weight);
    cudaFree(d_conv_output);

    cudaFree(d_final_output);
    cudaFree(d_weights_output);
    cudaFree(d_bias_output);

    return 0;
}