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

#include "utils/in_out_helper.h"
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

Matrix initMatrix(int height, int width) 
{
    Matrix mat;
    mat.height = height;
    mat.width = width;
    mat.stride = width;
    mat.elements = new float[height * width];
    if (mat.elements == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    return mat;
}

void freeMatrix(Matrix* mat) 
{
    free(mat->elements);
    mat->elements = NULL;  
}

Matrix setDimension_Matrix(int height, int width, int stride) 
{
    Matrix mat;
    mat.height = height;
    mat.width = width;
    mat.stride = stride;
    return mat;
}

__device__ float GetElement(Matrix A, int row, int col) 
{
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) 
{
    A.elements[row * A.stride + col] = value;
}


typedef struct {
    int size;
    float* elements;
} Vector;

Vector initVector(int size) {
    Vector vec;
    vec.size = size;
    vec.elements = new float[size];
    if (vec.elements == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    return vec;
}

void freeVector(Vector* vec) {
    free(vec->elements);
    vec->elements = NULL;  
}

Vector setDimension_Vector(int size) {
    Vector vec;
    vec.size = size;
    return vec;
}

__device__ float getVectorElement(Vector vec, int idx) {
    return vec.elements[idx];
}

__global__ void fullyConnectedLayer(Vector input, Matrix weights, Vector bias, Vector output) 
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < output.size) {
        float sum = 0.0f;
        for (int in_idx = 0; in_idx < input.size; in_idx++) 
        {
            sum += getVectorElement(input, in_idx) * GetElement(weights, in_idx, out_idx);  
        }
        output.elements[out_idx] = sum + bias.elements[out_idx];
    }
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

    // Allocate input Vector and normalized in range [0.0 - 1.0]
    Vector input = initVector(INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++) {
        input.elements[i] = input_image[i] / 255.0f;  // Normalize to [0.0, 1.0]
    }

    // Free image
    stbi_image_free(input_image);

    // Initialize and load weights and biases
    Matrix hidden_matrix = initMatrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix output_matrix = initMatrix(HIDDEN_SIZE, OUTPUT_SIZE);

    Vector hidden_bias = initVector(HIDDEN_SIZE);
    Vector output_bias = initVector(OUTPUT_SIZE);

    load_weights("weight/hidden_weights.txt", hidden_matrix.elements, hidden_matrix.height * hidden_matrix.width);
    load_weights("weight/hidden_bias.txt", hidden_bias.elements, HIDDEN_SIZE);
    load_weights("weight/output_weights.txt", output_matrix.elements, output_matrix.height * output_matrix.width);
    load_weights("weight/output_bias.txt", output_bias.elements, OUTPUT_SIZE);


    // Allocate DEVICE memory
    Vector d_input = setDimension_Vector(INPUT_SIZE);
    Matrix d_hidden_matrix = setDimension_Matrix(hidden_matrix.height, hidden_matrix.width, hidden_matrix.stride);
    Vector d_hidden_bias = setDimension_Vector(HIDDEN_SIZE);
    Vector d_hidden_output = setDimension_Vector(HIDDEN_SIZE);

    Matrix d_output_matrix = setDimension_Matrix(output_matrix.height, output_matrix.width, output_matrix.width);
    Vector d_output_bias = setDimension_Vector(OUTPUT_SIZE);
    Vector d_final_output = setDimension_Vector(OUTPUT_SIZE);

    cudaMalloc(&d_input.elements, d_input.size * sizeof(float));
    cudaMalloc(&d_hidden_output.elements, d_hidden_output.size * sizeof(float));
    cudaMalloc(&d_final_output.elements, d_final_output.size * sizeof(float));

    cudaMalloc(&d_hidden_matrix.elements, hidden_matrix.height * hidden_matrix.width * sizeof(float));
    cudaMalloc(&d_hidden_bias.elements, d_hidden_bias.size * sizeof(float));
    cudaMalloc(&d_output_matrix.elements, d_output_matrix.height * d_output_matrix.width * sizeof(float));
    cudaMalloc(&d_output_bias.elements, d_output_bias.size * sizeof(float));

    // Copy input, weights, and biases to DEVICE
    cudaMemcpy(d_input.elements, input.elements, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_hidden_matrix.elements, hidden_matrix.elements, d_hidden_matrix.height * d_hidden_matrix.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_bias.elements, hidden_bias.elements, d_hidden_bias.size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_matrix.elements, output_matrix.elements, d_output_matrix.height * d_output_matrix.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias.elements, output_bias.elements, d_output_bias.size * sizeof(float), cudaMemcpyHostToDevice);

    // Calling GPU kernel
    int dimBlock = BLOCK_SIZE;
    int dimGrid = (HIDDEN_SIZE + BLOCK_SIZE) / BLOCK_SIZE;

    GpuTimer timer;
    timer.Start();

    // 1. Kernel for hidden layer: Y_hidden = X * W_hidden + B_hidden
    fullyConnectedLayer<<<dimBlock, dimGrid>>>(d_input, d_hidden_matrix, d_hidden_bias, d_hidden_output);
    cudaDeviceSynchronize();  

    applyReLU_vector<<<dimBlock, dimGrid>>>(d_hidden_output.elements, HIDDEN_SIZE);
    cudaDeviceSynchronize();  

    // 2. Kernel for output layer: Y_output = Y_hidden * W_output + B_output
    int dimGrid_output = (OUTPUT_SIZE + BLOCK_SIZE) / BLOCK_SIZE;
    fullyConnectedLayer<<<dimBlock, dimGrid_output>>>(d_hidden_output, d_output_matrix, d_output_bias, d_final_output);
    cudaDeviceSynchronize();  // Ensure all threads are finished

    timer.Stop();
    printf("\nGPU time: %.3f ms\n", timer.Elapsed());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy output back to host memory
    float h_final_output[OUTPUT_SIZE];
    cudaMemcpy(h_final_output, d_final_output.elements, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
 
    // Apply softmax to the final output
    float h_softmax_output[OUTPUT_SIZE];
    softmax(h_final_output, h_softmax_output, OUTPUT_SIZE);

    cout << "Softmax output:" << endl;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cout << h_softmax_output[i] << endl;
    }

    cudaFree(d_input.elements);
    cudaFree(d_hidden_output.elements);
    cudaFree(d_final_output.elements);
    cudaFree(d_hidden_matrix.elements);
    cudaFree(d_hidden_bias.elements);
    cudaFree(d_output_matrix.elements);
    cudaFree(d_output_bias.elements);

    freeVector(&input);
    freeMatrix(&hidden_matrix);
    freeMatrix(&output_matrix);
    freeVector(&hidden_bias);
    freeVector(&output_bias);

    return 0;
}
