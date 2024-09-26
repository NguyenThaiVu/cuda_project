#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

using namespace std;

#define HISTOGRAM_SIZE 256

// Function compute the histogram of an image (width x height)
__global__ void computeHistogram(const unsigned char* input, int* histogram, int width, int height) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        atomicAdd(&histogram[input[idx]], 1);
    }
}

// Function compute the cumulative distribution function (CDF) of histogram
__global__ void computeCDF(int* histogram, float* cdf, int total_pixels) 
{
    __shared__ float local_hist[HISTOGRAM_SIZE];

    int tid = threadIdx.x;
    if (tid < HISTOGRAM_SIZE) {
        local_hist[tid] = histogram[tid];
    }

    __syncthreads();

    // Compute CDF in shared memory
    if (tid == 0) {
        cdf[0] = local_hist[0];
        for (int i = 1; i < HISTOGRAM_SIZE; i++) {
            cdf[i] = cdf[i - 1] + local_hist[i];
        }

        // Normalize the CDF
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            cdf[i] /= total_pixels;
        }
    }
}

// Function apply histogram equalization
__global__ void applyEqualization(unsigned char* input, unsigned char* output, float* cdf, int width, int height) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        output[idx] = static_cast<unsigned char>(255 * cdf[input[idx]]);
    }
}


int main() 
{
    // Load the image using stb_image (grayscale)

    const char* inputImagePath = "image/input_image_gray.jpg";
    const char* outputImagePath = "image/output_hist_equal.jpg";

    int width, height, channels;
    unsigned char* h_inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!h_inputImage) {
        cerr << "Error: Failed to load image!" << endl;
        return -1;
    }

    // Allocate memory for output image
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    // Device memory allocations
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    int* d_histogram;
    float* d_cdf;

    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogram, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc((void**)&d_cdf, HISTOGRAM_SIZE * sizeof(float));

    // Initialize histogram to 0
    cudaMemset(d_histogram, 0, HISTOGRAM_SIZE * sizeof(int));

    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (width * height + blockSize - 1) / blockSize;

    // Step 1: Compute the histogram
    computeHistogram<<<gridSize, blockSize>>>(d_inputImage, d_histogram, width, height);
    cudaDeviceSynchronize();

    // Step 2: Compute CDF from histogram
    computeCDF<<<1, HISTOGRAM_SIZE>>>(d_histogram, d_cdf, width * height);
    cudaDeviceSynchronize();

    // Step 3: Apply histogram equalization using CDF
    applyEqualization<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, d_cdf, width, height);
    cudaDeviceSynchronize();

    // Copy equalized image back to host and save output image
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_png(outputImagePath, width, height, 1, h_outputImage, width);

    // Free memory
    stbi_image_free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_histogram);
    cudaFree(d_cdf);

    cout << "Histogram equalization completed!" << endl;

    return 0;
}
