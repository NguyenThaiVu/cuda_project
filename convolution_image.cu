
#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

#include "utils/helper.cu"

#include <iostream>
#include <cuda_runtime.h>
using namespace std;


// Define the convolution kernel. Reference: https://en.wikipedia.org/wiki/Kernel_(image_processing)

// Sobel edge detection
// __constant__ float kernel[3][3] = {
//     {-1, 0, 1},
//     {-2, 0, 2},
//     {-1, 0, 1}
// };

// Sharpen
//__constant__ float kernel[3][3] = {
//    {0, -1, 0},
//    {-1, 5, -1},
//    {0, -1, 0}
//};


/*
Function to apply 2D convolution on an image.
inputImage: Input image
outputImage: Output image
width: Width of input image
height: Height of output image
channels: Number of channels in the image (RGB = 3)
kernel: Convolution kernel
width_kernel: Width of the convolution kernel
*/
__global__ void applyConvolution(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels,
                                 float* kernel, int width_kernel) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;  

    if (col > 0 && col < (width - 1) && row > 0 && row < (height - 1)) 
    {
        for (int c = 0; c < channels; c++) 
        {
            float pixelValue = 0.0f;

            for (int i = 0; i < width_kernel; i++) 
            {
                for (int j = 0; j < width_kernel; j++) 
                {
                    int idx_row_image = row + i;
                    int idx_col_image = col + j;
                    int idx_image = idx_row_image * width * channels + idx_col_image * channels + c;
                    pixelValue += inputImage[idx_image] * kernel[i * width_kernel + j];
                    // pixelValue += inputImage[(pixelY * width + pixelX) * channels + c] * kernel[i + 1][j + 1];
                }
            }

            pixelValue = min(max(pixelValue, 0.0f), 255.0f);  // Clip pixel to [0, 255]
            int idx_output = row * width * channels + col * channels + c;
            outputImage[idx_output] = static_cast<unsigned char>(pixelValue);
        }
    }
}

int main() 
{
    const char* inputImagePath = "image/input_image.jpg";
    const char* outputImagePath = "image/output_image_blur.jpg";

    // Load input image
    int width, height, channels;
    unsigned char* h_inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!h_inputImage) {
        cerr << "Error: Failed to load image!" << endl;
        return -1;
    }

    // Gaussian blur
    int kernel_width = 3;
    float kernel[3 * 3] = {0.0625, 0.125, 0.0625,
                        0.125, 0.25, 0.125,
                        0.0625, 0.125, 0.0625};

    // Allocate output image
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * channels);

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    float *d_kernel;
    cudaMalloc((void**)&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_kernel, kernel_width * kernel_width * sizeof(float));

    cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_width * kernel_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int BLOCK_DIMENSION = 16;
    dim3 blockDim(BLOCK_DIMENSION, BLOCK_DIMENSION);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    GpuTimer timer;
    timer.Start();
    applyConvolution<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, channels, d_kernel, kernel_width);
    cudaDeviceSynchronize();
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    // Copy the result back to the host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_png(outputImagePath, width, height, channels, h_outputImage, width * channels);
    cout << "Convolution applied. Result saved to: " << outputImagePath << endl;

    stbi_image_free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_kernel);

    return 0;
}
