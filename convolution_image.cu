
#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

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

// Gaussian blur
__constant__ float kernel[3][3] = {
   {0.0625, 0.125, 0.0625},
   {0.125, 0.25, 0.125},
   {0.0625, 0.125, 0.0625}
};



__global__ void applyConvolution(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  

    // Apply convolution only if the pixel is within bounds (excluding boundary)
    if (x > 0 && x < (width - 1) && y > 0 && y < (height - 1)) {
        for (int c = 0; c < channels; c++) {
            float pixelValue = 0.0f;

            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int pixelX = x + j;
                    int pixelY = y + i;
                    pixelValue += inputImage[(pixelY * width + pixelX) * channels + c] * kernel[i + 1][j + 1];
                }
            }

            // Clip pixel values to [0, 255]
            pixelValue = min(max(pixelValue, 0.0f), 255.0f);
            outputImage[(y * width + x) * channels + c] = static_cast<unsigned char>(pixelValue);
        }
    }
}

int main() 
{
    const char* inputImagePath = "image/input_image.jpg";
    const char* outputImagePath = "image/output_kernel_image.jpg";

    int width, height, channels;
    
    unsigned char* h_inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!h_inputImage) {
        cerr << "Error: Failed to load image!" << endl;
        return -1;
    }

    // Allocate output image
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * channels);

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * channels * sizeof(unsigned char));

    cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    applyConvolution<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, channels);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_png(outputImagePath, width, height, channels, h_outputImage, width * channels);
    cout << "Convolution applied. Result saved to: " << outputImagePath << endl;

    stbi_image_free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
