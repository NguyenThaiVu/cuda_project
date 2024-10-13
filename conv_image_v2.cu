#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

#include "utils/helper.cu"

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#include <stdio.h>

/*
Define helper functions for matrix
*/
typedef struct {
    int width;
    int height;
    int channels;
    int stride;
    unsigned char* elements;
} Tensor_RGB;

__device__ unsigned char GetElement(Tensor_RGB A, int row, int col, int channel)
{
    return A.elements[row * A.stride * A.channels + col * A.channels + channel];
}

__device__ void SetElement(Tensor_RGB A, int row, int col, int channel, unsigned char value)
{
    A.elements[row * A.stride * A.channels + col * A.channels + channel] = value;
}

// Get submatrix of A, which starts from (row, col) with height and width
__device__ Tensor_RGB GetSubTensor(Tensor_RGB A, int row, int col, int height, int width)
{
    Tensor_RGB Asub;
    Asub.width = width;
    Asub.height = height;
    Asub.channels = A.channels;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[row * A.stride * A.channels + col * A.channels];
    return Asub;
}


Tensor_RGB load_rgb_image(const char* filename)
{
    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 0);
    if (image == NULL) {
        cout << "Image is null" << endl;
    }

    Tensor_RGB img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.elements = image;

    return img;
}

void save_rgb_image(const char* filename, Tensor_RGB img)
{
    stbi_write_png(filename, img.width, img.height, img.channels, img.elements, img.width * img.channels);
}

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3
__global__ void apply_2d_conv(Tensor_RGB inputImage, Tensor_RGB outputImage, float* kernel)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;  

    if (col > 0 && col < (inputImage.width - 1) && row > 0 && row < (inputImage.height - 1)) 
    {
        for (int c = 0; c < inputImage.channels; c++) 
        {
            float pixelValue = 0.0f;

            for (int i = 0; i < KERNEL_SIZE; i++) 
            {
                for (int j = 0; j < KERNEL_SIZE; j++) 
                {
                    int idx_row_image = row + i;
                    int idx_col_image = col + j;
                    int idx_image = idx_row_image * inputImage.width * inputImage.channels + idx_col_image * inputImage.channels + c;
                    pixelValue += inputImage.elements[idx_image] * kernel[i * KERNEL_SIZE + j];
                }
            }

            pixelValue = min(max(pixelValue, 0.0f), 255.0f);  // Clip pixel to [0, 255]
            SetElement(outputImage, row, col, c, static_cast<unsigned char>(pixelValue));
        }
    }
}


/*
This function perform 2D convolution on an image.
This version uses shared memory, where each block compute each out_sub
*/
__global__ void apply_2d_conv_v2(Tensor_RGB inputImage, Tensor_RGB outputImage, float* kernel)
{
    int block_row_idx = blockIdx.y;
    int block_col_idx = blockIdx.x;

    // Extract submatrix of input image
    Tensor_RGB input_sub = GetSubTensor(inputImage, 
                                        block_row_idx * BLOCK_SIZE, block_col_idx * BLOCK_SIZE, 
                                        BLOCK_SIZE + KERNEL_SIZE - 1, BLOCK_SIZE + KERNEL_SIZE - 1);

    // // Load submatrix of input image into shared memory
    __shared__ unsigned char shared_inputImage[BLOCK_SIZE + KERNEL_SIZE -1][BLOCK_SIZE + KERNEL_SIZE - 1][3];
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    for (int c = 0; c < inputImage.channels; c++) 
    {
        shared_inputImage[row][col][c] = GetElement(input_sub, row, col, c);
        
        // Handle pixel outside block size
        if (row == BLOCK_SIZE - 1)
        {
            for (int i = 1; i < KERNEL_SIZE; i++)
            {
                shared_inputImage[row + i][col][c] = GetElement(input_sub, row + i, col, c);
            }
        }
        if (col == BLOCK_SIZE - 1)
        {
            for (int i = 1; i < KERNEL_SIZE; i++)
            {
                shared_inputImage[row][col + i][c] = GetElement(input_sub, row, col + i, c);
            }
        }
    }
    __syncthreads();

    // Compute output image
    
    for (int c = 0; c < input_sub.channels; c++) 
    {
        float pixel_value = 0.0f;

        for (int i = 0; i < KERNEL_SIZE; i++) 
        {
            for (int j = 0; j < KERNEL_SIZE; j++) 
            {
                pixel_value += shared_inputImage[row + i][col + j][c] * kernel[i * KERNEL_SIZE + j];
                // pixel_value += GetElement(input_sub, row + i, col + j, c) * kernel[i * KERNEL_SIZE + j];
            }
        }

        
        pixel_value = min(max(pixel_value, 0.0f), 255.0f);  // Clip pixel to [0, 255]
        SetElement(outputImage, block_row_idx * BLOCK_SIZE + row, block_col_idx * BLOCK_SIZE + col, c, 
                    static_cast<unsigned char>(pixel_value));
    }
}


int main()
{
    const char* inputImagePath = "image/input_image.png";
    const char* outputImagePath = "image/output_image_blur.png";

    // Load input image
    Tensor_RGB inputImage = load_rgb_image(inputImagePath);

    // Load input image into device
    Tensor_RGB d_inputImage;
    d_inputImage.width = inputImage.width;
    d_inputImage.height = inputImage.height;
    d_inputImage.channels = inputImage.channels;
    d_inputImage.stride = inputImage.width;
    size_t size_in = d_inputImage.width * d_inputImage.height * d_inputImage.channels;
    cudaMalloc(&d_inputImage.elements, size_in * sizeof(unsigned char));
    cudaMemcpy(d_inputImage.elements, inputImage.elements, size_in * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Allocate output image
    Tensor_RGB outputImage;
    outputImage.width = inputImage.width;
    outputImage.height = inputImage.height;
    outputImage.channels = inputImage.channels;
    outputImage.stride = outputImage.width;
    size_t size_out = outputImage.width * outputImage.height * outputImage.channels;
    outputImage.elements = (unsigned char*)malloc(size_out * sizeof(unsigned char));

    // Allocate output into device 
    Tensor_RGB d_outputImage;
    d_outputImage.width = outputImage.width;
    d_outputImage.height = outputImage.height;
    d_outputImage.channels = outputImage.channels;
    d_outputImage.stride = d_outputImage.width;
    cudaMalloc(&d_outputImage.elements, size_out * sizeof(unsigned char));

    // Define convolution kernel
    float kernel[KERNEL_SIZE * KERNEL_SIZE] = {0.0625, 0.125, 0.0625,
                        0.125, 0.25, 0.125,
                        0.0625, 0.125, 0.0625};

    // Allocate kernel into device
    float* d_kernel;
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((outputImage.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (outputImage.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Call kernel
    GpuTimer timer;
    timer.Start();
    // apply_2d_conv<<<dimGrid, dimBlock>>>(d_inputImage, d_outputImage, d_kernel);
    apply_2d_conv_v2<<<dimGrid, dimBlock>>>(d_inputImage, d_outputImage, d_kernel);
    cudaDeviceSynchronize();
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    
    // Copy output image from device to host
    cudaMemcpy(outputImage.elements, d_outputImage.elements, size_out * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    save_rgb_image(outputImagePath, outputImage);

    // Free memory
    cudaFree(d_inputImage.elements);
    cudaFree(d_outputImage.elements);
    cudaFree(d_kernel);
    free(inputImage.elements);
    free(outputImage.elements);

    return 0;
}