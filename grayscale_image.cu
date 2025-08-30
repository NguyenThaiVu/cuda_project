#include <iostream>
using namespace std;
#define STB_IMAGE_IMPLEMENTATION
#include </home/thaiv7/Desktop/cuda_project/utils/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include </home/thaiv7/Desktop/cuda_project/utils/stb_image_write.h>
#include <stdio.h>
#include </home/thaiv7/Desktop/cuda_project/utils/gpu_helper.cu>


void convert2GrayImageHost(unsigned char *input_image, unsigned char *output_image,
                                int width, int height, int channel)
{
    for (int col = 0; col < width; col++)
    {
        for (int row = 0; row < height; row++)
        {
            uint8_t r = (uint8_t)input_image[row * width * channel + col * channel + 0];
            uint8_t g = (uint8_t)input_image[row * width * channel + col * channel + 1];
            uint8_t b = (uint8_t)input_image[row * width * channel + col * channel + 2];

            float output_value = 0.299 * r + 0.587 * g + 0.114 * b;
            int index = row * width + col;
            output_image[index] = (unsigned char)output_value;
        }
    }
}

__global__ void convert2GrayImageKernel(unsigned char* input_image, unsigned char* output_image, 
                                        int width, int height, int channel)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < height) & ( col < width))
    {
        uint8_t r = (uint8_t)input_image[row * width * channel + col * channel + 0];
        uint8_t g = (uint8_t)input_image[row * width * channel + col * channel + 1];
        uint8_t b = (uint8_t)input_image[row * width * channel + col * channel + 2];

        float output_value = 0.299 * r + 0.587 * g + 0.114 * b;
        int index = row * width + col;
        output_image[index] = (unsigned char)output_value;
    }
}

void convert2GrayDevice(unsigned char* input_image, unsigned char* output_image, 
                                        int width, int height, int channel)
{
    unsigned char * input_image_d;
    unsigned char * output_image_d;

    cudaMalloc((void **)&input_image_d, width * height * channel * sizeof(unsigned char));
    cudaMalloc((void **)&output_image_d, width * height * 1 * sizeof(unsigned char));

    cudaMemcpy(input_image_d, input_image, width * height * channel * sizeof(unsigned char),
                cudaMemcpyHostToDevice);

    int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 gridDim(int(width/BLOCKSIZE) + 1, int(height/BLOCKSIZE) + 1, 1);

    GpuTimer timer;
    timer.Start();
    convert2GrayImageKernel<<<gridDim, blockDim>>>(input_image_d, output_image_d, 
                                                width, height, channel);
    cudaDeviceSynchronize(); 
    timer.Stop();
    printf("GPU Time: %.3f ms\n", timer.Elapsed());

    cudaMemcpy(output_image, output_image_d, width * height * 1 * sizeof(unsigned char),
            cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaFree(input_image_d);
    cudaFree(output_image_d);
}                                        



int main()
{
    const char *inputImagePath = "/home/thaiv7/Desktop/cuda_project/image/input_image.jpg";
    const char *outputImagePath = "/home/thaiv7/Desktop/cuda_project/image/gray_image.jpg";

    int width, height, channels;

    unsigned char *input_image = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!input_image)
    {
        printf("Error: Failed to load image! %s\n", stbi_failure_reason());
        return -1;
    }

    printf("Loaded image with width: %d, height: %d, channels: %d\n", width, height, channels);
    if (channels != 3)
    {
        printf("Error: Image is not in RGB format!\n");
        stbi_image_free(input_image);
        return -1;
    }

    unsigned char *gray_image = (unsigned char *)malloc(width * height + sizeof(int));

    // convert2GrayImageHost(input_image, gray_image, width, height, channels);
    convert2GrayDevice(input_image, gray_image, width, height, channels);

    // Save the grayscale image
    stbi_write_png(outputImagePath, width, height, 1, gray_image, 0);
    printf("Grayscale image saved!\n");

    stbi_image_free(input_image);

    return 0;
}
