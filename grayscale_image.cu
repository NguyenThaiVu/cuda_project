#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"      
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"  

#include <stdio.h>


__global__ void rgb_to_grayscale(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) 
    {
        int idx = (y * width + x) * channels;  
        int gray_idx = y * width + x;          

        unsigned char r = d_in[idx];       
        unsigned char g = d_in[idx + 1];   
        unsigned char b = d_in[idx + 2];   

        d_out[gray_idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}


int main() 
{
    const char* inputImagePath = "image/input_image.jpg";  
    const char* outputImagePath = "image/output_image.jpg";  

    int width, height, channels;
    
    unsigned char* input_image = stbi_load(inputImagePath, &width, &height, &channels, 0);
    if (!input_image) {
        printf("Error: Failed to load image!\n");
        return -1;
    }

    // Ensure the image is RGB
    printf("Loaded image with width: %d, height: %d, channels: %d\n", width, height, channels); 
    if (channels != 3) {
        printf("Error: Image is not in RGB format!\n");

        stbi_image_free(input_image);
        return -1;
    }

    // Allocate memory on the GPU for input and output
    int imageSize = width * height * channels;
    int grayImageSize = width * height;

    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, imageSize);
    cudaMalloc((void**)&d_out, grayImageSize);

    // Copy the image data from host to device 
    cudaMemcpy(d_in, input_image, imageSize, cudaMemcpyHostToDevice);
    
    // Launch the CUDA kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    rgb_to_grayscale<<<gridDim, blockDim>>>(d_in, d_out, width, height, channels);
    cudaDeviceSynchronize();

    // Copy the grayscale image from device to host
    unsigned char* output_image = (unsigned char*)malloc(grayImageSize);
    cudaMemcpy(output_image, d_out, grayImageSize, cudaMemcpyDeviceToHost);

    // Save the grayscale image
    stbi_write_png(outputImagePath, width, height, 1, output_image, width);

    printf("Grayscale image saved!\n");

    // Free the memory
    stbi_image_free(input_image);
    free(output_image);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
