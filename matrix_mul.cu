#include <iostream>
#include <fstream>
using namespace std;
#include <stdio.h>

void save_to_file(float x[], int size, string filename)
{
    ofstream myfile (filename);
    if (myfile.is_open())
    {
        for(int count = 0; count < size; count ++){
            myfile << x[count] << endl ;
        }
        myfile.close();
    }
    else cout << "Unable to open file";
}


/*
This function performa matrix multiplication C = A*B, where each thread computes one element of matrix C.
A: height_a x width_a
B: height_b x width_b
C: height_c x width_c
*/
__global__ void matrix_multiplication(float* A, float* B, float* C, 
                                    int height_a, int width_a, int height_b, int width_b, int height_c, int width_c)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    printf("row: %d, col: %d\n", row, col);

    if (row < height_c && col < width_c) 
    {
        float value = 0.0f;
        for (int w = 0; w < width_a; w++) 
        {
            value += A[row * width_a + w] * B[w * width_b + col];
        }

        int C_idx = (row * width_c + col);  
        C[C_idx] = value;
    }
}

int main(void) 
{
    int height_a = 50;
    int width_a = 20;
    int height_b = 20;
    int width_b = 30;
    
    int height_c = height_a;
    int width_c = width_b;

    float A[height_a * width_a];
    for (int i = 0; i < height_a*width_a; i++) {
        A[i] = i;
    }

    float B[height_b * width_b];
    for (int i = 0; i < height_b*width_b; i++) {
        B[i] = i;
    }

    float C[height_c * width_c];


    // Device pointers for vectors A, B, and C
    float *d_A, *d_B, *d_C; 

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, height_a * width_a * sizeof(float));
    cudaMalloc((void **)&d_B, height_b * width_b * sizeof(float));
    cudaMalloc((void **)&d_C, height_c * width_c * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, height_a * width_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, height_b * width_b * sizeof(float), cudaMemcpyHostToDevice);    

    int BLOCK_DIM = 16;
    dim3 BLOCK_SIZE(BLOCK_DIM, BLOCK_DIM);
    dim3 GRID_SIZE((width_c + BLOCK_DIM - 1) / BLOCK_DIM, (height_c + BLOCK_DIM - 1) / BLOCK_DIM);

    matrix_multiplication<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, height_a, width_a, height_b, width_b, height_c, width_c);
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(C, d_C, height_c * width_c * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    save_to_file(C, height_c * width_c, "output_matrix_C.txt");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
