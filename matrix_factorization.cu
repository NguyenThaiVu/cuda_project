#include <iostream>
#include <cuda_runtime.h>
using namespace std;

/*
This function perform the LU decomposition of a square matrix A.
In this implementation, each thread handles one column.
*/
__global__ void computeLU(float* A, float* L, float* U, int n) {
    int col = threadIdx.x;  
    for (int k = 0; k < n; ++k) {
        // Compute U
        if (col >= k) {
            U[k * n + col] = A[k * n + col];
            for (int p = 0; p < k; ++p) {
                U[k * n + col] -= L[k * n + p] * U[p * n + col];
            }
        }
        __syncthreads();

        // Compute L
        if (col > k) {
            L[col * n + k] = A[col * n + k];
            for (int p = 0; p < k; ++p) {
                L[col * n + k] -= L[col * n + p] * U[p * n + k];
            }
            L[col * n + k] /= U[k * n + k];
        }
        __syncthreads();
    }
}

int main() 
{
    int N = 4;
    // Initialize host matrix memory
    float h_A[N * N] = {
        2.0f, 1.0f, 1.0f, 0.0f,
        4.0f, 3.0f, 3.0f, 1.0f,
        8.0f, 7.0f, 9.0f, 5.0f,
        6.0f, 7.0f, 9.0f, 8.0f,
    };

    float h_L[N * N] = { 0.0f };
    float h_U[N * N] = { 0.0f };

    // Initialize L to have 1s on the diagonal
    for (int i = 0; i < N; ++i) {
        h_L[i * N + i] = 1.0f;
    }

    // Device memory
    float* d_A;
    float* d_L;
    float* d_U;

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_L, N * N * sizeof(float));
    cudaMalloc((void**)&d_U, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, h_L, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U, N * N * sizeof(float), cudaMemcpyHostToDevice);

    computeLU<<<1, N>>>(d_A, d_L, d_U, N);
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(h_L, d_L, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results
    cout << "Matrix L:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << h_L[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "Matrix U:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << h_U[i * N + j] << " ";
        }
        cout << endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    return 0;
}
