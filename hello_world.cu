#include <stdio.h>

__global__ void hello_from_gpu(void) 
{
    printf("Hello from GPU! Block ID: %d, Thread ID: %d\n", blockIdx.x, threadIdx.x);
}

int main(void) {

    int num_blocks = 2;
    int threads_per_block = 5;
    hello_from_gpu<<<num_blocks, threads_per_block>>>();

    cudaDeviceSynchronize();

    return 0;
}
