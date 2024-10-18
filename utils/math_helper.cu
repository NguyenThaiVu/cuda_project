#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <fstream>
using namespace std;


float random_float(float min=-0.5f, float max=0.5f) 
{
    random_device rd;  
    mt19937 generator(rd());  
    uniform_real_distribution<float> distribution(min, max);  
	return distribution(generator);
}


void softmax(float* input, float* output, int size) 
{
    float sum = 0.0f;
    
    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i]);  
        sum += output[i];  
    }
    
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;  
    }
}

