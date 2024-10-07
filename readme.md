# CUDA Projects

This repository contains various projects demonstrating the use of **CUDA** for high-performance parallel computing. 

<img src="./assert/nvidia_logo.jpg" width="300">


## Table of Contents

1. [Apply Convolutional Computation](#2-apply-convolutional-computation)
2. [Linear Regression](#3-linear-regression)
3. [Neural Network Inference](#4-neural-network-inference)
4. [Matrix Factorization (Decomposition)](#5-matrix-factorization-decomposition)
5. [Convolutional Neural Network Inference](#6-convolutional-neural-network-inference)
8. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [How to Run](#how-to-run)


## Contents


### 1. Apply Convolutional 
This project demonstrates applying convolutional filters on images, including **Gaussian Blur**, **Sharpen**, and **Sobel Edge Detection**.

- **Filters**:
  - **Gaussian Blur**: Smooths the image.
  - **Sharpen**: Enhances edges.
  - **Sobel Edge Detection**: Detects edges in the image.

- **Input**: RGB or Grayscale image
- **Output**: Filtered image

### 2. Linear Regression
A CUDA-based implementation of **Linear Regression** using **Stochastic Gradient Descent (SGD)**. 

- **Input**: Data points (features $X$ and targets $y$)
- **Output**: Trained parameters $w$ and $b$ 

### 3. Neural Network Inference
This project performs inference for a neural network using CUDA. The network includes:
- **Input**: Flatten grayscale image (size 784)
- **Hidden Layer**: Size 128 with ReLU activation
- **Output**: Classification probabilities (size 10)


**In the Neural Network Inference project:**
- Train the neural network on the MNIST dataset using the `TensorFlow` library.
- Save the trained weights to a `*.txt` file.
- Use CUDA to load the weights from the file and perform inference with the trained model.


### 4. Matrix Factorization (Decomposition)
This project implements matrix factorization specifically **LU Decomposition**, where matrix $A$ is decomposed into a lower triangular matrix $L$ and an upper triangular matrix $U$.

- **Input**: Matrix $A$
- **Output**: Matrices $L$ and $U$ such that $A = L \times U$


### 5. Convolutional Neural Network Inferenc
This project performs inference for a neural network using CUDA. The network architecture:
- **Input**: Flatten grayscale image (1 x 28 x 28)
- **Conv**: Filter size (32 x 3 x 3)
- **Pooling**: strides = 2, pool_size = 2.
- **Conv_1**: Filter size (64 x 3 x 3)
- **Pooling**: strides = 2, pool_size = 2.
- **Output**: Classification probabilities (size 10)



## Getting Started

### Prerequisites
- Install **CUDA Driver**: [link](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

### How to Run

Navigate to the respective project folder and compile the code using `nvcc`:
   ```bash
   nvcc -o matrix_mul matrix_mul.cu
   ./matrix_mul
   ```

   Replace `matrix_mul.cu` with the desired CUDA file (e.g., `nn_inference.cu` ).
