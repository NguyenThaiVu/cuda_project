# CUDA Projects

This repository contains various projects demonstrating the use of **CUDA** for high-performance parallel computing. 

<img src="./assert/nvidia_logo.jpg" width="300">


## Contents


### 1. Convert RGB to Grayscale
This project uses CUDA to convert an RGB image to grayscale, using the fomular:

```math
\text{Gray} = 0.299 \times R + 0.587 \times G + 0.114 \times B
```

- **Input**: RGB image
- **Output**: Grayscale image

### 2. Apply Convolutional 
This project demonstrates applying convolutional filters on images, including **Gaussian Blur**, **Sharpen**, and **Sobel Edge Detection**.

- **Filters**:
  - **Gaussian Blur**: Smooths the image.
  - **Sharpen**: Enhances edges.
  - **Sobel Edge Detection**: Detects edges in the image.

- **Input**: RGB or Grayscale image
- **Output**: Filtered image

### 3. Parallel Reduction Algorithms
This project demonstrates parallel reduction algorithms, specifically focusing on summing an array efficiently using CUDA

### 4. Linear Regression
A CUDA-based implementation of **Linear Regression** using **Stochastic Gradient Descent (SGD)**. 

- **Input**: Data points (features $X$ and targets $y$)
- **Output**: Trained parameters $w$ and $b$ 

### 5. Neural Network Inference
This project performs inference for a neural network using CUDA. The network includes:
- **Input**: Flatten grayscale image (size 784)
- **Hidden Layer**: Size 128 with ReLU activation
- **Output**: Classification probabilities (size 10)

### 6. Matrix Factorization (Decomposition)
This project implements matrix factorization specifically **LU Decomposition**, where matrix $A$ is decomposed into a lower triangular matrix $L$ and an upper triangular matrix $U$.

- **Input**: Matrix $A$
- **Output**: Matrices $L$ and $U$ such that $A = L \times U$

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
