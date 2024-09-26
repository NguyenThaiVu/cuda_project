# CUDA Projects

This repository contains various projects demonstrating the use of **CUDA** for high-performance parallel computing. 

<img src="./assert/nvidia_logo.jpg" width="300">


## Table of Contents

1. [Convert RGB to Grayscale](#1-convert-rgb-to-grayscale)
2. [Apply Convolutional Computation](#2-apply-convolutional-computation)
3. [Linear Regression](#3-linear-regression)
4. [Neural Network Inference](#4-neural-network-inference)
5. [Matrix Factorization (Decomposition)](#5-matrix-factorization-decomposition)
6. [Seam Carving](#6-seam-carving)
7. [Histogram Equalization](#7-histogram-equalization)
8. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [How to Run](#how-to-run)


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

### 3. Linear Regression
A CUDA-based implementation of **Linear Regression** using **Stochastic Gradient Descent (SGD)**. 

- **Input**: Data points (features $X$ and targets $y$)
- **Output**: Trained parameters $w$ and $b$ 

### 4. Neural Network Inference
This project performs inference for a neural network using CUDA. The network includes:
- **Input**: Flatten grayscale image (size 784)
- **Hidden Layer**: Size 128 with ReLU activation
- **Output**: Classification probabilities (size 10)


**In the Neural Network Inference project:**
- Train the neural network on the MNIST dataset using the `TensorFlow` library.
- Save the trained weights to a `*.txt` file.
- Use CUDA to load the weights from the file and perform inference with the trained model.


### 5. Matrix Factorization (Decomposition)
This project implements matrix factorization specifically **LU Decomposition**, where matrix $A$ is decomposed into a lower triangular matrix $L$ and an upper triangular matrix $U$.

- **Input**: Matrix $A$
- **Output**: Matrices $L$ and $U$ such that $A = L \times U$


### 6. Seam Carving 
This project demonstrates **seam carving**, a content-aware image resizing algorithm that removes seams (low-importance paths) to resize images while preserving key content. 

- **Input**: RGB or Grayscale image
- **Output**: Resized image with minimal distortion of important content

### 7. Histogram Equalization
This project implements **histogram equalization** to enhance image contrast by redistributing the intensity values of pixels. 
The parallel processing on the GPU makes the histogram computation and intensity mapping highly efficient.

- **Input**: Grayscale image
- **Output**: Contrast-enhanced image

**Detail**:
- Each thread processes one pixel and updates the histogram.
- The CDF is computed in shared memory to avoid multiple global memory accesses.
- When apply histogram equalization, each thread calculate the output pixel.


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
