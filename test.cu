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
    unsigned char* elements;
} Tensor_RGB;

unsigned char GetElement(Tensor_RGB A, int row, int col, int channel)
{
    return A.elements[row * A.width * A.channels + col * A.channels + channel];
}

void SetElement(Tensor_RGB A, int row, int col, int channel, unsigned char value)
{
    A.elements[row * A.width * A.channels + col * A.channels + channel] = value;
}

// Get submatrix of A, which starts from (row, col) with height and width
Tensor_RGB GetSubTensor(Tensor_RGB A, int row, int col, int height, int width)
{
    Tensor_RGB Asub;
    Asub.width = width;
    Asub.height = height;
    Asub.channels = A.channels;
    Asub.elements = &A.elements[row * A.width * A.channels + col * A.channels];
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

int main()
{
    Tensor_RGB img = load_rgb_image("random_image.png");
    cout << "Image width: " << img.width << endl;
    cout << "Image height: " << img.height << endl;
    cout << "Image channels: " << img.channels << endl;

    unsigned char pixel;

    for (int i=0; i< img.height; i++)
    {
        for (int j=0; j< img.width; j++)
        {
            for (int c = 0; c < img.channels; c++)
            {
                pixel = GetElement(img, i, j, c);
                cout << "Pixel value at (" << i << ", " << j << ", " << c << "): " << (int)pixel << endl;
            }
        }
    }


    return 0;
}