#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
using namespace std;



// Function to load weights/bias from a .txt file
void load_weights(const string& filename, float* weight, int size) 
{
    ifstream infile(filename);

    if (infile.is_open()) {
        for (int i = 0; i < size; ++i) {
            infile >> weight[i];
        }
    } else {
        std::cerr << "Could not open file: " << filename << std::endl;
    }

    infile.close();
}




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
