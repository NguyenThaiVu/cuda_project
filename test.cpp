#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

int main() {
    // Load the JSON file with model weights
    std::ifstream json_file("model_weights.json");
    if (!json_file.is_open()) {
        std::cerr << "Error: Could not open file model_weights.json" << std::endl;
        return -1;
    }

    json weights_json;
    json_file >> weights_json;

    // Convert JSON data to C++ vectors
    std::vector<std::vector<float>> layer1_weights;
    std::vector<std::vector<float>> layer2_weights;

    // Assuming the first layer weights and biases are stored in the first two elements
    for (const auto& layer_weights : weights_json[0]) {
        layer1_weights.push_back(layer_weights);
    }
    
    std::vector<float> layer1_biases = weights_json[1].get<std::vector<float>>();

    // Assuming the second layer weights and biases are stored in the next two elements
    for (const auto& layer_weights : weights_json[2]) {
        layer2_weights.push_back(layer_weights);
    }
    
    std::vector<float> layer2_biases = weights_json[3].get<std::vector<float>>();

    // Example: Print first 5 weights from the first layer
    std::cout << "First 5 weights from Layer 1:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << layer1_weights[i][0] << " ";
    }
    std::cout << std::endl;

    return 0;
}
