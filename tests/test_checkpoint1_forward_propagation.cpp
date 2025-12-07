#include "../include/Matrix.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>

// Test 1: tests forward propagation
void testMatrixVectorMultiplication() {
    std::cout << "Test 1: Matrix-Vector Multiplication..." << std::flush;

    Matrix W(3, 2);
    W.set_value(0, 0, 1.0);
    W.set_value(0, 1, 2.0);
    W.set_value(1, 0, 3.0);
    W.set_value(1, 1, 4.0);
    W.set_value(2, 0, 5.0);
    W.set_value(2, 1, 6.0);

    std::vector<double> input = {1.0, 2.0};

    // Perform dot product
    std::vector<double> result = W.dot_product(input);

    assert(result.size() == 3);
    assert(std::fabs(result[0] - 5.0) < 1e-6);
    assert(std::fabs(result[1] - 11.0) < 1e-6);
    assert(std::fabs(result[2] - 17.0) < 1e-6);

    std::cout << " PASSED" << std::endl;
}

// Test 2: tests ReLU and Sigmoid activation functions
void testActivationFunctions() {
    std::cout << "Test 2: Activation Functions..." << std::flush;

    // ReLU activation
    Layer relu_layer(1, 3, activation_type::ReLU);

    // ReLU(2.0) should be 2.0 
    assert(std::fabs(relu_layer.activate(2.0) - 2.0) < 1e-6);

    // ReLU(-1.0) should be 0.0 
    assert(std::fabs(relu_layer.activate(-1.0) - 0.0) < 1e-6);

    // ReLU(0.0) should be 0.0
    assert(std::fabs(relu_layer.activate(0.0) - 0.0) < 1e-6);

    // Sigmoid activation
    Layer sigmoid_layer(2, 1, activation_type::Sigmoid);

    // Sigmoid(0.0) should be 0.5
    assert(std::fabs(sigmoid_layer.activate(0.0) - 0.5) < 1e-6);

    // Sigmoid should approach 1.0
    assert(sigmoid_layer.activate(10.0) > 0.99);

    // Sigmoid should approach 0.0
    assert(sigmoid_layer.activate(-10.0) < 0.01);

    std::cout << " PASSED" << std::endl;
}

// Test 3: tests forward propagation
void testForwardPropagation() {
    std::cout << "Test 3: Forward Propagation..." << std::flush;

    // 2-layer network: 2 -> 3 -> 1
    // input layer (2 features)
    // hidden layer (3 neurons, ReLU)
    // output layer (1 neuron, Sigmoid)
    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),     
        Layer(1, 3, activation_type::ReLU),     
        Layer(2, 1, activation_type::Sigmoid)   
    };

    NeuralNetwork network(architecture);

    // input
    std::vector<double> input = {0.5, 0.5};

    // forward propagation
    std::vector<double> output = network.forward(input);

    // output should have 1 element 
    assert(output.size() == 1);

    // sigmoid output should be between 0 and 1
    assert(output[0] >= 0.0 && output[0] <= 1.0);

    // network should be, same input -> same output
    std::vector<double> output2 = network.forward(input);
    assert(std::fabs(output[0] - output2[0]) < 1e-6);

    std::cout << " PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Forward Propagation Tests ===" << std::endl;

    try {
        testMatrixVectorMultiplication();
        testActivationFunctions();
        testForwardPropagation();

        std::cout << "\nAll tests passed!" << std::endl;
        std::cout << "=== Test Summary ===" << std::endl;
        std::cout << "Total Tests: 3" << std::endl;
        std::cout << "Passed: 3" << std::endl;
        std::cout << "Failed: 0" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
