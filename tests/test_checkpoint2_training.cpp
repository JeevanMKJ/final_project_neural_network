#include "../include/Matrix.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>

// Test 1: checks MSE loss is accurate
void testLossComputation() {
    std::cout << "Test 1: Loss Function Calculation..." << std::flush;

    // create network
    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 1, activation_type::Sigmoid)
    };
    NeuralNetwork network(architecture);

    // create training data
    std::vector<std::vector<double>> X_train = {
        {0.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> y_train = {
        {0.0},
        {1.0}
    };

    std::vector<double> pred1 = network.predict(X_train[0]);
    std::vector<double> pred2 = network.predict(X_train[1]);

    double loss1 = 0.5 * (pred1[0] - y_train[0][0]) * (pred1[0] - y_train[0][0]);
    double loss2 = 0.5 * (pred2[0] - y_train[1][0]) * (pred2[0] - y_train[1][0]);

    assert(!std::isnan(loss1));
    assert(!std::isnan(loss2));
    assert(!std::isinf(loss1));
    assert(!std::isinf(loss2));
    assert(loss1 >= 0.0);
    assert(loss2 >= 0.0);

    std::cout << " PASSED" << std::endl;
}

// Test 2: tests that loss decreases over epoches/during training 
void testTrainingReducesLoss() {
    std::cout << "Test 2: Training Reduces Loss..." << std::flush;

    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 4, activation_type::ReLU),
        Layer(2, 1, activation_type::Sigmoid)
    };
    NeuralNetwork network(architecture);

    std::vector<std::vector<double>> X_train = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<std::vector<double>> y_train = {
        {0.0},
        {0.0},
        {0.0},
        {1.0}
    };

    double initial_loss = 0.0;
    for (size_t i = 0; i < X_train.size(); ++i) {
        std::vector<double> pred = network.predict(X_train[i]);
        double error = pred[0] - y_train[i][0];
        initial_loss += 0.5 * error * error;
    }
    initial_loss /= X_train.size();

    network.train(X_train, y_train, 0.1, 100, 4, false);

    double final_loss = 0.0;
    for (size_t i = 0; i < X_train.size(); ++i) {
        std::vector<double> pred = network.predict(X_train[i]);
        double error = pred[0] - y_train[i][0];
        final_loss += 0.5 * error * error;
    }
    final_loss /= X_train.size();

    assert(final_loss < initial_loss);
    assert(final_loss < 0.9 * initial_loss);

    std::cout << " PASSED" << std::endl;
    std::cout << "    Initial loss: " << initial_loss << std::endl;
    std::cout << "    Final loss:   " << final_loss << std::endl;
    std::cout << "    Improvement:  " << (1.0 - final_loss/initial_loss) * 100 << "%" << std::endl;
}

// Test 3: check that weights are updated 
void testWeightsUpdate() {
    std::cout << "Test 3: Weights Are Updated After Training..." << std::flush;

    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 3, activation_type::ReLU),
        Layer(2, 1, activation_type::Sigmoid)
    };

    NeuralNetwork network(architecture);

    std::vector<std::vector<double>> X_train = {
        {0.5, 0.5},
        {0.3, 0.7}
    };
    std::vector<std::vector<double>> y_train = {
        {1.0},
        {0.0}
    };

    std::vector<double> pred_before = network.predict(X_train[0]);

    network.train(X_train, y_train, 0.5, 200, 2, false);

    std::vector<double> pred_after = network.predict(X_train[0]);

    assert(std::fabs(pred_before[0] - pred_after[0]) > 1e-6);

    std::cout << " PASSED" << std::endl;
}


int main() {
    std::cout << "\n=== Training & Backpropagation Tests ===" << std::endl;

    try {
        testLossComputation();
        testTrainingReducesLoss();
        testWeightsUpdate();

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
