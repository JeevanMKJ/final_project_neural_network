#include "../include/Matrix.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sys/stat.h>

// helper function checks if file exists
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Test 1: checks if save_model() saved/created the model/file
void testSaveCreatesFile() {
    std::cout << "Test 1: Save Model Creates File..." << std::flush;

    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 3, activation_type::ReLU),
        Layer(2, 1, activation_type::Sigmoid)
    };
    NeuralNetwork network(architecture);

    std::string filename = "test_model_checkpoint3.csv";
    network.save_model(filename);

    assert(fileExists(filename));

    std::ifstream file(filename);
    assert(file.good());

    std::string first_line;
    std::getline(file, first_line);
    assert(!first_line.empty());

    file.close();

    std::remove(filename.c_str());

    std::cout << " PASSED" << std::endl;
}

// Test 2: save trained model, loads it and verifies the predictions
void testIdenticalPredictions() {
    std::cout << "Test 2: Loaded Model Produces Identical Predictions..." << std::flush;

    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 4, activation_type::ReLU),
        Layer(2, 1, activation_type::Sigmoid)
    };
    NeuralNetwork network1(architecture);

    std::vector<std::vector<double>> X_train = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<std::vector<double>> y_train = {
        {0.0}, {0.0}, {0.0}, {1.0}
    };
    network1.train(X_train, y_train, 0.5, 300, 4, false);

    std::vector<double> pred1_orig = network1.predict({0.0, 0.0});
    std::vector<double> pred2_orig = network1.predict({0.0, 1.0});
    std::vector<double> pred3_orig = network1.predict({1.0, 0.0});
    std::vector<double> pred4_orig = network1.predict({1.0, 1.0});

    std::string filename = "test_model_and.csv";
    network1.save_model(filename);

    NeuralNetwork network2(architecture);
    network2.load_model(filename);

    std::vector<double> pred1_loaded = network2.predict({0.0, 0.0});
    std::vector<double> pred2_loaded = network2.predict({0.0, 1.0});
    std::vector<double> pred3_loaded = network2.predict({1.0, 0.0});
    std::vector<double> pred4_loaded = network2.predict({1.0, 1.0});

    assert(std::fabs(pred1_orig[0] - pred1_loaded[0]) < 1e-9);
    assert(std::fabs(pred2_orig[0] - pred2_loaded[0]) < 1e-9);
    assert(std::fabs(pred3_orig[0] - pred3_loaded[0]) < 1e-9);
    assert(std::fabs(pred4_orig[0] - pred4_loaded[0]) < 1e-9);

    std::remove(filename.c_str());

    std::cout << " PASSED" << std::endl;
}

// Test 3: tests if loaded model has same performance as orignial
void testPreservesTrainingQuality() {
    std::cout << "Test 3: Save/Load Preserves Training Quality..." << std::flush;

    std::vector<Layer> architecture = {
        Layer(0, 2, activation_type::None),
        Layer(1, 4, activation_type::ReLU),
        Layer(2, 1, activation_type::Sigmoid)
    };
    NeuralNetwork network1(architecture);

    std::vector<std::vector<double>> X_train = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    std::vector<std::vector<double>> y_train = {
        {0.0}, {0.0}, {0.0}, {1.0}
    };

    network1.train(X_train, y_train, 0.5, 400, 4, false);

    double loss_original = 0.0;
    for (size_t i = 0; i < X_train.size(); ++i) {
        std::vector<double> pred = network1.predict(X_train[i]);
        double error = pred[0] - y_train[i][0];
        loss_original += 0.5 * error * error;
    }
    loss_original /= X_train.size();

    std::string filename = "test_model_quality.csv";
    network1.save_model(filename);

    NeuralNetwork network2(architecture);
    network2.load_model(filename);

    double loss_loaded = 0.0;
    for (size_t i = 0; i < X_train.size(); ++i) {
        std::vector<double> pred = network2.predict(X_train[i]);
        double error = pred[0] - y_train[i][0];
        loss_loaded += 0.5 * error * error;
    }
    loss_loaded /= X_train.size();

    assert(std::fabs(loss_original - loss_loaded) < 1e-9);

    assert(loss_original < 0.1);
    assert(loss_loaded < 0.1);

    std::remove(filename.c_str());

    std::cout << " PASSED" << std::endl;
    std::cout << "    Original loss: " << loss_original << std::endl;
    std::cout << "    Loaded loss:   " << loss_loaded << std::endl;
}

int main() {
    std::cout << "\n=== Checkpoint 3: Model Persistence Tests ===" << std::endl;

    try {
        testSaveCreatesFile();
        testIdenticalPredictions();
        testPreservesTrainingQuality();

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
