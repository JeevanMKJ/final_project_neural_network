#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include "Layer.h"
#include "Matrix.h"

// NeuralNetwork class manages entire neural network (layers, weights, training and prediction)
// Responsible for forward of inputs, back propigation errors, updating of weights and biases.
class NeuralNetwork {
private:
    // Vector of Layer objects (represents architecture of network)
    std::vector<Layer> layers;
    // Vector of Matrix objects (represents weights connecting layers)
    std::vector<Matrix> weights;

    // Connets layers of network
    void connect_layers();

    // Initializes weights matrices for connection between layers
    void initializeWeights();

public:
    // Constructor:
    // takes in: vector of layer objects (defines architecture of network)
    NeuralNetwork(const std::vector<Layer>& network_layers);

    // Responsible for forward propigation:
    // takes in: vector of doubles (represents input to network)
    // returns: vector of doubles (represents output of network)
    std::vector<double> forward(const std::vector<double>& input);

    // Responsible for predictions:
    // takes in: vector of doubles (represents input to network)
    // returns: vector of doubles (represnts network's prediction)
    std::vector<double> predict(const std::vector<double>& input);

    // Responsible for training network with input dataset and targets (backpropagation), takes in:
    // 1. vector of input vectors
    // 2. vector of target vectors
    // 3. learning_rate, steps size for updating weights and biases
    // 4. epochs, number of times to iterate over the total dataset
    // 5. batch_size, number of samples processed before updating weights
    // 6. print training progress
    void train(const std::vector<std::vector<double>>& inputs,
              const std::vector<std::vector<double>>& targets,
              double learning_rate,
              size_t epochs,
              size_t batch_size = 1,
              bool verbose = true);

    // Saves learned weights and biases of model (saves to file), takes in:
    // file name
    void save_model(const std::string& filename = "model.csv") const;

    // Loads weights and biases of pre-trained model (loads from file), takes in:
    // file name
    void load_model(const std::string& filename = "model.csv");
};

#endif //NEURALNETWORK_H
