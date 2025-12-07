#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>
#include "Layer.h"
#include "Matrix.h"

// ANn stands for Andrew Ng's notation.

// NeuralNetwork class: manages entire neural network: 
// 1. layers
// 2. weights
// 3. training and prediction
// 4. forward of porpagation 
// 5. back propigation errors
// 6. updating of weights and biases.
class NeuralNetwork {
private:
    // Vector of Layer objects (architecture of network)
    std::vector<Layer> layers;
    // Vector of Matrix objects (weights connecting layers)
    std::vector<Matrix> weights;

    // Connets network layers
    void connect_layers();

    // Initializes weights matrices for connection between layers
    void initializeWeights();

    // Loss(Error) Function
    // Mean Squared Error(mse): J = (1/2) * sig(y - y)^2 
    // Note: (1/2) used for cleaner derivative

    // compute_mse_loss: Calculates the loss(error) value for a single sample
    // Parameters: 
    // 1. ouput = predicted output
    // 2. target = actual target
    // Returns: loss value
    double compute_mse_loss(const std::vector<double>& output,
                           const std::vector<double>& target) const;

    // compute_mse_loss_derivative: Calculates dJ/da for MSE loss
    // For MSE: dJ/da = (a - y)
    // Parameters: 
    // 1. output = predicted value
    // 2. target = actual value
    // Returns: derivative of loss with respect to output
    double compute_mse_loss_derivative(double output, double target) const;

public:
    // Constructor:
    // Paramater: vector of layer objects (network architecture)
    NeuralNetwork(const std::vector<Layer>& network_layers);

    // Forward Propigation:
    // Paramater: vector of doubles (network input)
    // Returns: vector of doubles (network output)
    std::vector<double> forward(const std::vector<double>& input);

    // Predictions:
    // takes in: vector of doubles (represents input to network)
    // returns: vector of doubles (represnts network's prediction)
    std::vector<double> predict(const std::vector<double>& input);

    // Backpropagation (Training network):
    // Paramaters: 
    // 1. vector of input vectors
    // 2. vector of target vectors
    // 3. learning_rate (steps size for updating weights and biases)
    // 4. epochs (number of times to iterate over dataset)
    // 5. batch_size (number of samples processed before updating weights)
    // 6. print training progress
    void train(const std::vector<std::vector<double>>& inputs,
              const std::vector<std::vector<double>>& targets,
              double learning_rate,
              size_t epochs,
              size_t batch_size = 1,
              bool verbose = true);

    // Saves learned weights and biases:
    void save_model(const std::string& filename = "model.csv") const;

    // Loads weights and biases of trained model:
    void load_model(const std::string& filename = "model.csv");
};

#endif //NEURALNETWORK_H
