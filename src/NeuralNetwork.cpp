#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>

// ANn stands for Andrew Ng's notation.

NeuralNetwork::NeuralNetwork(const std::vector<Layer>& network_layers)
    : layers(network_layers) {

    // Validate: need at least 2 layers (input + output)
    if (layers.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers");
    }

    // Set up connections and initialize weights
    connect_layers();
    initializeWeights();
}

void NeuralNetwork::connect_layers() {
    // Create weight matrices for connections between layers
    // Number of weight matrices = number of layers - 1, (no weights before input layer)

    for (size_t l = 1; l < layers.size(); ++l) {
        int current_layer_size = layers[l].size;      // n^[l]
        int previous_layer_size = layers[l-1].size;   // n^[l-1]

        // Create weight matrix of size n^[l] × n^[l-1]
        // Rows = current layer neurons, Cols = previous layer neurons
        Matrix weight_matrix(current_layer_size, previous_layer_size);

        weights.push_back(weight_matrix);
    }
}

void NeuralNetwork::initializeWeights() {
    // Initialize all weight matrices with small random values
    for (auto& weight_matrix : weights) {
        weight_matrix.fill_random(-0.01, 0.01);
    }

    // Biases are already initialized to zero in Layer constructor
}


// compute_mse_loss: Mean Squared Error loss
// Formula: J = (1/2) * sig(y - y)^2
// the 1/2 factor cancels with the 2 from the derivative, making backprop cleaner
double NeuralNetwork::compute_mse_loss(const std::vector<double>& output,
                                        const std::vector<double>& target) const {
    double loss = 0.0;

    // Sum squared errors across all output neurons
    for (size_t i = 0; i < output.size(); ++i) {
        double error = output[i] - target[i];  // (y - y)
        loss += error * error;                  // (y - y)^2
    }

    return 0.5 * loss;  // Apply the 1/2 
}

// compute_mse_loss_derivative: Derivative of MSE with respect to output
// Formula: dJ/da = (y - y)
double NeuralNetwork::compute_mse_loss_derivative(double output, double target) const {
    return output - target;  // Simple: (y - y)
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    // Validate input size matches first layer
    if (input.size() != static_cast<size_t>(layers[0].size)) {
        throw std::invalid_argument("Input size and network input layer dont match.");
    }

    // Set input layer activations (a^[0] = input)
    layers[0].a = input;
    // Input layer doesn't compute z (no weights/biases before it)

    // Step 2: Forward propagation through remaining layers
    for (size_t l = 1; l < layers.size(); ++l) {
        Layer& current_layer = layers[l];
        Layer& previous_layer = layers[l-1];
        Matrix& W = weights[l-1];  // Weight matrix connecting layers

        // Compute z^[l] = W^[l] · a^[l-1] + b^[l]
        // Matrix-vector multiplication (W * a)
        current_layer.z = W.dot_product(previous_layer.a);

        // Add bias element-wise
        for (int i = 0; i < current_layer.size; ++i) {
            current_layer.z[i] += current_layer.bias[i];
        }

        // Compute a^[l] = g^[l](z^[l])
        for (int i = 0; i < current_layer.size; ++i) {
            current_layer.a[i] = current_layer.activate(current_layer.z[i]);
        }
    }

    // Return output layer activations 
    return layers.back().a;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    // Prediction is just forward propagation
    return forward(input);
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          double learning_rate,
                          size_t epochs,
                          size_t batch_size,
                          bool verbose) {

    // Validate inputs and targets match
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs and number of targets must match.");
    }

    size_t m = inputs.size();  // Number of training examples

    // Training loop: iterate over epochs
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        // Mini-batch gradient descent
        for (size_t batch_start = 0; batch_start < m; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, m);
            size_t current_batch_size = batch_end - batch_start;

            // Accumulates gradients over batch
            // Initializes gradient accumulators to zero
            std::vector<Matrix> weight_gradients;
            std::vector<std::vector<double>> bias_gradients;

            for (size_t l = 1; l < layers.size(); ++l) {
                weight_gradients.push_back(Matrix(layers[l].size, layers[l-1].size));
                bias_gradients.push_back(std::vector<double>(layers[l].size, 0.0));
            }

            // Process each example in the batch
            for (size_t i = batch_start; i < batch_end; ++i) {
                // Forward propagation
                std::vector<double> output = forward(inputs[i]);

                // Computes loss using explicit MSE function
                double loss = compute_mse_loss(output, targets[i]);
                total_loss += loss;

                // Backpropagation

                // --- Output Layer Gradient (l = L) ---
                // d^[L] = (a^[L] - y) dot g'^[L](z^[L])
                size_t L = layers.size() - 1;  

                for (int j = 0; j < layers[L].size; ++j) {
                    // Chain rule: d^[L] = dJ/da * g'(z)
                    double dJ_da = compute_mse_loss_derivative(layers[L].a[j], targets[i][j]);
                    double g_prime = layers[L].derive(layers[L].z[j]);
                    layers[L].gradient[j] = dJ_da * g_prime;
                }

                // --- Hidden Layer Gradients (l = L-1, L-2, ..., 1) ---
                // d^[l] = (W^[l+1])^T * d^[l+1] dot g'^[l](z^[l])
                for (int l = L - 1; l >= 1; --l) {
                    // Transpose the weight matrix
                    Matrix W_transpose = weights[l].transpose();

                    // Matrix-vector multiplication: (W^[l+1])^T · d^[l+1]
                    std::vector<double> weighted_gradient = W_transpose.dot_product(layers[l+1].gradient);

                    // Element-wise multiply by activation derivative (Hadamard product)
                    for (int j = 0; j < layers[l].size; ++j) {
                        double g_prime = layers[l].derive(layers[l].z[j]);
                        layers[l].gradient[j] = weighted_gradient[j] * g_prime;
                    }
                }

                // Accumulate gradients for this example
                for (size_t l = 1; l < layers.size(); ++l) {
                    // Weight gradients: dJ/dW^[l] = d^[l] * (a^[l-1])^T
                    for (int j = 0; j < layers[l].size; ++j) {
                        for (int k = 0; k < layers[l-1].size; ++k) {
                            double current_grad = weight_gradients[l-1].get_value(j, k);
                            double new_grad = current_grad + layers[l].gradient[j] * layers[l-1].a[k];
                            weight_gradients[l-1].set_value(j, k, new_grad);
                        }

                        // Bias gradients: dJ/db^[l] = d^[l]
                        bias_gradients[l-1][j] += layers[l].gradient[j];
                    }
                }
            }

            // Update weights and biases using averaged gradients
            for (size_t l = 1; l < layers.size(); ++l) {
                for (int j = 0; j < layers[l].size; ++j) {
                    for (int k = 0; k < layers[l-1].size; ++k) {
                        // reads current weight
                        double current_weight = weights[l-1].get_value(j, k);
                        // compute gradient update
                        double gradient = weight_gradients[l-1].get_value(j, k) / current_batch_size;
                        // apply learning rate and update
                        double new_weight = current_weight - learning_rate * gradient;
                        weights[l-1].set_value(j, k, new_weight);
                    }

                    // update bias
                    layers[l].bias[j] -= learning_rate * (bias_gradients[l-1][j] / current_batch_size);
                }
            }
        }

        // Print progress 
        if (verbose) {
            double avg_loss = total_loss / m;
            std::cout << "\rEpoch " << epoch << "/" << epochs
                      << " - Loss: " << avg_loss << "     " << std::flush;
        }
    }

    // Print newline after training completes
    if (verbose) {
        std::cout << std::endl;
    }
}

void NeuralNetwork::save_model(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Set precision for saving
    file << std::fixed << std::setprecision(10);

    // Save weights for each layer
    for (size_t l = 0; l < weights.size(); ++l) {
        file << "WEIGHTS_LAYER_" << (l+1) << std::endl;

        const Matrix& W = weights[l];
        for (size_t i = 0; i < W.get_rows(); ++i) {
            for (size_t j = 0; j < W.get_cols(); ++j) {
                file << W.get_value(i, j);
                if (j < W.get_cols() - 1) file << ",";
            }
            file << std::endl;
        }
    }

    // Save biases for each layer (except input layer)
    for (size_t l = 1; l < layers.size(); ++l) {
        file << "BIASES_LAYER_" << l << std::endl;

        for (size_t i = 0; i < layers[l].bias.size(); ++i) {
            file << layers[l].bias[i];
            if (i < layers[l].bias.size() - 1) file << ",";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    size_t weight_layer = 0;
    size_t bias_layer = 1;

    while (std::getline(file, line)) {
        // Check for weight section
        if (line.find("WEIGHTS_LAYER_") != std::string::npos) {
            // Read weight matrix
            Matrix& W = weights[weight_layer];

            for (size_t i = 0; i < W.get_rows(); ++i) {
                std::getline(file, line);
                std::stringstream ss(line);
                std::string value;

                for (size_t j = 0; j < W.get_cols(); ++j) {
                    std::getline(ss, value, ',');
                    W.set_value(i, j, std::stod(value));
                }
            }

            weight_layer++;
        }
        // Check for bias section
        else if (line.find("BIASES_LAYER_") != std::string::npos) {
            std::getline(file, line);
            std::stringstream ss(line);
            std::string value;

            for (size_t i = 0; i < layers[bias_layer].bias.size(); ++i) {
                std::getline(ss, value, ',');
                layers[bias_layer].bias[i] = std::stod(value);
            }

            bias_layer++;
        }
    }

    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}
