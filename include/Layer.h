#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cmath>
#include <stdexcept>

// ANn stands for Andrew Ng's notation.

// Enum to specify the type of activation function for a layer.
enum class activation_type {
    None,       // Identity function, input layer (no activation)
    ReLU,       // Rectified Linear Unit, hidden layers
    Sigmoid     // Sigmoid function, output layer (binary classification)
};

// Layer class represents a single layer of neurons in the neural network.
// Each layer has: 
// 1. specific size (number of neurons)
// 2. an activation type
// 3. stores the values related to the forward and backward passes of data.
class Layer {
private:
    activation_type type;

public:
    int layer_index; // index of layer in the network (0 = input, L = output)
    int size;        // number of neurons in layer (n^[l] in ANn)

    // Values stored during computation
    std::vector<double> z;        // Pre-activation values: z = W*a + b
    std::vector<double> a;        // Activation outputs: a = g(z)
    std::vector<double> bias;     // Biases for each neuron (b^[l])
    std::vector<double> gradient; // Gradients during backprop (δ^[l])

    // Constructor, takes in:
    // 1. index of the layer in network
    // 2. number of neurons in layer
    // 3. type of activation function
    Layer(int index, int size, activation_type act_type);

    // activate: Applies the activation function g(x) to a given value
    // ReLU:    g(x) = max(0, x)
    // Sigmoid: g(x) = 1 / (1 + e^(-x))
    // None:    g(x) = x (identity)
    double activate(double x) const;

    // derive: Computes the derivative g'(x) of the activation function
    // Used for backpropagation (applied the chain rule)
    // Note: It is interseting to see how mathematics lies at the foundation of many concepts in Computer Science
    // ReLU:    g'(x) = 1 if x > 0, else 0
    // Sigmoid: g'(x) = g(x) * (1 - g(x))
    // None:    g'(x) = 1
    double derive(double x) const;

    // Getter for activation type 
    activation_type get_activation_type() const;
};

#endif //LAYER_H
