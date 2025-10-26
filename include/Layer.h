#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <random>

// Defines type for activation functions, which take a double and return a double.
using activation_function = std::function<double(double)>;

// Definitions and desctiptions of the Weighted Sum, Relu and Sigmoid formulas
// can be found in definitions.md at Documentation/definitions.md
namespace activation {

    // Rectified Linear Unit (ReLU) activation function
    inline double relu(double x) { // The 'inline' keyword is used to prevent 'multiple definition error' by telling the compiler
        if (x > 0.0) {             // that it is alright that ther are more than one of the same function.
            return x;
        } else {
            return 0.0;
        }
    }
    // OR
    // inline double relu(double x) { return (x > 0.0) ? x : 0.0; }


    // Derivative of the ReLU function
    inline double relu_derivative(double x) {
        if (x > 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    // Or
    // inline double relu_derivative(double x) { return (x > 0.0) ? 1.0 : 0.0; }

    // Sigmoid activation function
    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

    // Derivative of the sigmoid function
    inline double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
}

// Enum to specify the type of activation function for a layer.
enum class activation_type {
    None,
    ReLU,
    Sigmoid
};

// activation_type: returns a pair containing an activation function and its derivative
// type: returns desired activation function type
inline std::pair<activation_function, activation_function>
get_activation_pair(activation_type type) {
    using namespace activation;
    if (type == activation_type::ReLU) {
        return {relu, relu_derivative};
    } else if (type == activation_type::Sigmoid) {
        return {sigmoid, sigmoid_derivative};
    } else { // activation_type::None or default
        return {activation_function{}, activation_function{}};
    }
}

// Layer class represents a single layer of neurons in the neural network.
// Each layer has a specific size (number of neurons), an activation function,
// and stores the values related to the forward and backward passes of data.
class Layer {
private:
    // Activation function for layer
    activation_function activation;
    // Derivative of activation function, used during backpropagation
    activation_function activation_derivative;

public:
    int layer_index; // The index of layer in the network
    int size;        // The number of neurons in layer

    // Values stored during computation
    std::vector<double> z;        // pre-activation values (weighted sum + bias).
    std::vector<double> a;        // activation output values (after applying the activation function).
    std::vector<double> bias;     // biases for each neuron in the layer (not used for the input layer).
    std::vector<double> gradient; // gradients calculated during backpropagation (not for the input layer).

    // Constructor, takes in:
    // 1. index of the layer in the network
    // 2. number of neurons in the layer
    // 3. type of activation function to use
    Layer(int index, int size, activation_type act_type);

    // Applies the layer's activation function to a given value
    double apply_activation(double x) const;

    // Applies the derivative of the layer's activation function
    double apply_activation_derivative(double x) const;

    // Optional helper methods to check if functions are set
    bool has_activation() const;
    bool has_derivative() const;
};

#endif //LAYER_H
