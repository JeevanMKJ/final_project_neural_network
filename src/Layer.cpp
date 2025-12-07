#include "Layer.h"

// ANn stands for Andrew Ng's notation.

// Constructor: Creates a layer with: 
// 1. specified index
// 2. size 
// 3. activation type
// ANn: layer l has n^[l] neurons
Layer::Layer(int index, int size, activation_type act_type)
    : type(act_type), layer_index(index), size(size) {

    // Resize vectors to match layer size
    // All initialized to 0.0 as starting values
    z.resize(size, 0.0);        // Pre-activation values: z^[l]
    a.resize(size, 0.0);        // Activation outputs: a^[l]
    bias.resize(size, 0.0);     // Biases: b^[l] (okay to init to zero)
    gradient.resize(size, 0.0); // Gradients: d^[l]
}

// activate: Applies activation function g(x)
// ReLU:    g(x) = max(0, x)
// Sigmoid: g(x) = 1 / (1 + e^(-x))
// None:    g(x) = x (identity)
double Layer::activate(double x) const {
    switch (type) {
        case activation_type::ReLU:
            // ReLU: Returns x if positive, otherwise 0
            return (x > 0.0) ? x : 0.0;

        case activation_type::Sigmoid:
            // Sigmoid: Squashes output to range (0, 1)
            return 1.0 / (1.0 + std::exp(-x));

        case activation_type::None:
        default:
            // Identity function: output equals input (no transformation needed)
            return x;
    }
}

// derive: Computes derivative g'(x) of activation function
// ReLU:    g'(x) = 1 if x > 0, else 0
// Sigmoid: g'(x) = g(x) * (1 - g(x))  
// None:    g'(x) = 1 
double Layer::derive(double x) const {
    switch (type) {
        case activation_type::ReLU:
            // ReLU derivative: 1 for positive inputs, 0 otherwise (can cause dead neurons)
            return (x > 0.0) ? 1.0 : 0.0;

        case activation_type::Sigmoid: {
            // Sigmoid derivative: sig'(x) = sig(x) * (1 - sig(x))
            double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            return sigmoid_x * (1.0 - sigmoid_x);
        }

        case activation_type::None:
        default:
            // Derivative: slope of y = x is always 1
            return 1.0;
    }
}

// Getter for activation type
activation_type Layer::get_activation_type() const {
    return type;
}
