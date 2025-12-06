# Deep Dive: Backpropagation & Gradient Descent

This document provides an extremely detailed explanation of the **Backpropagation** algorithm implemented in `src/NeuralNetwork.cpp`. It bridges the gap between high-level concepts ("The Forest"), mathematical derivation ("The Trees"), and the actual C++ code.

---

## 1. The Forest: The High-Level Goal

The goal of training a neural network is to find a set of **weights** ($W$) and **biases** ($b$) that minimize the **Cost Function** ($J$). The Cost Function measures the difference between the network's predictions ($\hat{y}$) and the actual targets ($y$).

**Backpropagation** is simply a method to calculate the **Gradient** (slope) of the Cost Function with respect to every single weight and bias in the network.

*   **Forward Propagation**: Input $\to$ Hidden Layers $\to$ Output $\to$ Loss.
*   **Backpropagation**: Loss $\to$ Output Error $\to$ Hidden Layer Errors $\to$ Gradients.
*   **Gradient Descent**: Update weights using the gradients to reduce loss.

---

## 2. Variables & Notation (Math vs. Code)

We adhere to the notation used in Andrew Ng's Deep Learning courses, which is consistent with the codebase.

| Symbol | Code Variable | Definition | Dimensions |
| :--- | :--- | :--- | :--- |
| $L$ | `layers.size() - 1` | Index of the last layer (Output Layer). | Scalar |
| $l$ | `l` | Current layer index ($0$ is input, $1, 2...$ hidden). | Scalar |
| $n^{[l]}$ | `layers[l].size` | Number of neurons in layer $l$. | Scalar |
| **$W^{[l]}$** | `weights[l-1]` | Weights connecting layer $l-1$ to layer $l$. | $n^{[l]} \times n^{[l-1]}$ |
| **$b^{[l]}$** | `layers[l].bias` | Biases for layer $l$. | $n^{[l]} \times 1$ |
| **$z^{[l]}$** | `layers[l].z` | **Pre-activation** (Linear input). $z = Wa + b$. | $n^{[l]} \times 1$ |
| **$a^{[l]}$** | `layers[l].a` | **Activation** (Non-linear output). $a = g(z)$. | $n^{[l]} \times 1$ |
| $g(\cdot)$ | `activate()` | Activation function (ReLU, Sigmoid). | Function |
| $g'(\cdot)$ | `derive()` | Derivative of activation function. | Function |
| **$\delta^{[l]}$** | `layers[l].gradient` | **Error term** for layer $l$. Represents $\frac{\partial J}{\partial z^{[l]}}$. | $n^{[l]} \times 1$ |
| $J$ | `loss` | Cost function (Mean Squared Error). | Scalar |
| $\alpha$ | `learning_rate` | Step size for Gradient Descent. | Scalar |

> **Critical Note on Indexing**: 
> In the code, `weights` vector is 0-indexed. 
> `weights[0]` connects Layer 0 to Layer 1. 
> Math notation $W^{[1]}$ corresponds to code `weights[0]`. 
> Math notation $W^{[l]}$ corresponds to code `weights[l-1]`.

---

## 3. The Trees: Step-by-Step Derivation

We use the **Chain Rule** of calculus to propagate error backwards.

### Step A: The Loss Function
We use Mean Squared Error (MSE). For a single example:
$$J = \frac{1}{2} (a^{[L]} - y)^2$$

The derivative of the loss with respect to the final prediction $a^{[L]}$ is:
$$\frac{\partial J}{\partial a^{[L]}} = (a^{[L]} - y)$$

**Code (`src/NeuralNetwork.cpp`):**
```cpp
double dJ_da = compute_mse_loss_derivative(layers[L].a[j], targets[i][j]);
// returns (output - target)
```

---


### Step B: Output Layer Error ($\delta^{[L]}$)

We need the error at the output layer, defined as $\delta^{[L]} = \frac{\partial J}{\partial z^{[L]}}$.
Using the Chain Rule:
$$\delta^{[L]} = \frac{\partial J}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}}$$

1.  $\frac{\partial J}{\partial a^{[L]}}$ is $(a^{[L]} - y)$ (from Step A).
2.  $\frac{\partial a^{[L]}}{\partial z^{[L]}}$ is the derivative of the activation function $g'(z^{[L]})$.

**Formula:**
$$\delta^{[L]} = (a^{[L]} - y) \odot g'(z^{[L]})$$
*($\odot$ denotes element-wise multiplication)*

**Code Implementation:**
```cpp
// L is the index of the output layer
for (int j = 0; j < layers[L].size; ++j) {
    // 1. Calculate dJ/da = (prediction - target)
    double dJ_da = compute_mse_loss_derivative(layers[L].a[j], targets[i][j]);
    
    // 2. Calculate g'(z)
    double g_prime = layers[L].derive(layers[L].z[j]);
    
    // 3. Combine: delta = dJ/da * g'(z)
    layers[L].gradient[j] = dJ_da * g_prime;
}
```

---


### Step C: Hidden Layer Errors ($\delta^{[l]}$)

Now we move backwards from layer $l+1$ to layer $l$. We want to find $\delta^{[l]}$.
The error at layer $l$ depends on the error at layer $l+1$ and the weights connecting them.

**The Chain Rule Logic:**
1.  The error $\delta^{[l+1]}$ exists at the next layer.
2.  We propagate this error *backwards* through the weights $W^{[l+1]}$.
    *   Mathematically, this is done by multiplying by the **transpose** matrix $(W^{[l+1]})^T$.
3.  We scale this by the derivative of the activation function at current layer $l$, $g'(z^{[l]})$.

**Formula:**
$$\delta^{[l]} = ((W^{[l+1]})^T \cdot \delta^{[l+1]}) \odot g'(z^{[l]})$$

**Why Transpose?**
*   $W^{[l+1]}$ maps dimensions $n^{[l]} \to n^{[l+1]}$ (Forward).
*   To go backward ($n^{[l+1]} \to n^{[l]}$), we must flip rows and cols using Transpose.

**Code Implementation:**
```cpp
// Loop backwards from L-1 down to 1
for (int l = L - 1; l >= 1; --l) {
    // 1. Transpose the weights of the *next* layer (weights[l] connects l to l+1)
    Matrix W_transpose = weights[l].transpose();

    // 2. Propagate error backward: weighted_gradient = W^T * delta[l+1]
    std::vector<double> weighted_gradient = W_transpose.dot_product(layers[l+1].gradient);

    // 3. Element-wise multiplication with g'(z[l])
    for (int j = 0; j < layers[l].size; ++j) {
        double g_prime = layers[l].derive(layers[l].z[j]);
        
        // delta[l] = (W^T * delta[l+1]) * g'(z)
        layers[l].gradient[j] = weighted_gradient[j] * g_prime;
    }
}
```

---


### Step D: Computing Gradients ($\frac{\partial J}{\partial W}$, $\frac{\partial J}{\partial b}$)

Now that we have the error term $\delta^{[l]}$ for every layer, we can calculate how much to change the weights and biases.

**1. Weight Gradients**
The gradient for weights $W^{[l]}$ is the error $\delta^{[l]}$ multiplied by the input that came into that connection, $a^{[l-1]}$.
$$\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} \cdot (a^{[l-1]})^T$$

**2. Bias Gradients**
The gradient for biases is just the error term itself.
$$\frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}$$

**Code Implementation (Accumulation Phase):**
```cpp
// Inside the loop over batch examples
for (size_t l = 1; l < layers.size(); ++l) {
    for (int j = 0; j < layers[l].size; ++j) {     // For each neuron in current layer
        for (int k = 0; k < layers[l-1].size; ++k) { // For each neuron in previous layer
            
            // Weight Gradient: delta[l][j] * a[l-1][k]
            // We accumulate this into weight_gradients for the batch
            double new_grad = weight_gradients[l-1].get_value(j, k) 
                            + layers[l].gradient[j] * layers[l-1].a[k];
            
            weight_gradients[l-1].set_value(j, k, new_grad);
        }
        
        // Bias Gradient: delta[l][j]
        bias_gradients[l-1][j] += layers[l].gradient[j];
    }
}
```

---


### Step E: Updating Weights (Gradient Descent)

Once we have accumulated gradients for the entire mini-batch, we average them and update the parameters.

**Update Rule:**
$$W^{[l]} := W^{[l]} - \alpha \cdot \frac{1}{m} \sum \frac{\partial J}{\partial W^{[l]}}$$
$$b^{[l]} := b^{[l]} - \alpha \cdot \frac{1}{m} \sum \frac{\partial J}{\partial b^{[l]}}$$

*   $\alpha$: Learning Rate.
*   $m$: Batch size.

**Code Implementation:**
```cpp
// Average and update
for (size_t l = 1; l < layers.size(); ++l) {
    for (int j = 0; j < layers[l].size; ++j) {
        for (int k = 0; k < layers[l-1].size; ++k) {
            // 1. Average the accumulated gradient
            double gradient = weight_gradients[l-1].get_value(j, k) / current_batch_size;
            
            // 2. Apply Update: New = Old - LearningRate * Gradient
            double current_weight = weights[l-1].get_value(j, k);
            double new_weight = current_weight - learning_rate * gradient;
            
            weights[l-1].set_value(j, k, new_weight);
        }
        // Update bias similarly...
    }
}
```

---

## 4. Summary Flowchart

```
FORWARD PASS:
Input -> [Weights] -> z -> [Activation] -> a -> [Weights] -> ... -> Output -> Cost

BACKWARD PASS:
1. Calculate Cost Derivative at Output: (a - y)
2. Output Error (delta): (a - y) * g'(z)
3. For each Hidden Layer (backwards):
    a. Transpose next layer's Weights
    b. Propagate error: W_transpose * delta_next
    c. Scale by activation derivative: * g'(z)
    d. Result is delta_current
4. Calculate Gradients:
    a. Weight Gradient: delta_current * a_previous
    b. Bias Gradient: delta_current
5. Update Parameters:
    W = W - alpha * avg_gradient
    b = b - alpha * avg_gradient
```
