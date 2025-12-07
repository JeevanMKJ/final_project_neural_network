# Checkpoint 2: Second Feature Plan

## Feature Name
Training (Backpropagation & Gradient Descent)

## Description
This feature will enable the network to learn. It implements the Mean Squared Error (MSE) loss function, calculating gradients via backpropagation, and updating weights using mini-batch gradient descent.

## Why This Feature?
Instead of a static predictor backpropagation will include the training element of a neural network, calculating the loss/error.

## Classes Involved
*   **NeuralNetwork**: Add `train()`, `backpropagate()`, and loss calculation methods.
*   **Layer**: Add `derive()` for activation derivatives and storage for gradients.
*   **Matrix**: Use `transpose()` and `dot_product()` for gradient calculations.

## Expected User Workflow
1.  Loads training data (inputs and targets).
2.  Calls `network.train(X, y, learning_rate, epochs)`.
3.  System iterates through epochs, updating weights to minimize loss.
4.  Evaluates performance with `predict()`.

## Estimated Time
20 hours