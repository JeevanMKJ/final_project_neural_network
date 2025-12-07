# Checkpoint 2 Reflection

## What I Implemented
Implemented the training pipeline, which includes:
*   **MSE Loss/Error**: Calculating the error between predictions and targets.
*   **Backpropagation**: Using the calculus chain rule to propagate error gradients backwards from output to input.
*   **Gradient Descent**: Updating weights and biases based on calculated gradients and a learning rate.

## How It Integrates
Without **forwardpropagation**, this feature (**backpropagation**) is not possible. The `train()` function calls `forward()` to get predictions, compares them to targets, and then uses the `Matrix` operations defined in Checkpoint 1 to compute gradients.

## Challenges I Faced
*   **Chain Rule Complexity**: Correctly implementing the math for `δ` values (error terms) at hidden layers was difficult. The calculus for backpropagation includes multi variable derivations, a concept taught in calculus 3. Teaching myself the mathematical concepts took a while. 
*   **Indexing**: Managing `l`, `l-1`, and `l+1` indices during the backward loop caused "off-by-one" errors.
*   **Hyperparameters**: Finding the right learning rate for the test cases took trial and error. From Andrew Ng's Machine Learning course I learned that there are methods to obtain the optimal hyperparameters, such as using Optuna, this is not a feature that I will be implementing in this project.

## What I Learned
*   **Backpropagation**: It is basically only applying the calculus chain rule repeatedly.
*   **Mini-batches**: Accumulating gradients over a batch stabilizes learning compared to single-sample updates.

## Testing Results
Tests confirmed that loss decreases over time and the network can learn simple logical functions (AND gate), proving the backpropagation logic is correct.

## Time Spent
Approximately 15 hours.