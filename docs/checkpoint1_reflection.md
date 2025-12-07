# Checkpoint 1 Reflection

## What I Implemented
Built out forward propagation across the `Matrix`, `Layer`, and `NeuralNetwork` classes. The nwetowrk can now take an input vector, perform weighted sums and activation functions (ReLU/Sigmoid) layer-by-layer, and output a prediction.

## Challenges I Faced
The greatest hurdel across all feature implementations was understanding the mathematics of each concept and implementing it into C++.
*   Both **Matrix Dimensions** and **Dot Product Logic** took a while to comprehend and even longer to poperly implement. 

## What I Learned
*   **Linear Algebra**: After having taught myself the relevant linear algebra I realized that neural networks are essentially a series of chained matrix transformations.
*   **Code Structure**: Learned how to separate concerns between Data, Math, and Logic classes in C++.

## Testing Results
Verified feature with 3 automated tests covering forward passes, activation functgions, and edge cases like zero-input. 

## Questions for Next Checkpoint
*   How do I handle the specific indices during backpropagation (layer `l` vs `l-1`)?
*   What is the best way to initialize weights to prevent vanishing gradients? (Saw that this will be a concern from Andrew Ng's course.)

## Time Spent
Approximately 15 hours