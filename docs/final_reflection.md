# Final Project Reflection

## Project Summary
Built a neural network and the calculations from scratch in C++.
**Implemented Features:**
1.  **Forward Propagation**: Matrix math and activation functions for prediction.
2.  **Training**: Backpropagation and MSE loss minimization.
3.  **Persistence**: Saving and loading models to CSV.
I chose to have code that is explicit, maintaining it's clearer mathematical expression, rather that hiding the math with abstraction. 
The model achieves >82% accuracy on real data.

## What Remains
**Status**: The project is complete and has achieved the original plan.
**Incomplete items**: None.

## Possible future Extensions

**1. Binary Cross-Entropy Loss (5 hours)**
* Replace MSE loss function with Binary Cross-Entropy
* update loss calculation
* modify gradient computation

**2. Multiclass Classification with Softmax (15 hours)**
* Replace output layer: 1 neuron -> N neurons (one per class)
* implement Softmax activation
* replace BCE with Cross-Entropy Loss
* modify all 3 test suites for multiclass scenarios.


## What I Learned
*   **C++ Skills**: Improved manual memory management (vectors), file I/O, and class design separation.
*   **Linear Algebra and Multivariable Calculus**: Teaching myself mathematics beyond the scope of previous understanding. 
*   **Debugging**: Debugging backpropagation indices (`l` vs `l-1`) took understanding the mathematics and much trial and error to get right.
*   **Data Structures**: Got me excited about possibilities of storing data of all sorts in `vector<vector<double>>`. 
*   **Machine Learning**: Taking Andrew Ng's course I learned more than just neural networks. With this project I got introduced to the many exciting and different applications of machine learning. 

## Section 5: Time Log
*   **Planning**: 5 hours
*   **Implementation**: +60 hours
*   **Testing/Debugging**: +10 hours
*   **Documentation**: 6 hours
*   **Total**: +80 hours