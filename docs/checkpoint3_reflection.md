# Checkpoint 3 Reflection

## What I Implemented
Implemented `save_model` and `load_model` methods in the `NeuralNetwork` class. These methods serialize the weight matrices and bias vectors to a CSV text format and parse them back to restore the network state.

## Integration
It saves the trained weights (result of Feature 2) so they can be utilized by Feature 1 (prediction) in a separate session.

## Challenges I Faced
*   **Precision**: Standard float formatting wasn't precise enough; I had to use `std::setprecision(10)` to prevent accuracy loss during round-trips.
*   **Parsing**: Reading variable-length lines from a text file required precise use of `stringstream`.

## What I Learned
*   **Serialization**: How to convert complex objects (layers of matrices) into flat storage formats. It took me a while till I could conceptually understand this idea.
*   **Numerical Stability**: The importance of high precision in file I/O for ML models.

## Testing Results
Tests verified that a loaded model behaves like the original, producing identical predictions and loss values.

## Overall Project Completion Progress
*   **Status**: 100% Complete.
*   **Original Plan**: Build a neural network and calculations from scratch (implementing  forwardpropagation, training, and persistence).
*   **Remaining**: None. 
*   **Adjustments**: None.

## Time Spent
Approximately 15 hours.