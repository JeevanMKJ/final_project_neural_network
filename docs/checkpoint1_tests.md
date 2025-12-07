# Checkpoint 1 Test Cases

## Feature: Forward Propagation & Prediction

### Test Setup
- **Test file**: `tests/test_checkpoint1_forward_propagation.cpp`
- **Build system**: CMake/CTest
- **Run command**: `ctest --verbose`

### Test 1: Normal Operation (Matrix & Forward)
- **What it tests**: Validates `Matrix::dot_product` correctness and a full forward pass through a multi-layer network.
- **Expected result**: Output dimension matches the last layer, and calculation matches expected math.

### Test 2: Activation Functions
- **What it tests**: Correct behavior of ReLU (max(0,z)) and Sigmoid (0-1 range).
- **Expected result**: Negative inputs to ReLU become 0; Sigmoid outputs are strictly between 0 and 1.


### Test 3: Edge Cases
- **What it tests**: Handling of single-neuron networks and zero-vector inputs.
- **Expected result**: No crashes, valid output within activation ranges.


### CTest Output Summary
```
Test project .../build
    Start 1: Checkpoint1ForwardPropagation
1/1 Test #1: Checkpoint1ForwardPropagation ....   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1
```