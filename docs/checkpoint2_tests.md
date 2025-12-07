# Checkpoint 2 Test Cases

## Feature: Training & Backpropagation

### Test Setup
- **Test file**: `tests/test_checkpoint2_training.cpp`
- **Build system**: CMake/CTest
- **Run command**: `ctest --verbose`

### Test 1: Normal Operation (Loss Reduction)
- **What it tests**: Verifies that MSE loss decreases over multiple training epochs on a simplified dataset.
- **Expected result**: Loss is significantly lower than initial loss.


### Test 2: Integration (Learning AND Function)
- **What it tests**: End-to-end test seeing if the network can learn the logical "AND" pattern.
- **Expected result**: Predictions for (1,1) are >0.5, others are <0.5.


### Test 3: Boundary/Edge (Weight Updates)
- **What it tests**: Checks weights are actually changing after each training step and checks batch size handling.
- **Expected result**: Weights before and after training are not identical; no crashes on different batch sizes.


### CTest Output Summary
```
Test project .../build
    Start 1: Checkpoint2Training
1/1 Test #1: Checkpoint2Training ..............   Passed    0.02 sec

100% tests passed, 0 tests failed out of 1
```