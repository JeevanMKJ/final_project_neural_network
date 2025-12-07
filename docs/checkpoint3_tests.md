# Checkpoint 3 Test Cases

## Feature: Model Persistence

### Test Setup
- **Test file**: `tests/test_checkpoint3_persistence.cpp`
- **Build system**: CMake/CTest
- **Run command**: `ctest --verbose`

### Test 1: Normal Operation (Save/Load)
- **What it tests**: Saving model to a file and loading it back up.
- **Expected result**: File is created, and loading up successfully.

### Test 2: Integration (Preservation)
- **What it tests**: Verifies that loaded model makes the same predictions as the original model.
- **Expected result**: Predictions match.

### Test 3: Edge Case (Data Integrity)
- **What it tests**: Ensures saving/loading multiple times (cycles) does not degrade model precision.
- **Expected result**: Loss/Predictions remain constant across save iterations.

### CTest Output Summary
```
Test project .../build
    Start 1: Checkpoint3Persistence
1/1 Test #1: Checkpoint3Persistence ...........   Passed    0.01 sec

100% tests passed, 0 tests failed out of 1
```