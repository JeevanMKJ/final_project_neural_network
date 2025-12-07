# Checkpoint 3: Third Feature Plan

## Feature Name
Model Persistence (Save/Load)

## Description
Feature will save the trained network's parameters (weights and biases) to a file and be able to load them. Prevents the need to retrain the model every time the program runs, which is completely unsustainable with larger or more complex data sets.

## Why This Feature?
Last step in machine learning lifecycle (Train -> Save -> Deploy). Adds to the previous features by serializing the weights learned in Feature 2 and used in Feature 1.

## Classes Involved
*   `NeuralNetwork`: Added `save_model(filename)` and `load_model(filename)`.
*   `Matrix`: Methods to access/modify data.

## Expected User Workflow
1.  Train a network (or have one ready).
2.  Call `network.save_model("model.csv")`.
3.  Create a new network instance.
4.  Call `network.load_model("model.csv")`.
5.  Use `predict()` immediately without retraining.

## Estimated Time
20 hours