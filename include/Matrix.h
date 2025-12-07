#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>

// ANn stands for Andrew Ng's notation.

// Matrix class implements mathematical operations for the neural network.
// Class stores a 2D matrix of doubles and provides methods for accessing elements.
// Matrix operations are used to store weights and perform calculations.
class Matrix {
private:
    // 2D vector to store the matrix data. A vector of vectors.
    std::vector<std::vector<double>> data;
    // Number of rows and columns in the matrix.
    size_t rows, cols;

public:
    // Constructor, takes in:
    // 1. r for number of rows
    // 2. c for number of columns
    Matrix(size_t r, size_t c);

    // Accessor Methods - Get matrix dimensions
    size_t get_rows() const;
    size_t get_cols() const;

    // Element Access Methods

    // get_value: Read a value at specific row/column (read-only)
    double get_value(size_t i, size_t j) const;

    // set_value: Write a value at specific row/column
    void set_value(size_t i, size_t j, double value);

    // Populates matrix with random values between min (-1.0) and max (1.0)
    // Used for weight initialization:
    // (breaking symmetry: if all weights were set to be the same then
    // all the gradients would be symmetrical)
    // seed: Optional random seed for reproducibility (0 = use hardware random)
    Matrix& fill_random(double min = -1.0, double max = 1.0, unsigned int seed = 0);

    // Linear Algebra Operations

    // dot_product: Matrix-Vector multiplication (W * x = result)
    // ANn: z = W * a
    // This is the fundamental operation in forward propagation
    // Input: vector of size cols (n^[l-1] neurons from previous layer)
    // Output: vector of size rows (n^[l] neurons in current layer)
    std::vector<double> dot_product(const std::vector<double>& input_vector) const;

    // transpose: Flips rows and columns of the matrix
    // ANn: (W^[l])^T - used in backpropagation
    // If original matrix is (m × n), transpose is (n × m)
    // Example: [1 2]^T = [1 3]
    //          [3 4]     [2 4]
    // Used for: d^[l] = (W^[l+1])^T · d^[l+1]
    Matrix transpose() const;

    // Scaling Operations 

    // scale: Returns a NEW matrix with all elements multiplied by scalar
    // Does NOT modify the original matrix 
    // Keeps original matrix unchanged.
    Matrix scale(double scalar) const;

    // scale_inplace: Modifies THIS matrix by multiplying all elements by scalar
    // More memory effecient.
    void scale_inplace(double scalar);

    // Output Method

    // print: Displays the matrix contents to console (for debugging)
    void print() const;
};

#endif //MATRIX_H
