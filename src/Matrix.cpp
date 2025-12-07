#include "Matrix.h"
#include <random>
#include <iomanip>
#include <iostream>

// ANn stands for Andrew Ng's notation.

// Constructor: creates an r×c matrix filled with zeros
// ANn: W^[l] e R^(r * c)
Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c) {
    // Initialize 2D vector with r rows, each containing c columns of zeros
    data.resize(rows, std::vector<double>(cols, 0.0));
}

// Returns number of rows in the matrix
// ANn: n^[l] (neurons in current layer)
size_t Matrix::get_rows() const {
    return rows;
}

// Returns number of columns in the matrix
// ANn: n^[l-1] (neurons in previous layer)
size_t Matrix::get_cols() const {
    return cols;
}

// get_value: Read element at row i, column j
// ANn: w_ij^[l]
// Returns the value at position (i, j)
double Matrix::get_value(size_t i, size_t j) const {
    return data[i][j];
}

// set_value: Write a value to row i, column j
// ANn: w_ij^[l] = value
void Matrix::set_value(size_t i, size_t j, double value) {
    data[i][j] = value;
}

// fill_random: Populates matrix with random values
// Used for weight initialization to break symmetry
// ANn: "Initialize weights to small random values"
// seed: If 0, uses hardware random. Otherwise uses provided seed for reproducibility.
Matrix& Matrix::fill_random(double min, double max, unsigned int seed) {
    // Create random number generator
    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;  
        gen.seed(rd());
    } else {
        gen.seed(seed);  
    }

    std::uniform_real_distribution<double> dist(min, max);

    // Fill each element with random value in [min, max]
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return *this;  // Return reference for method chaining
}

// dot_product: Matrix-Vector multiplication
// Computes: result = W * input_vector
// ANn: z = W * a 
//
// Mathematical formula for each output element:
//   result[i] = Σ(W[i][j] * input[j]) for all j
//
// Dimensions:
//   W: rows × cols (e.g., 16 × 23)
//   input_vector: cols elements (e.g., 23)
//   result: rows elements (e.g., 16)
std::vector<double> Matrix::dot_product(const std::vector<double>& input_vector) const {
    // Validate dimensions: input must match number of columns
    if (input_vector.size() != cols) {
        throw std::invalid_argument(
            "Dimension mismatch in dot_product: matrix has " +
            std::to_string(cols) + " columns but input has " +
            std::to_string(input_vector.size()) + " elements"
        );
    }

    // Create result vector with 'rows' elements, initialized to 0
    std::vector<double> result(rows, 0.0);

    // Compute matrix-vector multiplication
    // For each row i of the matrix:
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;

        // Compute dot product of row i with input vector
        // sum = W[i][0]*input[0] + W[i][1]*input[1] + ... + W[i][cols-1]*input[cols-1]
        for (size_t j = 0; j < cols; ++j) {
            sum += data[i][j] * input_vector[j];
        }

        result[i] = sum;
    }

    return result;
}

// transpose: Flips rows and columns
// ANn: (W^[l])^T - transpose of weight matrix
// Original matrix: m rows × n cols
// Transposed matrix: n rows × m cols

// Mathematical definition: (W^T)[i][j] = W[j][i]
// Used in backpropagation for computing gradients
Matrix Matrix::transpose() const {
    // Create new matrix with swapped dimensions
    // If original is (rows × cols), result is (cols × rows)
    Matrix result(cols, rows);

    // Swap rows and columns: result[j][i] = original[i][j]
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Original position: (i, j)
            // Transposed position: (j, i)
            result.set_value(j, i, data[i][j]);
        }
    }

    return result;
}

// scale: Returns a NEW matrix with all elements multiplied by scalar
// Does NOT modify the original matrix
// Useful for: creating scaled copies without changing original
Matrix Matrix::scale(double scalar) const {
    Matrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.set_value(i, j, data[i][j] * scalar);
        }
    }

    return result;
}

// scale_inplace: Modifies THIS matrix by multiplying all elements by scalar
// Useful for: weight updates during gradient descent
// W := W - α * gradient  (where α is the learning rate)
void Matrix::scale_inplace(double scalar) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] *= scalar;
        }
    }
}

// print: Displays the matrix contents to console
// Format: Each row on a new line, values comma-separated
void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(4);  

    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << data[i][j];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << " ]" << std::endl;
    }
}
