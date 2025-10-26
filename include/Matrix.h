#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>

// Matris class implements the mathematical operations for the neural network.
// The class stores a 2D matrix of doubles and provides methods for accessing elements
// Simple matrix operations areused are used to store weights and other data.
class Matrix {
private:
    // 2D vector to store the matrix data.
    std::vector<std::vector<double>> data;
    // Number of rows and columns in the matrix.
    size_t rows, cols;

public:
    // Construstor, takes in:
    // 1. r for number of rows
    // 2. c for number of columns
    Matrix(size_t r, size_t c);

    // Accessor Methods
    size_t get_rows() const;
    size_t get_cols() const;

    // Gets element at specifix row and column.
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    // Populates matrix with random started weights
    Matrix& fill_random(double min = -1.0, double max = 1.0);

    Matrix operator*(double scalar) const;

    Matrix& operator*=(double scalar);

    // Friend function to allow printing the matrix to an output stream.
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
};

#endif //MATRIX_H
