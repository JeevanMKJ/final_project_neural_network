#include "Data.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

Data::Data(const std::string& filepath)
    : filename(filepath), is_loaded(false), num_samples(0), num_features(0) {
}

void Data::parse_line(const std::string& line, std::vector<double>& row) {
    std::stringstream ss(line);
    std::string value;

    while (std::getline(ss, value, ';')) { 
        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());

        if (!value.empty()) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::invalid_argument& e) {
                continue;
            }
        }
    }
}

void Data::read_from_csv() {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    int line_count = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line_count < 2) {
            line_count++;
            continue;
        }

        std::vector<double> row;
        parse_line(line, row);

        if (!row.empty()) {
            raw_data.push_back(row);
        }
        line_count++;
    }

    file.close();

    num_samples = raw_data.size();

    if (num_samples == 0) {
        throw std::runtime_error("Could not load data from file.");
    }

    num_features = raw_data[0].size() - 2;  

    features.resize(num_samples);
    labels.resize(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        features[i].assign(raw_data[i].begin() + 1, raw_data[i].end() - 1);

        labels[i].push_back(raw_data[i].back());
    }

    is_loaded = true;
    normalize_features();
}

void Data::normalize_features() {
    if (num_samples == 0 || num_features == 0) return;

    std::vector<double> means(num_features, 0.0);
    std::vector<double> stds(num_features, 0.0);

    for (size_t j = 0; j < num_features; ++j) {
        for (size_t i = 0; i < num_samples; ++i) {
            means[j] += features[i][j];
        }
        means[j] /= num_samples;
    }

    for (size_t j = 0; j < num_features; ++j) {
        for (size_t i = 0; i < num_samples; ++i) {
            double diff = features[i][j] - means[j];
            stds[j] += diff * diff;
        }
        stds[j] = std::sqrt(stds[j] / num_samples);

        if (stds[j] < 1e-10) {
            stds[j] = 1.0;
        }
    }

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features[i][j] = (features[i][j] - means[j]) / stds[j];
        }
    }
}

void Data::report() const {
    if (!is_loaded) {
        std::cout << "No data loaded." << std::endl;
        return;
    }

    std::cout << "=== Data Summary ===" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "Samples: " << num_samples << std::endl;
    std::cout << "Features: " << num_features << std::endl;

    std::cout << "\nLabel distribution:" << std::endl;
    int count_0 = 0, count_1 = 0;
    for (const auto& label : labels) {
        if (label[0] < 0.5) count_0++;
        else count_1++;
    }
    std::cout << "  Class 0: " << count_0 << " (" << (100.0 * count_0 / num_samples) << "%)" << std::endl;
    std::cout << "  Class 1: " << count_1 << " (" << (100.0 * count_1 / num_samples) << "%)" << std::endl;
}

void Data::train_test_split(double test_ratio,
                           std::vector<std::vector<double>>& X_train,
                           std::vector<std::vector<double>>& y_train,
                           std::vector<std::vector<double>>& X_test,
                           std::vector<std::vector<double>>& y_test) {

    if (!is_loaded) {
        throw std::runtime_error("Data not loaded. Call read_from_csv() first.");
    }

    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    size_t test_size = static_cast<size_t>(num_samples * test_ratio);
    size_t train_size = num_samples - test_size;

    X_train.clear();
    y_train.clear();
    X_test.clear();
    y_test.clear();

    for (size_t i = 0; i < train_size; ++i) {
        X_train.push_back(features[indices[i]]);
        y_train.push_back(labels[indices[i]]);
    }

    for (size_t i = train_size; i < num_samples; ++i) {
        X_test.push_back(features[indices[i]]);
        y_test.push_back(labels[indices[i]]);
    }

    std::cout << "Train set: " << train_size << " samples" << std::endl;
    std::cout << "Test set: " << test_size << " samples" << std::endl;
}

const std::vector<std::vector<double>>& Data::get_features() const {
    return features;
}

const std::vector<std::vector<double>>& Data::get_labels() const {
    return labels;
}

size_t Data::get_num_samples() const {
    return num_samples;
}

size_t Data::get_num_features() const {
    return num_features;
}
