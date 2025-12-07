#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

class Data {
private:
    std::string filename;
    std::vector<std::vector<double>> raw_data;
    std::vector<std::vector<double>> features;  // X
    std::vector<std::vector<double>> labels;    // y

    bool is_loaded;
    size_t num_samples;
    size_t num_features;

    // Helper methods
    void normalize_features();
    void parse_line(const std::string& line, std::vector<double>& row);

public:
    // Constructor
    Data(const std::string& filepath);

    // Load data from CSV
    void read_from_csv();

    // Print data summary
    void report() const;

    // Split into train and test sets
    void train_test_split(double test_ratio,
                         std::vector<std::vector<double>>& X_train,
                         std::vector<std::vector<double>>& y_train,
                         std::vector<std::vector<double>>& X_test,
                         std::vector<std::vector<double>>& y_test);

    // Accessors
    const std::vector<std::vector<double>>& get_features() const;
    const std::vector<std::vector<double>>& get_labels() const;
    size_t get_num_samples() const;
    size_t get_num_features() const;
};

#endif 
