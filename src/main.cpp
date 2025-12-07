#include "Data.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include <iostream>
#include <iomanip>

int main() {
    try {
        std::cout << "=== Neural Network Training ===" << std::endl << std::endl;

        std::cout << "Loading data..." << std::endl;
        Data data("default of credit card clients.csv");
        data.read_from_csv();
        data.report();
        std::cout << std::endl;

        std::cout << "Splitting data..." << std::endl;
        std::vector<std::vector<double>> X_train, y_train, X_test, y_test;
        data.train_test_split(0.2, X_train, y_train, X_test, y_test);
        std::cout << std::endl;

        size_t input_size = data.get_num_features();

        std::cout << "Building neural network..." << std::endl;
        std::vector<Layer> architecture = {
            Layer(0, input_size, activation_type::None),    // input layer
            Layer(1, 16, activation_type::ReLU),            // hidden layer 1
            Layer(2, 8, activation_type::ReLU),             // hidden layer 2
            Layer(3, 1, activation_type::Sigmoid)           // output layer
        };

        std::cout << "Architecture: " << input_size << " -> 16 -> 8 -> 1" << std::endl;
        std::cout << std::endl;

        NeuralNetwork network(architecture);

        std::cout << "Training network..." << std::endl;
        network.train(
            X_train,           // training inputs
            y_train,           // training targets
            0.01,              // learning rate (Andrew Ng: start with 0.01)
            1000,              // epochs
            32,                // batch size
            true               // verbose (print progress)
        );
        std::cout << std::endl;

        std::cout << "Evaluating on test set..." << std::endl;
        int correct = 0;

        for (size_t i = 0; i < X_test.size(); ++i) {
            std::vector<double> prediction = network.predict(X_test[i]);

            int predicted_class = (prediction[0] >= 0.5) ? 1 : 0;
            int true_class = static_cast<int>(y_test[i][0]);

            if (predicted_class == true_class) {
                correct++;
            }
        }

        double accuracy = 100.0 * correct / X_test.size();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Correct: " << correct << "/" << X_test.size() << std::endl;
        std::cout << std::endl;

        std::cout << "Saving model..." << std::endl;
        network.save_model("trained_model.csv");
        std::cout << std::endl;

        std::cout << "Testing model loading..." << std::endl;
        NeuralNetwork loaded_network(architecture);
        loaded_network.load_model("trained_model.csv");

        std::vector<double> test_prediction = loaded_network.predict(X_test[0]);
        std::cout << "Loaded model prediction: " << test_prediction[0] << std::endl;
        std::cout << "True label: " << y_test[0][0] << std::endl;

        std::cout << "\n=== Training Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
