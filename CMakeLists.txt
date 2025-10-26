cmake_minimum_required(VERSION 3.20)
project(final_project_neural_network)

set(CMAKE_CXX_STANDARD 17)

add_executable(final_project_neural_network src/main.cpp src/Layer.cpp src/Matrix.cpp src/NeuralNetwork.cpp)

target_include_directories(final_project_neural_network PUBLIC include)