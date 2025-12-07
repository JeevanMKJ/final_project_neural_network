cmake_minimum_required(VERSION 3.20)
project(final_project_neural_network)

# totally AI generated, I accidentally "rm" my original CMakeLists.txt file 
# and could not git restore it. 

set(CMAKE_CXX_STANDARD 17)

add_executable(final_project_neural_network
    src/main.cpp
    src/Data.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
)

target_include_directories(final_project_neural_network PUBLIC include)

enable_testing()

add_executable(test_checkpoint1_forward_propagation
    tests/test_checkpoint1_forward_propagation.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
)

target_include_directories(test_checkpoint1_forward_propagation PUBLIC include)

add_test(NAME Checkpoint1ForwardPropagation COMMAND test_checkpoint1_forward_propagation)

add_executable(test_checkpoint2_training
    tests/test_checkpoint2_training.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
)

target_include_directories(test_checkpoint2_training PUBLIC include)

add_test(NAME Checkpoint2Training COMMAND test_checkpoint2_training)

add_executable(test_checkpoint3_persistence
    tests/test_checkpoint3_persistence.cpp
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNetwork.cpp
)

target_include_directories(test_checkpoint3_persistence PUBLIC include)

add_test(NAME Checkpoint3Persistence COMMAND test_checkpoint3_persistence)
