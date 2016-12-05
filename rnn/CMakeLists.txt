cmake_minimum_required(VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)

project(RNNPrototype)

add_library(network network.cpp)
add_executable(RNNPrototype main.cpp)


TARGET_LINK_LIBRARIES(RNNPrototype network)

add_library(weights weights.cpp)
TARGET_LINK_LIBRARIES(RNNPrototype weights)
add_library(ulti ulti.cpp)
TARGET_LINK_LIBRARIES(RNNPrototype ulti)
install(TARGETS RNNPrototype RUNTIME DESTINATION bin)
