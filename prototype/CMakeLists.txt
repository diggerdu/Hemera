cmake_minimum_required(VERSION 2.6)
project(prototype)

add_executable(prototype main.cpp)

install(TARGETS prototype RUNTIME DESTINATION bin)
