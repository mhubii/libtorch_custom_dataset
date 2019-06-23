cmake_minimum_required(VERSION 3.10)

find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES})