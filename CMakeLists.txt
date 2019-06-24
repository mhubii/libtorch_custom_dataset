cmake_minimum_required(VERSION 3.10)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(train train.cpp)
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(classify classify.cpp)
target_link_libraries(classify ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})