cmake_minimum_required(VERSION 2.6)
project(aruco-detection)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(aruco-detection aruco-detection.cpp)
target_link_libraries(aruco-detection ${OpenCV_LIBS})

add_executable(image-capture image-capture.cpp)
target_link_libraries(image-capture ${OpenCV_LIBS})