cmake_minimum_required(VERSION 3.16)
project(ov_yolo_video CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED)

add_executable(ov_yolo_video main.cpp)

target_link_libraries(ov_yolo_video
    PRIVATE
        ${OpenCV_LIBS}
        openvino::runtime
)


target_include_directories(ov_yolo_video
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        C:/Intel/openvino_2026/runtime/include
)

