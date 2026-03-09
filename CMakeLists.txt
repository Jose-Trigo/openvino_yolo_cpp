cmake_minimum_required(VERSION 3.16)

project(ov_yolo_video CXX)

# ---------------- C++ standard ----------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- Find packages ----------------
find_package(OpenVINO REQUIRED)       # OpenVINO runtime & headers
find_package(OpenCV REQUIRED)

# ---------------- Create executable ----------------
add_executable(ov_yolo_video 
    main.cpp
    inference.cpp
)

# ---------------- Include directories ----------------
target_include_directories(ov_yolo_video
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}    # For inference.h
        ${OpenCV_INCLUDE_DIRS}
        ${OpenVINO_INCLUDE_DIRS}
)

# ---------------- Link libraries ----------------
target_link_libraries(ov_yolo_video
    PRIVATE
        ${OpenCV_LIBS}
        openvino::runtime
)