cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(inference)

set(ONNXRUNTIME_INCLUDE_DIR /home/rohan/VAD/onnxruntime/include)
set(ONNXRUNTIME_LIBRARY_DIR /home/rohan/VAD/onnxruntime/build/Linux/Release)
set(SNDFILE_INCLUDE_DIR /usr/local/include)
set(CMAKE_PREFIX_PATH /home/rohan/VAD/inference/pynote/libtorch ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)

# Add the executable
add_executable(inference inference/inference.cpp)

# Include the headers
target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${SNDFILE_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})

# Link the libraries
target_link_libraries(inference "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so" sndfile "${TORCH_LIBRARIES}")

# Set the compile options
target_compile_options(inference PRIVATE ${TORCH_CXX_FLAGS})