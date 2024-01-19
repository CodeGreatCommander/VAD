cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(inference)

set(ONNXRUNTIME_INCLUDE_DIR ../../dependencies/onnxruntime/include)
set(ONNXRUNTIME_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime/build/Linux/Release)
# set(SNDFILE_INCLUDE_DIR /usr/local/include)
set(SNDFILE_INCLUDE_DIR ../../dependencies/libsndfile/include)
set(CMAKE_PREFIX_PATH ../../dependencies/libtorch ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)

# Add the executable
add_executable(inference inference/inference.cpp)

# Include the headers
target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR} ${SNDFILE_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})

# Link the libraries
target_link_libraries(inference "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so" sndfile "${TORCH_LIBRARIES}")

# Set the compile options
target_compile_options(inference PRIVATE ${TORCH_CXX_FLAGS})