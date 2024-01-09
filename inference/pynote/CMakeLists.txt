cmake_minimum_required(VERSION 3.28.1 FATAL_ERROR)
project(pynote)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find_package(SndFile REQUIRED)
set(SNDFILE_INCLUDE_DIR /usr/include)
set(SNDFILE_LIBRARY /usr/lib/x86_64-linux-gnu/libsndfile.so)

# Get a list of all .cpp files in the complete_work/utils directory
file(GLOB SOURCE_FILES "complete_work/*.cpp")

add_executable(example-app ${SOURCE_FILES})
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${SNDFILE_LIBRARY}")
target_include_directories(example-app PUBLIC ${SNDFILE_INCLUDE_DIR})
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)