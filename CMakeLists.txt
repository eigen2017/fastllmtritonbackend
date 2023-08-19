cmake_minimum_required(VERSION 3.5)

project(glmbackendcmakeproject LANGUAGES C CXX)

option(TRITON_ENABLE_GPU "enable gpu support in backend" OFF)
option(TRITON_ENABLE_STATS "include statistics collections in backend" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

FetchContent_Declare(
    repo-common
    PREFIX repo-common
    SOURCE_DIR ../../common-main
)
FetchContent_Declare(
    repo-core
    PREFIX repo-core
    SOURCE_DIR ../../core-main
)
FetchContent_Declare(
    repo-backend
    PREFIX repo-backend
    SOURCE_DIR ../../backend-main
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

configure_file(src/libtriton_glmbackend.ldscript libtriton_glmbackend.ldscript COPYONLY)

add_library(
    triton_glmbackend SHARED
    src/glmbackend.cc
)

enable_language(CUDA)
add_compile_definitions(USE_CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")

target_include_directories(
    triton_glmbackend
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /models/glm/fastllm-master/include
    /models/glm/fastllm-master/include/utils
    /models/glm/fastllm-master/include/models
    /models/glm/fastllm-master/include/devices/cuda
)

target_compile_features(triton_glmbackend PRIVATE cXX_std_11)
target_compile_options(
    triton_glmbackend PRIVATE
    -pthread --std=c++17 -march-native -g
)

target_link_libraries(
    triton_glmbackend
    PRIVATE
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    triton-backend-utils
    /models/glm/fastllm-master/build/libfastllm.so
)

set_target_properties (
    triton_glmbackend PROPERTIES
    PROITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_glmbackend
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_glmbackend.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_glmbackend.ldscript"
)


