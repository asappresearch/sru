cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(sru_cpp_example)

find_package(Torch REQUIRED)

# Define sru_cpu and link against LibTorch
add_library(sru_cpu SHARED sru_cpu_impl.cpp)
target_compile_features(sru_cpu PRIVATE cxx_std_14)
target_link_libraries(sru_cpu "${TORCH_LIBRARIES}")

message(STATUS "  CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "  TORCH LIBS: ${TORCH_LIBRARIES}")

# Define sru_gpu and link against LibTorch
find_package(CUDA)
if (CUDA_FOUND)
    # When libtorch is built with the old GCC ABI, dependent libraries must too.
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if(DEFINED GLIBCXX_USE_CXX11_ABI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}" )
        endif()
    endif()
    message(STATUS "  GLIBCXX_USE_CXX11_ABI: ${GLIBCXX_USE_CXX11_ABI}")
    message(STATUS "  CXXFLAGS: ${CMAKE_CXX_FLAGS}")

    CUDA_ADD_LIBRARY(sru_cuda SHARED sru_cuda_kernel.cu sru_cuda_impl.cpp)
    target_compile_features(sru_cuda PRIVATE cxx_std_14)
    target_link_libraries(sru_cuda "${TORCH_LIBRARIES}")
else()
    message(STATUS "CUDA not found. Use dummy SRU_CUDA implementation.")
    add_library(sru_cuda SHARED sru_cuda_impl_dummy.cpp)
    target_compile_features(sru_cuda PRIVATE cxx_std_14)
    target_link_libraries(sru_cuda "${TORCH_LIBRARIES}")
endif()

add_executable(example_app main_test_cpp.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
if (UNIX AND NOT APPLE)
    target_link_libraries(example_app -Wl,--no-as-needed sru_cpu)
    target_link_libraries(example_app -Wl,--no-as-needed sru_cuda)
else()
    target_link_libraries(example_app -Wl,-all_load sru_cpu)
    target_link_libraries(example_app -Wl,-all_load sru_cuda)
endif()
target_compile_features(example_app PRIVATE cxx_std_14)
