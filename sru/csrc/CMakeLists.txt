cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(foo)

find_package(Torch REQUIRED)

# Define our library target
add_library(sru_cpu SHARED sru_cpu_impl.cpp)
# Enable C++14
target_compile_features(sru_cpu PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(sru_cpu "${TORCH_LIBRARIES}")

add_executable(example_app main_test_cpp.cpp)
target_link_libraries(example_app "${TORCH_LIBRARIES}")
if (UNIX AND NOT APPLE)
    target_link_libraries(example_app -Wl,--no-as-needed sru_cpu)
else()
    target_link_libraries(example_app -Wl,-all_load sru_cpu)
endif()
target_compile_features(example_app PRIVATE cxx_std_14)
