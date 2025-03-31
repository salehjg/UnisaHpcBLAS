cmake_minimum_required(VERSION 3.10)
project(MyBlas LANGUAGES CXX C) # if you don't include C, openblas won't find openmp_c!
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)






find_package(OpenMP REQUIRED)
find_package(OpenMP COMPONENTS C) # Explicitly find OpenMP_C
if (OpenMP_C_FOUND)
    message(STATUS "OpenMP_C found")
else()
    message(STATUS "OpenMP_C not found")
endif()
find_package(OpenBLAS REQUIRED)



add_subdirectory(common)
add_subdirectory(src)
add_subdirectory(benchmark)
