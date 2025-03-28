cmake_minimum_required(VERSION 3.10)
project(HpcCourseProjects LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Options
set(USE_OPENMP ON CACHE BOOL "Use OpenMP for parallelization")
set(BUILD_MODULE01 ON CACHE BOOL "Build Module 01")



# Handle OpenMP
find_package(OpenMP)
if (USE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found")
endif()

add_subdirectory(common)

# Handle Module 01
if (BUILD_MODULE01)
    add_subdirectory(module01)
endif()