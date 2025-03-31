
add_executable(benchmark_example example.cpp)
target_link_libraries(benchmark_example
        PRIVATE
        common
        blas
        OpenBLAS::OpenBLAS
)