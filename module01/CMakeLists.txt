add_executable(module01 "")
set_target_properties(module01 PROPERTIES LINKER_LANGUAGE CXX)
#target_compile_options(common PRIVATE ${DF_FLAGS})

# find package for OpenBLAS
find_package(OpenBLAS REQUIRED)

target_sources(module01
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/main.cc
)
target_include_directories(module01
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
target_link_libraries(module01
        PRIVATE
        common
        OpenBLAS::OpenBLAS
)