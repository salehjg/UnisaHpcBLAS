add_library(blas "")
set_target_properties(blas PROPERTIES LINKER_LANGUAGE CXX)
#target_compile_options(blas PRIVATE ${DF_FLAGS})

target_sources(blas
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/interface.cpp
        ${CMAKE_CURRENT_LIST_DIR}/impl_scalar.cpp
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/interface.h
        ${CMAKE_CURRENT_LIST_DIR}/impl_scalar.h
)
target_include_directories(blas
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)
#target_link_libraries(blas
#        #PRIVATE
#        #
#)