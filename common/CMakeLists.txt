add_library(common STATIC common.h)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)
#target_compile_options(common PRIVATE ${DF_FLAGS})

target_sources(common
        #PRIVATE
        #
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/common.h
)
target_include_directories(common
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
#target_link_libraries(common
#        #PRIVATE
#        #
#)