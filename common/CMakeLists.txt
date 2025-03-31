add_library(common STATIC common.h)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CXX)
#target_compile_options(common PRIVATE ${DF_FLAGS})

target_sources(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/aligned_tensor.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timer_scope.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timer_stats.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/common.h
        ${CMAKE_CURRENT_LIST_DIR}/aligned_tensor.h
        ${CMAKE_CURRENT_LIST_DIR}/timer_scope.h
        ${CMAKE_CURRENT_LIST_DIR}/timer_stats.h
)
target_include_directories(common
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
#target_link_libraries(common
#        #PRIVATE
#        #
#)