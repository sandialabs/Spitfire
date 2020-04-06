include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB PERF_HDRS ${PROJECT_SOURCE_DIR}/performance_testing/*.h)
file(GLOB PERF_SRCS ${PROJECT_SOURCE_DIR}/performance_testing/*.cpp)
add_library(griffon_cpp_perf_testing STATIC ${PERF_SRCS})
target_link_libraries(griffon_cpp_perf_testing griffon_cpp ${BLASLAPACK})
install(TARGETS griffon_cpp_perf_testing DESTINATION ${CMAKE_BINARY_DIR}/lib)
install(FILES ${PERF_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)

# ----------------------------------------------------------------

add_executable(mmw_timer ${PROJECT_SOURCE_DIR}/performance_testing/mmw_timer.cpp)
target_link_libraries(mmw_timer griffon_cpp_perf_testing griffon_cpp ${BLASLAPACK})
install(TARGETS mmw_timer DESTINATION ${CMAKE_BINARY_DIR}/bin)

add_executable(flamelet_rhs_timer ${PROJECT_SOURCE_DIR}/performance_testing/flamelet_rhs_timer.cpp)
target_link_libraries(flamelet_rhs_timer griffon_cpp_perf_testing griffon_cpp ${BLASLAPACK})
install(TARGETS flamelet_rhs_timer DESTINATION ${CMAKE_BINARY_DIR}/bin)