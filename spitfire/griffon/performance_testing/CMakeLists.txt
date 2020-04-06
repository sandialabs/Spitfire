include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(mmw_timer ${PROJECT_SOURCE_DIR}/performance_testing/mmw_timer.cpp)
target_link_libraries(mmw_timer griffon_cpp ${BLASLAPACK})
install(TARGETS mmw_timer DESTINATION ${CMAKE_BINARY_DIR}/bin)