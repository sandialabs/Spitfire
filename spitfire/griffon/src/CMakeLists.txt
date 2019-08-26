include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB HDRS ${PROJECT_SOURCE_DIR}/include/*.h)
file(GLOB SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(griffon_cpp STATIC ${SRCS})
target_link_libraries(griffon_cpp ${BLASLAPACK})

install(TARGETS griffon_cpp DESTINATION ${CMAKE_BINARY_DIR}/lib)
install(FILES ${HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)
