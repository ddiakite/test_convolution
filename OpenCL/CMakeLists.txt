cmake_minimum_required(VERSION 3.8)

project(con2D LANGUAGES CXX CUDA)


set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_VERBOSE_MAKEFILE TRUE)


set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

#find_package(OpenCL REQUIRED)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/inc" "$ENV{INTELFPGAOCLSDKROOT}/host/include" "$ENV{INTELFPGAOCLSDKROOT}/examples_aoc/common/inc" "$ENV{INTELFPGAOCLSDKROOT}/examples_aoc/common/inc/AOCLUtils")

file(GLOB CPP_FILES ${ROOT_DIR}/*.cpp)
#file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h $ENV{INTELFPGAOCLSDKROOT}/examples_aoc/common/inc/AOCLUtils/*.h)

add_library(Conv ${CPP_FILES} ${HEADER_FILES})
target_compile_features(Conv PUBLIC cxx_std_11)




add_executable(convolution "${ROOT_DIR}/main.cpp")


target_link_libraries(convolution ${CUDA_LIBRARY} ${TBB_LIBRARY} ${Boost_LIBRARIES} rt pthread Conv OpenCL freeimage)
