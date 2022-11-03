cmake_minimum_required(VERSION 3.13.0)

project(gyroScatterEff VERSION 0.3.0 LANGUAGES CXX)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

find_package(Cabana REQUIRED)
find_package(meshFields CONFIG REQUIRED)

set(DATA_DIR "" CACHE PATH "Path to a local copy of the gyroScatter dataset.")
if(NOT EXISTS ${DATA_DIR})
  message(FATAL_ERROR "DATA_DIR \"${DATA_DIR}\" is not accessible. See README.md "
                      "for instructions on downloading the input data.")
endif()

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

add_executable(gyroScatterEff gyroScatterEff.cpp)
target_include_directories(gyroScatterEff PUBLIC ${DATA_DIR})
target_link_libraries(gyroScatterEff Omega_h::omega_h)
target_link_libraries(gyroScatterEff Cabana::cabanacore)
target_link_libraries(gyroScatterEff meshFields::meshFields)
target_compile_definitions(gyroScatterEff PUBLIC ENABLE_CABANA)
