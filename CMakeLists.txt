cmake_minimum_required(VERSION 3.13.0)

project(gyroScatterEff VERSION 1.0.0 LANGUAGES CXX)

find_package(Omega_h REQUIRED)
#Clear the omegah compilation flags that it passes to cuda.  Using the
# kokkos target, and nvcc_wrapper, provide sufficient flags.
set_property(TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS "")

option(ENABLE_CABANA "Build with Cabana" OFF)
if(ENABLE_CABANA)
  find_package(Cabana REQUIRED)
  add_definitions(-DENABLE_CAB)
endif()

set(DATA_DIR "" CACHE PATH "Path to a local copy of the gyroScatter dataset.")
if(NOT EXISTS ${DATA_DIR})
  message(FATAL_ERROR "DATA_DIR \"${DATA_DIR}\" is not accessible")
endif()

#Settings options for testing
enable_testing()
include(CTest)

option(IS_TESTING "Build for CTest" OFF)
message(STATUS "IS_TESTING: ${IS_TESTING}")

add_executable(gyroScatterEff gyroScatterEff.cpp)
target_link_libraries(gyroScatterEff Omega_h::omega_h)
if(ENABLE_CABANA)
  target_link_libraries(gyroScatterEff Cabana::cabana)
endif()
