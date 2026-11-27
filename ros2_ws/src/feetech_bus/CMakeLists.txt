cmake_minimum_required(VERSION 3.16)
project(feetech_bus CXX)

# Pure C++17, no ROS dependency: builds inside a colcon workspace (ament) AND as a plain CMake
# project (useful for CI or a quick check without ROS).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(feetech_bus src/protocol.cpp src/serial_port.cpp src/bus.cpp)
# the ros2_control plugin (a shared library) will link this static library -> needs -fPIC
set_target_properties(feetech_bus PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(feetech_bus PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_executable(feetech_cli tools/feetech_cli.cpp)
target_link_libraries(feetech_cli feetech_bus)

find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  install(TARGETS feetech_bus EXPORT export_feetech_bus
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
  install(TARGETS feetech_cli RUNTIME DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)
  ament_export_targets(export_feetech_bus HAS_LIBRARY_TARGET)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_protocol test/test_protocol.cpp)
    target_link_libraries(test_protocol feetech_bus)
    ament_add_gtest(test_bus_pty test/test_bus_pty.cpp)
    target_link_libraries(test_bus_pty feetech_bus util pthread)
  endif()
  ament_package()
else()
  # plain CMake: fetch GoogleTest
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  foreach(t test_protocol test_bus_pty)
    add_executable(${t} test/${t}.cpp)
    target_link_libraries(${t} feetech_bus GTest::gtest_main util pthread)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
