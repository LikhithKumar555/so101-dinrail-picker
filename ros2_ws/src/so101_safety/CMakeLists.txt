cmake_minimum_required(VERSION 3.10)
project(so101_safety)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(so101_interfaces REQUIRED)

# 1) Pure C++ library with the safety rules (no ROS) -> unit-testable
add_library(safety_logic src/safety_logic.cpp)
target_include_directories(safety_logic PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# 2) The ROS node that wraps it
add_executable(safety_monitor src/safety_monitor_node.cpp)
target_link_libraries(safety_monitor safety_logic)
ament_target_dependencies(safety_monitor
  rclcpp std_msgs std_srvs diagnostic_msgs controller_manager_msgs so101_interfaces)

install(TARGETS safety_logic safety_monitor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/fake_telemetry.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_safety_logic test/test_safety_logic.cpp)
  target_link_libraries(test_safety_logic safety_logic)
endif()

ament_package()
