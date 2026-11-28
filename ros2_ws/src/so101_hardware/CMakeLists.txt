cmake_minimum_required(VERSION 3.16)
project(so101_hardware CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(feetech_bus REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(so101_interfaces REQUIRED)

# Tested with ros2_control 4.50 (Jazzy). Older 4.x releases may lack parts of the API used here.
if(hardware_interface_VERSION VERSION_LESS 4.50)
  message(WARNING "hardware_interface ${hardware_interface_VERSION} found, 4.50+ expected: "
    "run 'sudo apt update && sudo apt upgrade' if the build fails")
endif()

# 1) ROS-free core: joint codec + whole-arm bus access. Static, position independent (it goes
#    into the plugin, which is a shared library).
add_library(so101_hardware_core STATIC src/joint_codec.cpp src/arm_bus.cpp)
set_target_properties(so101_hardware_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(so101_hardware_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(so101_hardware_core PUBLIC feetech_bus::feetech_bus)

# 2) the ros2_control plugin, loaded at run time by the controller manager (pluginlib)
add_library(so101_hardware SHARED src/feetech_system.cpp)
target_link_libraries(so101_hardware PUBLIC
  so101_hardware_core
  hardware_interface::hardware_interface
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${so101_interfaces_TARGETS})
pluginlib_export_plugin_description_file(hardware_interface so101_hardware.xml)

install(TARGETS so101_hardware
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(lifecycle_msgs REQUIRED)
  ament_add_gtest(test_joint_codec test/test_joint_codec.cpp)
  target_link_libraries(test_joint_codec so101_hardware_core)
  ament_add_gtest(test_arm_bus test/test_arm_bus.cpp)
  target_link_libraries(test_arm_bus so101_hardware_core util pthread)
  # loads the INSTALLED plugin through pluginlib, exactly like the controller manager
  ament_add_gtest(test_feetech_system test/test_feetech_system.cpp
    APPEND_ENV AMENT_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
    TIMEOUT 60)
  target_link_libraries(test_feetech_system so101_hardware_core
    hardware_interface::hardware_interface rclcpp::rclcpp rclcpp_lifecycle::rclcpp_lifecycle
    ${lifecycle_msgs_TARGETS} util pthread)
endif()

ament_package()
