cmake_minimum_required(VERSION 3.10)
project(so101_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generates C++ and Python code for the message at build time
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ServoTelemetry.msg"
  DEPENDENCIES std_msgs)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
