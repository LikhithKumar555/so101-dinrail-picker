cmake_minimum_required(VERSION 3.10)
project(so101_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/compare_ik_moveit.py scripts/generate_acm.py DESTINATION lib/${PROJECT_NAME})

ament_package()
