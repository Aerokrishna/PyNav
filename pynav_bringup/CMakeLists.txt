cmake_minimum_required(VERSION 3.8)
project(pynav_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch config rviz launch maps pynav_manager
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  pynav_manager/pynav_manager.py
  pynav_client/pynav_client.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()