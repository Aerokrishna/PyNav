cmake_minimum_required(VERSION 3.8)
project(pynav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C99
set(CMAKE_C_STANDARD 99)


# Default to C++14
set(CMAKE_CXX_STANDARD 14)
# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch costmap planner controller
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  costmap/costmap.py
  planner/a_star.py
  planner/a_star_planner.py
  controller/dwa_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
