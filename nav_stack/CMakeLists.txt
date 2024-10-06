cmake_minimum_required(VERSION 3.8)
project(nav_stack)

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
  launch maps costmap config global_planner local_planner
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  costmap/occupancy_grid.py
  global_planner/a_star.py
  global_planner/path_plan.py
  local_planner/dwa_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
