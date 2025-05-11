# a launch file that launches the navigation stack
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the YAML parameter file
    param_file_path = os.path.join(
        get_package_share_directory('pynav_bringup'),  # Replace 'my_package' with your package name
        'config',
        'pynav.yaml'
    )

    # Path to the RViz config file (optional, for a custom RViz layout)
    rviz_config_path = os.path.join(
        get_package_share_directory('pynav_bringup'),  # Replace 'my_package' with your package name
        'rviz',
        'navigation.rviz'
    )

    # Node definition: occupancygrid.py
    occupancy_grid_node = Node(
        package='pynav',            # Replace with your package name
        executable='costmap.py',   # Replace with your Python node file name
        name='costmap',
        output='screen',
        parameters=[param_file_path]
    )

    # Node definition: occupancygrid.py
    a_star_planner_node = Node(
        package='pynav',            # Replace with your package name
        executable='a_star_planner.py',   # Replace with your Python node file name
        name='a_star_planner',
        output='screen',
        parameters=[param_file_path]
    )
    # Node definition: occupancygrid.py
    dwa_controller_node = Node(
        package='pynav',            # Replace with your package name
        executable='dwa_planner.py',   # Replace with your Python node file name
        name='dwa_planner',
        output='screen',
        parameters=[param_file_path]
    )
    # Node definition: occupancygrid.py
    navigation_manager_node = Node(
        package='pynav_bringup',            # Replace with your package name
        executable='pynav_manager.py',   # Replace with your Python node file name
        name='pynav_manager',
        output='screen',
        parameters=[param_file_path]
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path]  # Load RViz configuration file if available
    )
    return LaunchDescription([
        occupancy_grid_node,
        a_star_planner_node,
        dwa_controller_node,
        navigation_manager_node,
        # rviz_node
    ])
