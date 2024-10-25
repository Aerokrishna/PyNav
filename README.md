# Autonomous Navigation Stack Python

This is repository contains the code base required for autonomous navigation of mobile robots utilizing the state of the art path planning and control algorithms coded in python. This code base has been tested and validated using Turtlebot3 robot simulated on Gazebo. But it can be adapted for any differential drive robot.

![navstack2](https://github.com/user-attachments/assets/b99a608c-9f4b-45a3-9a3d-6879694a94dd)

-> make a video which demonstrates working of the global and local planner

# About the Navigation Stack

1) **Costmap** : An occupancy grid map image along with a yaml file, generated as a result of SLAM, is provided to the stack. The occupancy_grid node converts the occupancy grid into a three layered costmap by inflating the obstacles. The costmap is updated with a fixed frequency.
2) **Global Planner** : The global planner takes the costmap and goal position as an input. It plans a path consisting of a number stamped states. The global path is updated with respect to the bot's position.
3) **Local Planner** : The local planner takes real-time feedback from the on-board sensors to avoid unexpected and dynamic obstacles. It generates short term trajectories considering the kinematic constraints of the robot as well as ensuring obstacles avoidance. It computes the control commands, linear velocity and angular velocity, for the robot.
4) **Feedback** : The robot must publish its position and orientation. The lidar feedbac is received in the form of range and angle values. The robot receives the control commands and converts them into wheel velocities with a lower level gazebo controller plugin.

**Global Planners**
1) A-star

**Local Planners**
1) Dynamic Window Approach

https://youtu.be/oVkC_Z6I-0c
