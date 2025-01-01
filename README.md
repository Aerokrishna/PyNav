# Autonomous Navigation Stack Python

This is repository contains the code base required for autonomous navigation of mobile robots utilizing the state of the art path planning and control algorithms coded in python. This code base has been tested and validated using Turtlebot3 robot simulated on Gazebo. But it can be adapted for any differential drive robot.



https://github.com/user-attachments/assets/98c0b61e-c397-4ffa-874f-f0255a87a9ad


# About the Navigation Stack

**Costmap** 

An occupancy grid map image along with a yaml file, generated as a result of SLAM, is provided to the stack. The occupancy_grid node converts the occupancy grid into a three layered costmap by inflating the obstacles. The costmap is updated with a fixed frequency.

**Global Planner** 

The global planner takes the costmap and goal position as an input. It plans a path consisting of a number stamped states. The global path is updated with respect to the bot's position. The global planners currently present in the stack are :
1) A-star
2) RRT*

**Local Planner** 

The local planner takes real-time feedback from the on-board sensors to avoid unexpected and dynamic obstacles. It generates short term trajectories considering the kinematic constraints of the robot as well as ensuring obstacles avoidance. It computes the control commands, linear velocity and angular velocity, for the robot. The local planners currently present in the stack are :
1) Dynamic Window Approach Controller

**Feedback** 
The robot must publish its position and orientation. The lidar feedbac is received in the form of range and angle values. The robot receives the control commands and converts them into wheel velocities with a lower level gazebo controller plugin.

https://youtu.be/oVkC_Z6I-0c
