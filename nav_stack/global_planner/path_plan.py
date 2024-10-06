#!/usr/bin/env python3
#publisher with timer
import rclpy
import rclpy.clock
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import rclpy.time
import cv2
import numpy as np
import pkg_resources
from ament_index_python.packages import get_package_share_directory
import os
import matplotlib.pyplot as plt
from tf_transformations import quaternion_from_euler,euler_from_quaternion
from a_star import a_star
from visualization_msgs.msg import Marker


# from nav_msgs.msg import GL

class PlanGlobalPath(Node):
    def __init__(self):
        super().__init__("min_publisher")
        self.ogrid_pub_=self.create_subscription(OccupancyGrid, "occupancy_grid",self.ogrid_callback,10)
        self.odom_sub=self.create_subscription(Odometry, "/odom",self.odom_callback,10)

        self.marker_publisher_ = self.create_publisher(Marker,'/visualization_marker',10)
        self.path_publisher = self.create_publisher(Path, "global_path",10)
        
        self.marker_timer_ = self.create_timer(0.1, self.marker_timer)
        self.path_timer_ = self.create_timer(1.0, self.publish_path)


        # self.path_planner =  AStar()
        self.occupancy_grid = []

        self.initial_pose = np.array([0.5, 0.5])
        self.goal_pose = np.array([3.8,2.6])
        self.res = 0.05
        self.path = []

        self.robot_x = 0.0
        self.robot_y = 0.0

        self.path_pub_timer = self.create_timer(1.0, self.compute_path)

    def odom_callback(self, odom_msg : Odometry):
        self.robot_x = odom_msg.pose.pose.position.x
        self.robot_y = odom_msg.pose.pose.position.y

        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z
        w = odom_msg.pose.pose.orientation.w

        self.robot_yaw = euler_from_quaternion([x,y,z,w])[2]

    def ogrid_callback(self, ogrid_msg : OccupancyGrid):
        self.res = ogrid_msg.info.resolution
        self.occupancy_grid = np.reshape(np.array(ogrid_msg.data), (ogrid_msg.info.height, ogrid_msg.info.width))

    def compute_path(self):
        path_msg = Path()
        
        if self.occupancy_grid is not None:
            initial_pose = np.array([self.robot_x, self.robot_y])
            initial_pose = (initial_pose/self.res).astype("int32")

            goal_pose = (self.goal_pose/self.res).astype("int32")
            self.occupancy_grid = self.occupancy_grid.transpose()

            # plt.imshow(self.occupancy_grid, cmap='gray_r', interpolation='nearest')
            # plt.show()  

            grid_size = self.occupancy_grid.shape
            start_node = initial_pose[0] * grid_size[1] + initial_pose[1]
            goal_node = goal_pose[0] * grid_size[1] + goal_pose[1]

            self.path = np.array(a_star(start_node, goal_node, self.occupancy_grid))

            if self.path is not None:

                self.path = self.path * 0.05
                # print(self.path)

            
            else:
                self.get_logger().error("NO PATH FOUND")
    
    def publish_path(self):
        path_msg = Path
        path_stamped = []
        for i in range(len(self.path)):
            pose = PoseStamped()
            pose.header = Header()
            pose.header.frame_id = 'map_frame'
            pose.header.stamp = self.get_clock().now().to_msg()

            # Create a simple straight line path with some small deviation
            pose.pose.position.x = self.path[i][0]
            pose.pose.position.y = self.path[i][1]

            # Orientation (no rotation in this example)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 0.0

            path_stamped.append(pose)
        
        path_msg = Path()
        path_msg.header.frame_id = 'map_frame'  # Set the reference frame for the path
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = path_stamped
        self.path_publisher.publish(path_msg)

        # print(path)
    def marker_timer(self):
        marker_msg = Marker()
        marker_msg.header.frame_id = "map_frame" # wrt which frame we are taking the coordinates
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.type = Marker.SPHERE
    
        marker_msg.scale.x = 0.07
        marker_msg.scale.y = 0.07
        marker_msg.scale.z = 0.07

        marker_msg.color.r = 255.0
        marker_msg.color.g = 0.0
        marker_msg.color.b = 0.0
        marker_msg.color.a = 1.0


        for i in range(len(self.path)):
          
            marker_msg.pose.position.x = self.path[i][0]
            marker_msg.pose.position.y = self.path[i][1]
            marker_msg.pose.orientation.w = 0.0

            marker_msg.id = i
            # self.marker_publisher_.publish(marker_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node=PlanGlobalPath()
    rclpy.spin(node)
    rclpy.shutdown()

# checks if the script is run directly, if not calls the main function
if __name__ == '__main__':
    main()

