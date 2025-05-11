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
import numpy as np
from tf_transformations import quaternion_from_euler,euler_from_quaternion
from a_star import a_star
from rclpy.lifecycle.node import LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleNode
from pynav_interfaces.msg import GoalPose
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# from nav_msgs.msg import GL

class GlobalPath(LifecycleNode):
    def __init__(self):
        super().__init__("a_star_planner")

        self.get_logger().info("PLANNER UNCONFIGURED")
        self.res = 0.05
        self.path = None
        self.occupancy_grid = None

    '''
    UNCONFIGURE TO INACTIVE
    '''
    def on_configure(self, previous_state: LifecycleState):

        self.ogrid_sub_=self.create_subscription(OccupancyGrid, "/occupancy_grid",self.ogrid_callback,10)
        self.odom_sub=self.create_subscription(Odometry, "/odometry/filtered",self.odom_callback,10)
        self.goal_pose_sub=self.create_subscription(GoalPose, "pynav/goal_pose",self.goal_pose_callback,10)
        
        self.path_publisher = self.create_lifecycle_publisher(Path, "/global_path",10)
        self.path_timer_ = self.create_timer(0.5, self.publish_path)
        self.path_timer_.cancel()

        callback_group=ReentrantCallbackGroup()

        # self.path_planner =  AStar()
        self.declare_parameter('initial_pose', [0.0, 0.0])
        
        self.initial_pose = np.array(self.get_parameter('initial_pose').get_parameter_value().double_array_value)

        # node has successfully tranfered to inactive
        self.get_logger().info("PLANNER INACTIVE")
        return TransitionCallbackReturn.SUCCESS
    
    '''
    INACTIVE TO ACTIVE
    '''
    def on_activate(self, previous_state: LifecycleState):
        self.path_timer_.reset()

        # call super on activate to tell the node that it should be active
        self.get_logger().info("PLANNER ACTIVE")
        return super().on_activate(previous_state)
    
    def on_deactivate(self, previous_state: LifecycleState):
        self.get_logger().info("PLANNER DEACTIVATED")
        self.path_timer_.cancel()

        return super().on_deactivate(previous_state)
    
    '''
    ANY STATE TO FINALIZED 
    '''
    def on_shutdown(self, state: LifecycleState):
    
        self.destroy_node("a_star_planner")
        self.get_logger().info("PLANNER KILLED")
        return TransitionCallbackReturn.SUCCESS 
    
    def odom_callback(self, odom_msg : Odometry):
        self.robot_x = odom_msg.pose.pose.position.x
        self.robot_y = odom_msg.pose.pose.position.y

        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z
        w = odom_msg.pose.pose.orientation.w

        self.robot_yaw = euler_from_quaternion([x,y,z,w])[2]
    
    def goal_pose_callback(self, goal_pose : GoalPose):
        self.goal_pose = np.array([goal_pose.goal_pose_x, goal_pose.goal_pose_y])
        # self.get_logger().info(f'{self.goal_pose}')

    def ogrid_callback(self, ogrid_msg : OccupancyGrid):
        self.res = ogrid_msg.info.resolution
        occupancy_grid = np.reshape(np.array(ogrid_msg.data), (ogrid_msg.info.height, ogrid_msg.info.width))
        self.occupancy_grid = occupancy_grid.transpose()

    def compute_path(self):
    
        if self.occupancy_grid is not None:
            initial_pose = np.array([self.robot_x, self.robot_y])
            self.get_logger().info(f"{initial_pose}")

            initial_pose = (initial_pose/self.res).astype("int32")
            goal_pose = (self.goal_pose/self.res).astype("int32")

            # plt.imshow(self.occupancy_grid, cmap='gray_r', interpolation='nearest')
            # plt.show()  

            grid_size = self.occupancy_grid.shape
            start_node = initial_pose[0] * grid_size[1] + initial_pose[1]
            goal_node = goal_pose[0] * grid_size[1] + goal_pose[1]

            self.path = np.array(a_star(start_node, goal_node, self.occupancy_grid))

            if self.path is not None:
                self.path = self.path * 0.05

            else:
                self.get_logger().error("NO PATH FOUND")
    
    def publish_path(self):
        path_msg = Path()
        path_stamped = []
        self.compute_path()
        self.get_logger().info("GLOBAL PATH")
        if self.path is not None:

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
            
def main(args=None):
    rclpy.init(args=args)
    node=GlobalPath()
    # executor = MultiThreadedExecutor(num_threads=5)
    # executor.add_node(node)
    rclpy.spin(node)
    rclpy.shutdown()

# checks if the script is run directly, if not calls the main function
if __name__ == '__main__':
    main()

