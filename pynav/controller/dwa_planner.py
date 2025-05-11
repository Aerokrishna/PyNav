#!/usr/bin/env python3
#publisher with timer
import rclpy
import rclpy.clock
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf2_geometry_msgs import PointStamped
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from std_msgs.msg import Header
import rclpy.time
import numpy as np
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.lifecycle.node import LifecycleState, TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode
from pynav_interfaces.msg import GoalPose
import time

# DWA PLANNER FOR A DIFFERENTIAL-DRIVE ROBOT

class DWAParameters:
    def __init__(self, node: Node):

        # Fetch parameters or use defaults
        self.min_vel = node.get_parameter("min_vel").value
        self.max_vel = node.get_parameter("max_vel").value
        self.min_w = node.get_parameter("min_w").value
        self.max_w = node.get_parameter("max_w").value
        self.max_accel = node.get_parameter("max_accel").value
        self.max_w_accel = node.get_parameter("max_w_accel",).value
        self.time_step = node.get_parameter("time_step").value
        self.time_period = node.get_parameter("time_period").value
        self.v_resolution = node.get_parameter("v_resolution").value
        self.w_resolution = node.get_parameter("w_resolution").value
        self.speed_cost_gain = node.get_parameter("speed_cost_gain",).value
        self.obstacle_cost_gain = node.get_parameter("obstacle_cost_gain").value
        self.goal_cost_gain = node.get_parameter("goal_cost_gain").value
        
class DWAFunctions(Node):
    def __init__(self):
        self.kp = 0
    
    def get_dynamic_window(self, state, params):

        # Dynamic window from robot specification
        Vs = [params.min_vel, params.max_vel,
            params.min_w, params.max_w]

        # Dynamic window from motion model
        Vd = [state[3] - params.max_accel * params.time_step,
            state[3] + params.max_accel * params.time_step,
            state[4] - params.max_w_accel * params.time_step,
            state[4] + params.max_w_accel * params.time_step] # getting the maximum and minimum values of final velocities achievable with given acceleration

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        
        # when given maximum decceleration to the current velocity, if the final velocity becomes less than the vmin of robot, limit it to v_min
        # when given maximum acceleration to current velocity, the final velocity should be within vmax

        return dw

    def motion(self, state, control, dt):
        
        # calculate final state of the robot with respect to the given time and control command. Using differential drive model

        state[2] += control[1] * dt
        state[0] += control[0] * np.cos(state[2]) * dt
        state[1] += control[0] * np.sin(state[2]) * dt
        state[3] = control[0]
        state[4] = control[1]

        return state

    def predict_trajectory(self, current_state, v, w, params):
        
        state = np.array(current_state)

        trajectory = np.array(state)
        time = 0

        while time < params.time_period:
            state = self.motion(state, [v, w], params.time_step)

            # stack the positions to form a trajectory
            trajectory = np.vstack((trajectory, state))
            time += params.time_step

        return trajectory


class DWAPlanner(LifecycleNode):
    def __init__(self):
        super().__init__("dwa_planner")

        '''
        PARAMS 
        '''

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ("min_vel", 0.0),
                ("max_vel", 0.22),
                ("min_w", -1.84),
                ("max_w", 1.84),
                ("max_accel", 0.3),
                ("max_w_accel", 1.0),
                ("time_step", 0.5),
                ("time_period", 5.0),
                ("v_resolution", 0.05),
                ("w_resolution", 0.05),
                ("speed_cost_gain", 10.0),
                ("obstacle_cost_gain", 0.5),
                ("goal_cost_gain", 1.0),
            ]
        )

        self.v = 0.0
        self.w = 0.0

        self.params = DWAParameters(self)

        self.dwa_functions = DWAFunctions()
        callback_group=ReentrantCallbackGroup()

        # transformation 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.reached_wp = False
        self.obstacles = []
        self.waypoint = (1.0,1.0)
        self.theta_goal = []
        self.local_path = []
        self.goal_pose = [0.0,0.0,0.0]
        self.prev_goal_pose = [1.0,0.0,0.0]
        self.cnt = 0


        self.get_logger().info("CONTROLLER UNCONFIGURED")
    
    '''
    UNCONFIGURE TO INACTIVE
    '''
    def on_configure(self, previous_state: LifecycleState):

        # subscribe to odom, scan, global_path
        self.scan_sub_ = self.create_subscription(LaserScan, "/scan",self.scan_callback, 10)
        self.odom_sub_ = self.create_subscription(Odometry,'/odometry/filtered',self.odom_callback, 10)
        self.path_publisher = self.create_subscription(Path, "global_path",self.path_callback, 10)
        self.goal_pose_sub=self.create_subscription(GoalPose, "pynav/goal_pose",self.goal_pose_callback,10)


        self.cmd_vel_pub_ = self.create_lifecycle_publisher(Twist, '/cmd_vel', 10)
        self.local_path_pub_ = self.create_lifecycle_publisher(Path, '/local_path', 10)
        self.local_path_timer = self.create_timer(0.2, self.publish_local_path)
        self.cmd_vel_timer = self.create_timer(0.02, self.publish_cmd_vel)

        self.cmd_vel_timer.cancel()
        self.local_path_timer.cancel()

        # node has successfully tranfered to inactive
        self.get_logger().info("CONTROLLER INACTIVE")
        return TransitionCallbackReturn.SUCCESS
    
    '''
    INACTIVE TO ACTIVE
    '''
    def on_activate(self, previous_state: LifecycleState):
        self.local_path_timer.reset()
        self.cmd_vel_timer.reset()

        # call super on activate to tell the node that it should be active
        self.get_logger().info("CONTROLLER ACTIVE")
        return super().on_activate(previous_state)
    
    def on_deactivate(self, previous_state: LifecycleState):
        self.get_logger().info("CONTROLLER DEACTIVATED")
        self.cmd_vel_timer.cancel()

        vel_msg = Twist()

        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0

        self.cmd_vel_pub_.publish(vel_msg)

        self.local_path_timer.cancel()

        return super().on_deactivate(previous_state)
    
    '''
    ANY STATE TO FINALIZED 
    '''
    def on_shutdown(self, state: LifecycleState):
        self.destroy_node("dwa_planner")
        self.get_logger().info("CONTROLLER HAS BEEN KILLED")
        return TransitionCallbackReturn.SUCCESS 
    
    def goal_pose_callback(self, goal_pose : GoalPose):
        self.goal_pose = np.array([goal_pose.goal_pose_x, goal_pose.goal_pose_y])
        if self.goal_pose[0] != self.prev_goal_pose[0] and self.goal_pose[1] != self.prev_goal_pose[1]:

            self.reached_wp = True

        # self.get_logger().info(f'{self.goal_pose}')

    def compute_local_path(self):
        max_range = 5

        state = [self.robot_x,self.robot_y,self.robot_yaw,0.0,0.0]

        dw = self.dwa_functions.get_dynamic_window(state, self.params)
        trajs, best_traj = self.local_plan(dw, state, self.params)

        self.obstacles = []
        # print("heading_space", heading_space)
        # heading_space = [285,75]
        heading_space = [len(self.range_data)-300, 300]
        for i in range(heading_space[1]):
            range_val = self.range_data[i]
            theta = self.angle_min + (i * self.angle_increament)

            if range_val < max_range:

                x,y = self.scan_to_pose(range_val, theta)
                self.obstacles.append((x,y))
        
        for j in range(heading_space[0], len(self.range_data)-1):
            range_val = self.range_data[j]
            theta = self.angle_min + (j * self.angle_increament)

            if range_val < max_range:

                x,y = self.scan_to_pose(range_val, theta)
                self.obstacles.append((x,y))

        # print("best trajecory",best_traj)
        self.local_path = best_traj
        self.v = best_traj[-1][3]
        self.w = best_traj[-1][4]

        if self.reached_wp == True:
            self.v = 0.0
            self.w = 0.0

        self.prev_goal_pose = self.goal_pose

        if self.counter_delay(40):
            self.reached_wp = False
            self.cnt = 0
        
        # if self.reached_goal == 1:
        #     self.v = 0.0
        #     self.w = 0.0
        # print("CONTROL COMMAND v = ", self.v, "w = ", self.w)

        # print(len(self.obstacles))
        # self.get_logger().info(f"the best trajectory is {best_traj}")

        # self.get_logger().info(f"trajectories {trajs}")
    def counter_delay(self, cnt):
        self.cnt += 1

        if self.cnt == cnt:
            return True
        else:
            return False
        
    def scan_callback(self, scan_msg : LaserScan):
        self.range_data = scan_msg.ranges
        self.angle_increament = scan_msg.angle_increment
        self.angle_min = scan_msg.angle_min
    
    def publish_local_path(self):
        self.compute_local_path()
        path_msg = Path
        path_stamped = []
        for i in range(len(self.local_path)):
            pose = PoseStamped()
            pose.header = Header()
            pose.header.frame_id = 'map_frame'
            pose.header.stamp = self.get_clock().now().to_msg()

            # Create a simple straight line local_path with some small deviation
            pose.pose.position.x = self.local_path[i][0]
            pose.pose.position.y = self.local_path[i][1]

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
        self.local_path_pub_.publish(path_msg)

    def publish_cmd_vel(self):
        vel_msg = Twist()

        vel_msg.linear.x = self.v
        vel_msg.angular.z = self.w

        self.cmd_vel_pub_.publish(vel_msg)

    def compute_vector(self, vect1, vect2):
        return np.arctan2((vect2[1] - vect1[1]),(vect2[0] - vect1[0])) 

    def scan_to_pose(self, range_val, theta):

        local_x = range_val * np.cos(theta)
        local_y = range_val * np.sin(theta)

        global_x,global_y = self.transform_point(local_x,local_y)
        # global_theta = np.arctan2(global_y - self.robot_y, global_x - self.robot_x)

        return global_x, global_y

    def transform_point(self, local_x = 0.0, local_y = 0.0):
        local_point = PointStamped()
        local_point.header.frame_id = "base_link"
        
        local_point.point.x = local_x  
        local_point.point.y = local_y  
        local_point.point.z = 0.0  
        
        now = self.get_clock().now()

        try:
            self.tf_buffer.lookup_transform("odom", "base_link",Time(nanoseconds=0),Duration(seconds=1.0))
            
            global_point = self.tf_buffer.transform(local_point, "odom")

            return global_point.point.x,global_point.point.y

        except:
            self.get_logger().error("error!")
            return 0.0,0.0

    def odom_callback(self, odom_msg : Odometry):

        self.robot_x = odom_msg.pose.pose.position.x
        self.robot_y = odom_msg.pose.pose.position.y

        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z
        w = odom_msg.pose.pose.orientation.w

        self.robot_yaw = euler_from_quaternion([x,y,z,w])[2]
        
    # path is defined as a set of waypoints
    # go through the array once and get the distances of all points from the robot
    # get the waypoint closest to the robot, this is the interm goal taken as reference for goal cost
    # while you are going through the array check if the robot has reached the proximity with any waypoint, if yes then remove the remaining waypoints before that point

    def path_callback(self, path_msg = Path):
        path = []

        self.theta_goal = []
        len_path = len(path_msg.poses)

        self.goal = [path_msg.poses[-1].pose.position.x,path_msg.poses[-1].pose.position.y]
        if len_path > 12:
            for i in range(1,12):
                wpx,wpy = path_msg.poses[i].pose.position.x,path_msg.poses[i].pose.position.y
                x = wpx - self.robot_x
                y = wpy - self.robot_y
                
                self.theta_goal.append(np.arctan2(y, x))

    def goal_cost(self, goal, trajectory):

        wp_cost = 0.0
        if self.theta_goal != []:
            for i in range(len(trajectory)):
                wp_cost += abs(self.theta_goal[i] - trajectory[i][2])
            goal_cost = wp_cost/10
            
        else:
            x = goal[0] - self.robot_x
            y = goal[1] - self.robot_y

            theta_goal = np.arctan2(y, x)
            goal_cost = abs(theta_goal - trajectory[-1][2])

        return goal_cost

    def obstacle_distance_cost(self, trajectory, obstacles):
        min_dis = np.inf
        min_traj = ()
        
        for obs in obstacles:
            
            if trajectory[-1][3] != 0:
                # dis = self.get_distance(traj[0], traj[1], obs[0], obs[1])
                dis = self.get_distance(trajectory[-1][0], trajectory[-1][1], obs[0], obs[1])
                # print(dis,trajectory[-1][0], trajectory[-1][1], obs[0], obs[1])

                if dis < min_dis:
                    min_dis = dis
                    min_traj = (trajectory[-1][0], trajectory[-1][1], obs[0], obs[1])
                    # print(1/min_dis,min_traj)

            else:
                return np.inf, min_traj

        return 1/min_dis,min_traj
 
    def get_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)    

    def local_plan(self, dw, state, params):
        current_state = state[:]
        min_cost = np.inf
        best_control = [0.0, 0.0]
        best_trajectory = np.array([current_state])

        # evaluate all trajectory with sampled input in dynamic window
        traj_lis = []
        velocity_ = np.arange(dw[0], dw[1], params.v_resolution)
        angular_velocity_ = np.arange(dw[2], dw[3], params.w_resolution)
  
        for v in velocity_:
            for w in angular_velocity_:
                trajectory = self.dwa_functions.predict_trajectory(current_state, v, w, params)

                traj_lis.append(trajectory)
                cost,pts = self.obstacle_distance_cost(trajectory, self.obstacles)
                obs_cost = params.obstacle_cost_gain * cost
                speed_cost = params.speed_cost_gain * (params.max_vel - trajectory[-1, 3])
                goal_cost = params.goal_cost_gain * (self.goal_cost(self.goal,trajectory))

                final_cost = goal_cost + speed_cost + obs_cost
                if final_cost < min_cost:
                    # print('goal cost',goal_cost)/
                    # print('speed cost',speed_cost)
                    # print('obstacle cost',obs_cost)
                    # print('final_cost',final_cost)

                    min_cost = final_cost
                    best_trajectory = trajectory

        return traj_lis, best_trajectory

def main(args=None):
    rclpy.init(args=args)
    node=DWAPlanner()
    # executor = MultiThreadedExecutor(num_threads=5)
    # executor.add_node(node)
    # executor.spin()
    rclpy.spin(node)
    rclpy.shutdown()

# checks if the script is run directly, if not calls the main function
if __name__ == '__main__':
    main()
