#!/usr/bin/env python3
#publisher with timer
import rclpy
import rclpy.clock
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf2_geometry_msgs import PointStamped
from tf2_ros import TransformListener, Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from std_msgs.msg import Header
import rclpy.time
import cv2
import numpy as np
import pkg_resources
from ament_index_python.packages import get_package_share_directory
import os
import matplotlib.pyplot as plt
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker
from rclpy.duration import Duration
from rclpy.time import Time


# DWA PLANNER FOR A DIFFERENTIAL-DRIVE ROBOT

class Params():
    def __init__(self) -> None:
        self.min_vel = 0.0
        self.max_vel = 0.22
        self.min_w = - 2.84 # rad/s
        self.max_w =  2.84 # rad/s
        self.max_accel = 0.3
        self.max_w_accel = 1.00
        self.time_step = 0.5
        self.time_period = 5.0
        self.v_resolution = 0.05
        self.w_resolution = 0.05
        self.speed_cost_gain = 10.0
        self.obstacle_cost_gain = 0.5
        self.goal_cost_gain = 0.8
        
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


class DWAPlanner(Node):
    def __init__(self):
        super().__init__("dwa_planner")
        # subscribe to odom, scan, global_path
        self.ogrid_pub_=self.create_subscription(LaserScan, "/scan",self.scan_callback, 10)
        self.marker_publisher_ = self.create_subscription(Odometry,'/odom',self.odom_callback, 10)
        self.path_publisher = self.create_subscription(Path, "global_path",self.path_callback, 10)

        self.cmd_vel_pub_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.v = 0.0
        self.w = 0.0

        self.params = Params()
        self.dwa_functions = DWAFunctions()

        # transformation 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.reached_goal = 0

        self.trajectory_range = [0,360]

        self.ranges = []
        self.obstacles = []
        self.waypoint = (0.0,0.0)
    
    def scan_callback(self, scan_msg : LaserScan):
        self.range_data = scan_msg.ranges
        self.angle_increament = scan_msg.angle_increment
        self.angle_min = scan_msg.angle_min

        max_range = 5

        state = [self.robot_x,self.robot_y,self.robot_yaw,0.0,0.0]

        dw = self.dwa_functions.get_dynamic_window(state, self.params)
        trajs, best_traj = self.local_plan(dw, state, self.params)

        self.obstacles = []

        for i in range(self.trajectory_range[1]):
            range_val = self.range_data[i]
            theta = self.angle_min + (i * self.angle_increament)

            if range_val < max_range:

                x,y = self.scan_to_pose(range_val, theta)
                self.obstacles.append((x,y))
        
        for j in range(self.trajectory_range[0], len(self.range_data)):
            range_val = self.range_data[j]
            theta = self.angle_min + (j * self.angle_increament)

            if range_val < max_range:

                x,y = self.scan_to_pose(range_val, theta)
                self.obstacles.append((x,y))
        # print("best trajecory",best_traj)
        print("CONTROL COMMAND v = ", best_traj[-1][3], "w = ", best_traj[-1][4])

        self.v = best_traj[-1][3]
        self.w = best_traj[-1][4]

        if self.reached_goal == 1:
            self.v = 0.0
            self.w = 0.0

        vel_msg = Twist()

        vel_msg.linear.x = self.v
        vel_msg.angular.z = self.w

        self.cmd_vel_pub_.publish(vel_msg)

        # print(len(self.obstacles))
        # self.get_logger().info(f"the best trajectory is {best_traj}")
        # for i in range(len(trajs)):

        #     plt.plot(trajs[i][:, 0], trajs[i][:, 1], "-r")

        # plt.plot(best_traj[:, 0], best_traj[:, 1], "-g")   
        # plt.show()

        # self.get_logger().info(f"trajectories {trajs}")
    


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
        for i in range(len(path_msg.poses)):
            wpx,wpy = path_msg.poses[i].pose.position.x,path_msg.poses[i].pose.position.y

            # if abs(self.robot_x - wpx) < 0.2 and abs(self.robot_y - wpy) < 0.2:
            #     path = []
            #     continue

            path.append((wpx,wpy))

        global_path = np.array(path)
        self.waypoint = global_path[-1]

        for i in range(len(global_path)-5):

            if abs(self.robot_x - global_path[-1][0]) < 0.3 and abs(self.robot_y - global_path[-1][1]) < 0.3:
                self.reached_goal = 1
                
            if abs(self.robot_x - global_path[i][0]) < 0.1 and abs(self.robot_y - global_path[i][1]) < 0.1:
                index = int((len(global_path) - i)/2)
                # self.waypoint = global_path[i+index]
                self.waypoint = global_path[i+5]

                break


        print(self.waypoint)

    def obstacle_distance_cost(self, trajectory, obstacles):
        min_dis = np.inf
        min_traj = ()
        # for traj in trajectory:
        
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

    def goal_cost(self, goal, trajectory):

        x = goal[0] - self.robot_x
        y = goal[1] - self.robot_y

        theta_goal = np.arctan2(y, x)
        goal_cost = abs(theta_goal - trajectory[-1][2])

        # print(goal_cost)

        return goal_cost

    def local_plan(self, dw, state, params):
        current_state = state[:]
        min_cost = np.inf
        best_control = [0.0, 0.0]
        best_trajectory = np.array([current_state])

        # evaluate all trajectory with sampled input in dynamic window
        traj_lis = []
        velocity_ = np.arange(dw[0], dw[1], params.v_resolution)
        angular_velocity_ = np.arange(dw[2], dw[3], params.w_resolution)
        cnt = 0
        cnt2 = 0
        for v in velocity_:
            for w in angular_velocity_:
                trajectory = self.dwa_functions.predict_trajectory(current_state, v, w, params)

                if v == velocity_[1] and (w == dw[2] or w == angular_velocity_[-1]):
                    
                        theta = self.compute_vector((self.robot_x, self.robot_y),(trajectory[-1][0], trajectory[-1][1]))
                        if theta < 0:
                            theta += 2*np.pi
                        scan_or = np.rad2deg(theta - self.robot_yaw) 
                        if scan_or < 0:
                            scan_or+=360
                        scan_index = round(scan_or)
                        self.trajectory_range[cnt] = scan_index
                        cnt+=1

                traj_lis.append(trajectory)
                cost,pts = self.obstacle_distance_cost(trajectory, self.obstacles)
                obs_cost = params.obstacle_cost_gain * cost
                speed_cost = params.speed_cost_gain * (params.max_vel - trajectory[-1, 3])
                goal_cost = params.goal_cost_gain * (self.goal_cost(self.waypoint,trajectory))

                final_cost = goal_cost + speed_cost + obs_cost
                # print(obs_cost, final_cost, min_cost)
                if final_cost < min_cost:
                    # print('goal cost',goal_cost)
                    # print('speed cost',speed_cost)
                    # print('obstacle cost',obs_cost)
                    # print('final_cost',final_cost)


                    min_cost = final_cost
                    best_trajectory = trajectory

        # print(self.trajectory_range)
        return traj_lis, best_trajectory

def main(args=None):
    rclpy.init(args=args)
    node=DWAPlanner()
    # node.goal_cost(goal=[2.0,2.0,0.0], robot_pose=[0.0,0.0,-0.78])
    # state = [0.0,0.0,0.0,0.0,0.0]
    # dw = node.dwa_functions.get_dynamic_window(state, node.params)
    # trajs = node.local_plan(dw, state, node.params)

    # for i in range(len(trajs)):

    #     plt.plot(trajs[i][:, 0], trajs[i][:, 1], "-r")

    # # plt.plot(best_traj[:, 0], best_traj[:, 1], "-g")   
    # plt.show()
    rclpy.spin(node)
    rclpy.shutdown()

# checks if the script is run directly, if not calls the main function
if __name__ == '__main__':
    main()
# params = Params()
      
# # trajectory = predict_trajectory([0.0,0.0,0.0,1.0,0.0], 2.0, -0.1, params)

# state = [0.0,0.0,0.0,0.3,0.0]
# dw = get_dynamic_window(state, params)

# trajs, best_traj = get_best_trajectory(dw, state, params)
# # print(dw)
# print(len(trajs))
# for i in range(len(trajs)):

#     plt.plot(trajs[i][:, 0], trajs[i][:, 1], "-r")

# plt.plot(best_traj[:, 0], best_traj[:, 1], "-g")        
# plt.show()
# # print(get_best_trajectory())