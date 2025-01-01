#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from rclpy.action import ActionServer
from pynav_interfaces.action import PynavGoal
from pynav_interfaces.msg import GoalPose
from rclpy.action.server import ServerGoalHandle, GoalResponse, CancelResponse
import threading
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler,euler_from_quaternion

'''
configure all the nodes : all nodes are inactive : costmap, planner, controller
activate occupancy grid
delay
activate planner : according to the planning alg requested
delay
activate controller : according to the controller alg requested
'''
class NavigationManager(Node):
    def __init__(self):
        super().__init__("navigation_manager")
        # self.declare_parameter("managed_node_name", rclpy.Parameter.Type.STRING)
        # node_name = self.get_parameter("managed_node_name").value

        costmap_node = "costmap"
        a_star_planner_node = "a_star_planner"
        dwa_controller_node = "dwa_planner"

        costmap_service = "/" + costmap_node + "/change_state"
        planner_service = "/" + a_star_planner_node + "/change_state"
        controller_service = "/" + dwa_controller_node + "/change_state"

        self.costmap_client = self.create_client(ChangeState, costmap_service)
        self.planner_client = self.create_client(ChangeState, planner_service)
        self.controller_client = self.create_client(ChangeState, controller_service)

        self.get_logger().info("NAVIGATION MANAGER ")

        '''
        action server
        '''
        self.goal_handle_ : ServerGoalHandle = None
        self.goal_lock_ = threading.Lock()
        self.goal_queue_ = []

        self.count_until_server_ = ActionServer(
            self, 
            PynavGoal,  # interface name
            "pynav/navigation_goal", # this is where client will send request
            goal_callback=self.goal_callback, # if this callback returns "accpted" then only execute_callback is called
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback, # runs when cancel request is received - can cancel immediately or delayed
            execute_callback=self.pynav_navigator, # main function which executes the goal
            callback_group=ReentrantCallbackGroup()) #to run multiple threads
         
        self.odom_sub=self.create_subscription(Odometry, "/odom",self.odom_callback,10)
        self.cmd_vel_sub=self.create_subscription(Twist, "/cmd_vel",self.cmd_vel_callback,10)

        self.goal_pose_pub_ = self.create_publisher(GoalPose, "pynav/goal_pose", 10)

        self.robot_pose = [0.0,0.0,0.0]
        self.control = [0.0,0.0]

    '''
    A callback to REJECT/ ACCEPT request
    '''
    def goal_callback(self, goal_request: PynavGoal.Goal):
        
        # check if it is possible to reach the goal
        # check if a goal is already executing
        # queue or reject goal?

        # reject the goal if it is not valid

        
        # reject the goal if a goal is already executing
        with self.goal_lock_:
            if self.goal_handle_ is not None and self.goal_handle_.is_active:
                self.get_logger().error("GOAL ACTIVE...rejecting new goal")
                return GoalResponse.REJECT

        # cancel the active goal and accept new one
        # with self.goal_lock_:
        #     if self.goal_handle_ is not None and self.goal_handle_.is_active:
        #         self.get_logger().warn("New goal received aborting current goal")
        #         self.goal_handle_.abort()
        #         return GoalResponse.ACCEPT
            
        self.get_logger().info("GOAL ACCEPTED!")
        return GoalResponse.ACCEPT
    
    def handle_accepted_callback(self, goal_handle: ServerGoalHandle):

        goal_handle.execute()
    '''
    when a cancel request is received from the client we cancel the execution
    '''
    def cancel_callback(self, goal_handle: ServerGoalHandle): # goal handle is going to be cancelled as client will cancel a particular goal
        self.get_logger().info("cancel request received...")
        # we accept the cancel request
        return CancelResponse.ACCEPT

    '''
    main logic of execution of the goal
    '''
    def pynav_navigator(self, goal_handle : ServerGoalHandle):
        # we can have that navigation function called here?
        # when goal has been reacched we can return the response
        
        with self.goal_lock_:
            self.goal_handle_ = goal_handle

        # catch the request coming from the client
        goal_pose_x = goal_handle.request.goal_pose_x
        goal_pose_y = goal_handle.request.goal_pose_y
        goal_pose_theta = goal_handle.request.goal_pose_theta

        # EXECUTE THE ACTION
        # self.get_logger().info(f"Executing the GOAL{goal_pose_x}{goal_pose_y}")
        counter = 0
        feedback = PynavGoal.Feedback()
        result = PynavGoal.Result()
        
        # PUBLISH THE FIRST GOAL 
        self.send_goal([goal_pose_x[0],goal_pose_y[0], goal_pose_theta[0]])
        
        self.activate_navigation()
        waypoint_index = 0

        while waypoint_index < len(goal_pose_x):
            goal_pose = [goal_pose_x[waypoint_index], goal_pose_y[waypoint_index], goal_pose_theta[waypoint_index]]
            # self.get_logger().info(f"NAVIGATING TO WAYPOINT : {goal_pose[0]} {goal_pose[1]}")
            # self.get_logger().info(f"{self.robot_pose}{self.control}")
            # to account for cancellation of goal
            if goal_handle.is_cancel_requested:
                self.get_logger().info("CANCELLING GOAL")

                self.deactivate_navigation()
                goal_handle.canceled() # cancel the goal, similarly we can set it as aborted or succeeded

                result.final_pose = self.robot_pose
                result.message = "Navigation Request CANCELLED!"
                return result # return whatever result

            # send feedback in every loop
            feedback.current_goal_pose = goal_pose
            feedback.control = self.control
            feedback.robot_pose = self.robot_pose

            goal_handle.publish_feedback(feedback)
            
            if self.goal_reached(goal_pose, self.robot_pose):
                waypoint_index += 1

                # if last waypoint reached
                if waypoint_index == len(goal_pose_x):
                    break
                
                # if not the last send the next goal
                # time.sleep(3.0)
                goal_pose = [goal_pose_x[waypoint_index], goal_pose_y[waypoint_index], goal_pose_theta[waypoint_index]]
                self.send_goal(goal_pose)

        # once done, set goal final state
        goal_handle.succeed()
        self.deactivate_navigation()

        # and send the result
        result.final_pose = self.robot_pose
        result.message = "Navigation Successful !"

        return result # execute callback has to return the result
    

    def change_state(self, transition: Transition, client):
        client.wait_for_service()
        request = ChangeState.Request()
        request.transition = transition
        future = client.call_async(request)
        # rclpy.spin_until_future_complete(self, future)
    
    def initialize_navigation(self):
        # Unconfigured to Inactive
        transition = Transition()

        time.sleep(2.0)
        '''
        CONFIGURE ALL NODES
        '''
        transition.id = Transition.TRANSITION_CONFIGURE
        transition.label = "configure"
        
        self.change_state(transition, self.costmap_client)
        self.change_state(transition, self.planner_client)
        self.change_state(transition, self.controller_client)

        # time.sleep(5.0)
        self.get_logger().info("MANAGER : Costmap, Planner, Controller are INACTIVE")

    def activate_navigation(self):
        # Unconfigured to Inactive
        transition = Transition()

        time.sleep(1.0)
        transition.id = Transition.TRANSITION_ACTIVATE
        transition.label = "activate"

        self.change_state(transition, self.costmap_client)
        # self.get_logger().info("COSTMAP ACTIVE!")

        time.sleep(1.0)

        self.change_state(transition, self.planner_client)
        # self.get_logger().info("PLANNER ACTIVE!")

        time.sleep(2.0)

        self.change_state(transition, self.controller_client)
        # self.get_logger().info("CONTROLLER ACTIVE !")

        self.get_logger().info("MANAGER : Costmap, Planner, Controller are ACTIVE")


    def deactivate_navigation(self):
        # Unconfigured to Inactive
        transition = Transition()

        transition.id = Transition.TRANSITION_DEACTIVATE
        transition.label = "deactivate"

        self.change_state(transition, self.costmap_client)
        self.change_state(transition, self.planner_client)
        self.change_state(transition, self.controller_client)

        self.get_logger().info("MANAGER : Costmap, Planner, Controller are DEACTIVED")

    
    def shutdown_navigation(self):
        # Unconfigured to Inactive
        transition = Transition()

        transition.id = Transition.TRANSITION_INACTIVE_SHUTDOWN
        transition.label = "shutdown"

        self.change_state(transition, self.costmap_client)
        self.change_state(transition, self.planner_client)
        self.change_state(transition, self.controller_client)

    def odom_callback(self, odom_msg : Odometry):
        self.robot_pose[0] = odom_msg.pose.pose.position.x
        self.robot_pose[1] = odom_msg.pose.pose.position.y

        x = odom_msg.pose.pose.orientation.x
        y = odom_msg.pose.pose.orientation.y
        z = odom_msg.pose.pose.orientation.z
        w = odom_msg.pose.pose.orientation.w

        self.robot_pose[2] = euler_from_quaternion([x,y,z,w])[2]
        # print("ODOM CALLBACK")
    
    def cmd_vel_callback(self, cmd_vel_msg : Twist):
        # print("CMD VEL CALLBACK")
        self.control = [cmd_vel_msg.linear.x , cmd_vel_msg.angular.z]

    def send_goal(self, goal_pose):
        goal_pose_msg = GoalPose()

        goal_pose_msg.goal_pose_x = goal_pose[0]
        goal_pose_msg.goal_pose_y = goal_pose[1]
        goal_pose_msg.goal_pose_theta = goal_pose[2]

        # self.get_logger().info(f"{goal_pose}")
        self.goal_pose_pub_.publish(goal_pose_msg)

    def goal_reached(self, goal_pose, robot_pose):

        if abs(goal_pose[0] - robot_pose[0]) < 0.15 and abs(goal_pose[1] - robot_pose[1]) < 0.15:
            return True

        return False

def main(args=None):
    rclpy.init(args=args)

    node = NavigationManager()
    node.initialize_navigation()
    executor = MultiThreadedExecutor(5)
    executor.add_node(node)
    executor.spin()

    if KeyboardInterrupt():
        node.shutdown_navigation()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
