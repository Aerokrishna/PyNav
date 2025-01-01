#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, GoalStatus
from pynav_interfaces.action import PynavGoal
from lingoNav_client import InferHumanCommand


'''
WHAT CAN A CLIENT DO?

- send goal request
- receive response of whether goal is accepted or rejected
- receive feedback while the server is executing the task
- receive the result once the server is done executing
- can cancel a request mid way of execution
'''

class NavigationRequest(Node):
    def __init__(self):
        super().__init__("count_until_server")

        # create action client
        self.count_until_client_ = ActionClient(self, PynavGoal, "pynav/navigation_goal")
        self.goal_reached = False


    '''
    Runs once whenever we want to send goal
    '''
    def send_goal(self, waypoints):

        self.get_logger().info("waiting for server....")
        self.count_until_client_.wait_for_server() # you can provide a timer to wait for the server inside

        self.get_logger().info("server found!")

        # Define your goal as your custom action
        goal = PynavGoal.Goal()

        gx = []
        gy = []
        gtheta = []

        for i in range(len(waypoints)):
            gx.append(waypoints[i][0])
            gy.append(waypoints[i][1])
            gtheta.append(0.0)

        goal.goal_pose_x = gx
        goal.goal_pose_y = gy
        goal.goal_pose_theta = gtheta

        self.get_logger().info("GOAL SENDING !")

        # send your goal request to the action server
        # add a callback for whenever a feedback is received
        # add a callback to check whether the goal is accepted by the server or not 
        self.count_until_client_.send_goal_async(goal, feedback_callback=self.goal_feedback_callback).add_done_callback(self.goal_response_callback) 
        
    '''
    Runs async whenever goal request is ACCEPTED/ REJECTED
    ''' 
    def goal_response_callback(self, future):
        # callback to see if goal was accpeted
        self.goal_handle_: ClientGoalHandle = future.result()

        if self.goal_handle_.accepted:
            self.get_logger().info("GOAL ACCEPTED!")

            # add a callback which runs when a result is received
            self.goal_handle_.get_result_async().add_done_callback(self.goal_result_callback) # call the future callback
        else:
            self.get_logger().warn("GOAL REJECTED")

    '''
    Runs async whenever FEEDBACK is received
    '''
    def goal_feedback_callback(self, feedback_msg : PynavGoal):

        current_goal = feedback_msg.feedback.current_goal_pose
        robot_pose = feedback_msg.feedback.robot_pose
        control = feedback_msg.feedback.control
   
        if not self.goal_reached:
            self.get_logger().info(f'YOUR ROBOT IS NAVIGATING TO THIS GOAL : {current_goal}')
        # self.get_logger().info(f'{current_goal}')
        # if number == 4:
        #     self.cancel_goal()
    
    '''
    Runs async whenever RESULT is received
    '''
    def goal_result_callback(self,future):
        status = future.result().status
        result = future.result().result # is the reached number interface made in actions

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.goal_reached = True
            self.get_logger().info("NAVIGATION SUCCESSFULL !")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("ABORTED")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error("CANCELLED")

    def cancel_goal(self):
        self.get_logger().info("Sending cancel request")
        self.goal_handle_.cancel_goal_async()
         
def main(args=None):
    rclpy.init(args=args)

    nlp = InferHumanCommand()
    # Convert the structured output into an array of goal points
    structured_output = nlp.infer_human_command()
    waypoints = nlp.parse_waypoints(structured_output)
    print("Goal Points:", waypoints)

    node = NavigationRequest()
    # waypoints = [(2.0, 2.0, 0.0), (2.7, 3.6, 0.0)]
    node.send_goal(waypoints) # call the send goal function
    rclpy.spin(node) # then spin the node to wait for result
    rclpy.shutdown()

if __name__=="__main__":
    main()