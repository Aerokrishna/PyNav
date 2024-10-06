#!/usr/bin/env python3
#publisher with timer
import rclpy
import rclpy.clock
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import rclpy.time
import cv2
import numpy as np
import pkg_resources
from ament_index_python.packages import get_package_share_directory
import os
import matplotlib.pyplot as plt
from tf_transformations import quaternion_from_euler
# from nav_msgs.msg import GL
class OccupancygridPub(Node):
    def __init__(self):
        super().__init__("min_publisher")
        self.ogrid_pub_=self.create_publisher(OccupancyGrid, "occupancy_grid",10)
        self.timer_=self.create_timer(0.5,self.ogrid_publisher_callback)
        self.occupancy_grid = []
        print("ogrid started")

        self.threshold = 100

    def create_ogrid(self):
        #image = pkg_resources.resource_filename('nav_stack', 'maps/real_map4.png')
        image_path = os.path.join(get_package_share_directory('nav_stack'),'maps/real_map4.png') 
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        np.set_printoptions(threshold=np.inf)

        # print(image)
        height, width, _ = image.shape

        self.occupancy_grid = np.zeros((height, width), dtype=np.uint8)
        
        # Reshape and aggregate the image
        reshaped_img = image.reshape(height , 1, width , 1, 3)
        
        # Calculate mean color in each cell block
        mean_colors = reshaped_img.mean(axis=(1, 3))  # Averaging over the cell_size dimension
        
        # Determine if the block is "white" or "black" based on the threshold
        white_mask = np.all(mean_colors < self.threshold, axis=-1)  # Check if all color channels are below the threshold
        self.occupancy_grid[white_mask] = 100  # Occupied by white color  
        # print(self.occupancy_grid.flatten().tolist())   
        self.occupancy_grid = self.occupancy_grid.transpose()

    def costmap(self):

        actions = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
        shape = self.occupancy_grid.shape

        occupied_cells = np.argwhere(self.occupancy_grid == 100)
       
        # print(occupied_cells)
        for m, n in occupied_cells:
            # print(m,n)
            # Calculate the neighboring cells
            neighbors = actions + [m, n] # adding two matrices, basically getting a list of all the neighbouring 

            # conditionally check the neighbours
            valid_neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < shape[0]) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < shape[1])]

            self.occupancy_grid[valid_neighbors[:, 0], valid_neighbors[:, 1]] = np.where(
                self.occupancy_grid[valid_neighbors[:, 0], valid_neighbors[:, 1]] == 100, 
                100, 
                50
            ) # if neighbouring cell occupied value is kept 1 if it is not then value changed to 0.5
       
        occupied_cells = np.argwhere(self.occupancy_grid == 50)
    
        for m, n in occupied_cells:

            neighbors = actions + [m, n] # adding two matrices, basically getting a list of all the neighbouring 

            # conditionally check the neighbours
            valid_neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < shape[0]) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < shape[1])]

            self.occupancy_grid[valid_neighbors[:, 0], valid_neighbors[:, 1]] = np.where(
                self.occupancy_grid[valid_neighbors[:, 0], valid_neighbors[:, 1]] == 0, 
                30, 
                self.occupancy_grid[valid_neighbors[:, 0], valid_neighbors[:, 1]]
            ) # if condition is true replaces it with 1 else 0.3

    def ogrid_publisher_callback(self):
        msg=OccupancyGrid()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map_frame"
        msg.info.resolution = 0.05
        msg.info.height = self.occupancy_grid.shape[0]
        msg.info.width = self.occupancy_grid.shape[1]

        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        q = quaternion_from_euler(0.0,0.0,0.0)
        msg.info.origin.orientation.x = q[0]
        msg.info.origin.orientation.y = q[1]
        msg.info.origin.orientation.z = q[2]
        msg.info.origin.orientation.w = q[3]
        ogrid_list = self.occupancy_grid.ravel().tolist()
        msg.data = ogrid_list

        print(self.occupancy_grid)
        plt.imshow(self.occupancy_grid, cmap='gray_r', interpolation='nearest')
        # plt.show()  

        # msg.data = [100,100,0,0,0,0,0,0,0]
        print(len(msg.data))

        self.ogrid_pub_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node=OccupancygridPub()
    node.create_ogrid()
    node.costmap()
    rclpy.spin(node)
    rclpy.shutdown()

# checks if the script is run directly, if not calls the main function
if __name__ == '__main__':
    main()

