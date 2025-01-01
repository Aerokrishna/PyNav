import numpy as np
import matplotlib.pyplot as plt

def is_valid(cell_x,cell_y,occupancyGrid, closed_nodes):
    next_node = cell_x * occupancyGrid.shape[1] + cell_y

    if next_node in closed_nodes:
        return False
    
    elif cell_x < 0 or cell_x >= occupancyGrid.shape[0] or cell_y < 0 or cell_y >= occupancyGrid.shape[1]:
        return False
    
    elif occupancyGrid[cell_x][cell_y] > 29:
        return False
    
    else:
        return True

def get_distance(x1,y1,x2,y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_cost(node,occupancyGrid,goal,start):
    (i,j) = np.unravel_index(node, occupancyGrid.shape)
    (gi,gj) = np.unravel_index(goal, occupancyGrid.shape)
    (si,sj) = np.unravel_index(start, occupancyGrid.shape)
    
    h_cost = get_distance(i,j,gi,gj)
    g_cost = get_distance(i,j,si,sj)

    f_cost = h_cost + g_cost

    return f_cost,g_cost

def retrace_path(goal_node, start_node, parent_nodes):
    current_node = goal_node
    ideal_nodes = []

    while current_node != start_node:
        parent_node = parent_nodes[current_node]
        current_node = parent_node
        ideal_nodes.insert(0,current_node)
    
    
    return ideal_nodes    
    
def a_star(start_node, goal_node, occupancyGrid):
    ideal_nodes_coords = []
    closed_nodes = {}
    parent_nodes = {}
    open_nodes = {start_node : (np.inf,0)}


    # if start_node == goal_node:
    #     print("already at the DESTINATION")
    #     return []

    
    # if open_nodes is None:
    #     print("Path does not Exist!")
    #     return []


    actions = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    

    while len(open_nodes) > 0:

        current_node = min(open_nodes, key=lambda k: open_nodes[k][0])
        closed_nodes[current_node] = open_nodes.pop(current_node)
   
        for i in range(len(actions)):
            # print(occupancyGrid.shape)
            current_node_x, current_node_y = np.unravel_index(current_node, occupancyGrid.shape)
        
            current_node_x += actions[i][0]
            current_node_y += actions[i][1]
            
            if is_valid(current_node_x,current_node_y,occupancyGrid, closed_nodes):
                next_node = current_node_x * occupancyGrid.shape[1] + current_node_y

                # print(current_node_x,current_node_y)
                f_cost,g_cost = get_cost(next_node,occupancyGrid,goal_node,start_node)
                
                if next_node not in open_nodes or open_nodes[next_node][0] > f_cost:
                    
                    # re compute the fcost and gcost
                    open_nodes[next_node] = (f_cost,g_cost)
                    parent_nodes[next_node] = current_node

            else:
                continue
            
        if current_node == goal_node:
            ideal_nodes = retrace_path(goal_node,start_node,parent_nodes)
        
            for k in ideal_nodes:
                (x,y) = np.unravel_index(k, occupancyGrid.shape)
                ideal_nodes_coords.append([x,y])
                
            
    return ideal_nodes_coords

