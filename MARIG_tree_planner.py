'''
Adaptive implementation of RRT_star
Replans after every budget expenditure of ~0.4

'''
from cgi import print_directory
import csv
import os
from traceback import print_tb
import numpy as np
from copy import deepcopy
import time
from RRTGenerator import RRTGraph

from RIG_original import *
from classes.Gaussian2D import *
from classes.Graph import *
from gp_ipp import *
from predictor import predictor

from parameter import *

import traceback


class RIG_planner:
    def __init__(self, i):
        self.path = f'RRT_s_IPP/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.i = i
        self.env = None
        self.rrt = None
        
        self.radius = 0.8
        self.step_sample = 0.2
        self.info = None
        self.budget_history = []
        self.obj_history = []


        self.gaussian = Gaussian2D(i)
        np.random.seed() # Remove seed set by Gaussian2D()

        self.node_coords = np.array([[]])

        self.allnode_coords = dict()
        for i in range(1, NUM_AGENTS+ 1):
            self.allnode_coords[f"{i}"] = []

        # Create copy of ground truth for future use
        self.underlying_distribution = deepcopy(self.gaussian)

        # Initialize GP
        self.gp = GaussianProcessForIPP()
        self.ground_truth = self.get_ground_truth()
        self.high_info_area = self.gp.get_high_info_area()

    def global_planner(self):
        self.Tree = Graph()
        self.cov_trace = float('infinity')

        self.generator = RRTGraph(self.step_sample)

    def agent_planner(self, agent_ID, agent_route, agent_pos):

        global covariance_trace

        best_node = None
        length = 0

        ti = time.time()

        self.budget = BUDGET_SIZE # 8
        connected_edges = None

        # Start node & init RRT
        #start = np.array([agent_ID * 0.2, agent_ID * 0.2]) 
        start = np.array([0, 0]) 

        self.updated_node_coords = np.array([[]])

        # Find tree, initially all infos = 0.0
        counts = 0
        self.node_coords = np.array([[]])
        
        # Execution stuff
        self.dist_residual = 0
        self.added_node_coord = start # np.array([[0.0, 0.0]])

        self.current_node_index = 0 # start_node
        start_node_index = int(0)

        pre_length = len(self.updated_node_coords.reshape(-1,2))

        while self.budget > 0:

            if counts != 0:
                pre_length -= 1

            # Generate tree
            self.rrt = RRT(50, 1.0, 1.0, self.radius, self.step_sample, self.gp, self.underlying_distribution)

            nodes = self.rrt.RRT_planner(start, iterations=175, info = self.gp)
                
            for each_node in nodes:
                append_it = np.array([[each_node.x, each_node.y]])
                self.node_coords = np.array(np.append(self.node_coords, append_it))
                
            self.node_coords = self.node_coords.reshape(-1,2)
       

            self.updated_node_coords = np.array(np.append(self.updated_node_coords, self.node_coords))
            
            # Generating graph & predictor
            Predictor = predictor(self.node_coords, self.step_sample, self.gaussian, self.gp)
            graph = self.generator.create_graph(self.node_coords)

            cost = 0.0
            path = []
            while cost < 0.3:
                
                print(f"start_node_index type is {type(start_node_index)}")
                #connected_edges = graph.edges[str(start_node_index)]
                
                #connected_edges = graph.edges[str(start_node_index)]
                
                try:
                    connected_edges = graph.edges[str(start_node_index)]
                except Exception as e:
                    traceback.print_exc()
                
                ''' 
                try:
                    connected_edges = graph.edges[str(start_node_index)]
                except:
                    connected_edges = graph.edges[start_node_index]
                '''
                #print("get connect edges with type int")
                
                try:
                    print(start_node_index)
                except Exception as e:
                    traceback.print_exc()

                #print(connected_edges)
                #print(start_node_index)
                #print(type(start_node_index))
                #print(type(connected_edges))
                
                for node, edge in connected_edges.items():
                    if edge.length != 0.0 and node not in path:
                        pred_copy = deepcopy(Predictor)
                        
                        cov_new, dist = pred_copy.prediction(node, int(start_node_index), self.high_info_area)
                        '''
                        if type(start_node_index) == 'string':
                            cov_new, dist = pred_copy.prediction(node, int(start_node_index), self.high_info_area)
                        else: # type(start_node_index)= 'int'
                            cov_new, dist = pred_copy.prediction(node, start_node_index, self.high_info_area)
                        '''
                        
                        if cov_new < self.cov_trace:
                            self.cov_trace = cov_new
                            best_node = node
                            length = dist

                start_node_index = best_node
                path.append(best_node)
                
                cost += length

            start_index = 0

            print(f"path of agent {agent_ID} is {path}")

            for each_step in path:
                if self.budget < 0.0:
                    break
                self.added_node_coord = np.append(self.added_node_coord, self.node_coords[int(each_step)])
                covariance_trace = self.execute_path(each_step, start_index)
                
                start_index = int(each_step)
                self.budget_history.append(8-self.budget)
                self.obj_history.append(covariance_trace)

                agent_route[f"{agent_ID}"].append(each_step)

                print(f"agent route is {agent_route}")

                agent_pos[f"{agent_ID}"].append(self.node_coords[int(each_step)])
                print(f"agent_pos is {agent_pos}")
                

                print(f"current cov_trace of agent {agent_ID} is {covariance_trace}")
            
            '''
            counts += 1
            start = self.node_coords[int(path[-1])]

            self.node_coords = np.array([[]])

            end = np.array([1,1])
            dist2end = np.linalg.norm(start - end)
            if self.budget < dist2end + 0.5:
                self.added_node_coord = np.append(self.added_node_coord, end)
                self.node_coords = np.array([start, end])
                covariance_trace = self.execute_path(1, 0)
                self.budget_history.append(8-self.budget)
                self.obj_history.append(covariance_trace)
                break
            '''

        self.added_node_coord = self.added_node_coord.reshape(-1, 2)
        for i in range(len(self.added_node_coord)):
            if i+1!= len(self.added_node_coord):
                self.Tree.add_node(str(i))
                self.Tree.add_edge(str(i), str(i+1), np.linalg.norm(self.added_node_coord[i] - self.added_node_coord[i+1]))
                self.Tree.add_edge(str(i+1), str(i), np.linalg.norm(self.added_node_coord[i] - self.added_node_coord[i+1]))

        tf = time.time()
        # generator.visualize_graph(self.Tree, self.path, self.i, self.gp, self.ground_truth, 'Tree', self.added_node_coord, self.budget, covariance_trace, tf-ti)

#        print('Time - ' + str(tf - ti))
#        print('Done!')
        self.plot(agent_ID=agent_ID,  agent_pos=agent_pos)
        
        return covariance_trace, tf-ti, agent_route, agent_pos

    def execute_path(self, each_step, prev_step, index_input=True):
#        print('steping from ' + str(prev_step) + ' to ' + str(each_step))
        current_node_index = int(prev_step)
        dist = np.linalg.norm(self.node_coords[current_node_index] - self.node_coords[int(each_step)])#next_node_index])
        remain_length = dist
        next_length = self.step_sample - self.dist_residual
#        reward = 0

        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[int(each_step)] - self.node_coords[
                    current_node_index]) * next_length / dist + self.node_coords[current_node_index]
            else:
                self.sample = (self.node_coords[int(each_step)] - self.node_coords[
                    current_node_index]) * next_length / dist + self.sample
            observed_value = self.underlying_distribution.distribution_function(
                self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
            self.gp.add_observed_point(self.sample, observed_value)
            remain_length -= next_length
            next_length = self.step_sample
            no_sample = False

        self.gp.update_gp()
        self.high_info_area = self.gp.get_high_info_area()# if ADAPTIVE_AREA else None
        cov_trace = self.gp.evaluate_cov_trace(self.high_info_area)

        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        self.budget -= dist
        print(f"current budget is {self.budget}")
#        self.current_node_index = int(each_step)
        return cov_trace #done, self.node_info, self.node_std, self.budget #reward, done, self.node_info, self.node_std, self.budget


    def get_ground_truth(self):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth
    
    def plot(self, agent_ID, agent_pos):
        # def plot(self, route, n, path, ground_truth, remain_budget, agent_ID, CMAES_route=False):
            # Plotting shorest path
        plt.switch_backend('agg')

        self.gp.plot(self.ground_truth)

        # plt.subplot(1,3,1)
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        # plt.scatter(self.node_coords[f"{agent_ID}"][1][0], self.node_coords[f"{agent_ID}"][1][1], c='r', marker='*',
                    #s=15 ** 2)

        for ID in range(NUM_AGENTS):

            pointsToDisplay = [path for path in agent_pos[f"{ID}"]]

            x = [item[0] for item in pointsToDisplay]
            y = [item[1] for item in pointsToDisplay]
            for i in range(len(x) - 1):
                plt.plot(x[i:i + 2], y[i:i + 2], c=colorlist[agent_ID - 1], linewidth=4, zorder=5,
                         alpha=0.25 + 0.6 * i / len(x))

            # plt.scatter(self.node_coords[agent_ID][0], self.node_coords[agent_ID][1], c='r', s=35)

        plt.subplot(2, 2, 4)
        plt.title('Interesting area')
        x = self.high_info_area[:, 0]
        y = self.high_info_area[:, 1]
        plt.hist2d(x, y, bins=30, vmin=0, vmax=1)


        plt.suptitle('Cov trace: {:.4g}'.format(self.cov_trace))
        # plt.tight_layout()
        #plt.savefig('{}/{}.png'.format(path, n), dpi=150)
        #plt.savefig('./gifs/{agent_ID}.png')
        plt.savefig('./gifs/{}.png'.format(agent_ID), dpi=150)




if __name__ == '__main__':
    NUM_REPEAT = 1 ## 10
    NUM_TEST = 1 ## 10
    SAVE_TRAJECTORY_HISTORY = True
    SAVE_CSV_RESULT = True
    NUM_AGENTS = 3
    sub_results = []
    results = []
    all_results = []
    t0 = time.time()

    agent_route = dict()
    for agent_ID in range(NUM_AGENTS):
        agent_route[f"{agent_ID}"] = []

    agent_pos = dict()
    for agent_ID in range(NUM_AGENTS):
        agent_pos[f"{agent_ID}"] = []

    for j in range(NUM_REPEAT):
        for i in range(NUM_TEST):
            print('Loop:', i, j)
            rig = RIG_planner(i)
            rig.global_planner()
            #print('test successfully')
            for agent_ID in range(NUM_AGENTS):
                cov_trace, time_used, agent_route, agent_pos = rig.agent_planner(agent_ID, agent_route, agent_pos)
                print(f"total usedtime is {time_used}, final cov_trace is {cov_trace}")
            sub_results.append(cov_trace)
            budget_history = np.array(rig.budget_history)
            obj_history = np.array(rig.obj_history)
            
