'''
Adaptive implementation of RRT_star
Replans after every budget expenditure of ~0.4

'''

from cgi import print_directory
import csv
import os
from pickle import NONE
from random import sample

from traceback import print_tb
from matplotlib.cbook import print_cycles
from matplotlib.pyplot import plot
import numpy as np
from copy import deepcopy
import time

from wandb import agent
from RRTGenerator import RRTGraph

from RIG_original import *
from classes.Gaussian2D import *
from classes.Graph import *
from gp_ipp import *
from predictor import predictor

from RIG_parameter import *

import traceback


class RIG_planner:
    def __init__(self, i):
        self.path = f'RRT_s_IPP/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.i = i
        self.env = None
        self.rrt = None
        
        self.radius = RIG_RADIUS
        self.step_sample = 0.2
        self.info = None
        self.budget_history = []
        self.obj_history = []

        self.node_coords = np.array([[]])
        self.updated_node_coords = np.array([[]])

        self.gaussian = Gaussian2D(i)
        np.random.seed() # Remove seed set by Gaussian2D()

        self.allnode_coords = dict()
        for i in range(1, NUM_AGENTS+ 1):
            self.allnode_coords[f"{i}"] = []

        # Create copy of ground truth for future use
        self.underlying_distribution = deepcopy(self.gaussian)

        # Initialize GP
        self.gp = GaussianProcessForIPP()
        self.ground_truth = self.get_ground_truth()
        self.high_info_area = self.gp.get_high_info_area()
        
        self.agent_budget = dict()
        for agent_ID in range(NUM_AGENTS):
            self.agent_budget[f"{agent_ID}"] = np.array(BUDGET_SIZE)
        
        self.step_used_budget = 0

    def global_planner(self):
        self.Tree = Graph()
        self.cov_trace = float('infinity')

        self.generator = RRTGraph(self.step_sample)

    def agent_replan(self, agent_ID):
    
        self.added_node_coord = self.start
            
        best_node = None
        length = 0
        
        #self.budget = BUDGET_SIZE # 8
        connected_edges = None
        
        counts = 0
        
        if counts != 0:
            self.pre_length -= 1

        # Generate tree
        self.rrt = RRT(20, 1.0, 1.0, self.radius, self.step_sample, self.gp, self.underlying_distribution) # 50

        nodes = self.rrt.RRT_planner(self.start, iterations=175, info = self.gp)
            
        #print(f"nodes are {nodes}")
            
        for each_node in nodes:
            append_it = np.array([[each_node.x, each_node.y]])
            self.node_coords = np.array(np.append(self.node_coords, append_it))
            
        self.node_coords = self.node_coords.reshape(-1,2)
        #print(np.shape(self.node_coords))
    
        self.updated_node_coords = np.array(np.append(self.updated_node_coords, self.node_coords))
        
        # gp = deepcopy(self.gp)
        # Generating graph & predictor
        
        distance = BUDGET_SIZE - self.agent_budget[f"{agent_ID}"]
        sample_number = distance // self.step_sample
                
        Predictor = predictor(self.node_coords, self.step_sample, self.gaussian,  self.gp, self.measurement_points, agent_ID, sample_number, self.high_info_area)
        graph = self.generator.create_graph(self.node_coords)

        cost = 0.0
        path = []
        while cost < REPLAN_LENGTH:

           #print(f"graph edge is {graph.edges}")
            #print(f"start node index is {self.start_node_index}")
            try:
                connected_edges = graph.edges[str(self.start_node_index[f"{agent_ID}"])] 
            except:
                print("connected_edege errors")
                connected_edges = graph.edges[str(int(0))] 
            
            pred_copy = deepcopy(Predictor)
            #old_cov_trace = pred_copy.cov_trace
            old_cov_trace = pred_copy.cov_trace
            #m = 0
            
            for node, edge in connected_edges.items():
                if edge.length != 0.0 and node not in path:
                    # bug here
                    cov_new, dist = pred_copy.prediction(node, int(self.start_node_index[f"{agent_ID}"]), self.high_info_area)
                    
                    #if cov_new < old_cov_trace:
                    if cov_new < old_cov_trace:
                        #if (old_cov_trace - cov_new) / dist > m:
                            #m = (old_cov_trace - cov_new) / dist
                        old_cov_trace = cov_new
                            
                        #print(f"node is {node}")
                        best_node = node
                        length = dist
            
            if best_node == None:
                best_node = node
                #print("cannot get next node which can decrease cov_trace")
                            
            #print(f"best node is {best_node}")
            self.start_node_index[f"{agent_ID}"] = best_node
            path.append(best_node)
            
            cost += length

        #start_index = 0

        #print(f"path of agent is {path}")

        for each_step in path:
            if self.agent_budget[f"{agent_ID}"] < 0.0:
                break
            
            self.added_node_coord = np.append(self.added_node_coord, self.node_coords[int(each_step)])

            current_pos = self.global_agent_pos[f"{agent_ID}"][-1]
            each_step_pos = self.node_coords[int(each_step)]
            covariance_trace = self.execute_path(each_step_pos, current_pos, agent_ID, sample_number)
            
            #start_index = int(each_step)
            
            ## record global agent position
            #print(f"current pos of agent {agent_ID}")
            #print(self.node_coords[int(each_step)])
            
            self.global_agent_pos[f"{agent_ID}"].append(self.node_coords[int(each_step)])
            #print(len(self.global_agent_pos))
            
            if USE_PLOT == True:
                self.plot_num += 1
                self.plot(plot_num=self.plot_num, global_agent_pos=self.global_agent_pos,agent_ID=agent_ID)

        print(f"current cov_trace of agent {agent_ID} is {covariance_trace}")
        
        counts += 1
    
        return covariance_trace
        
    def agent_planner(self):
        
        # Start node & init RRT
        self.start = np.array([START_X, START_Y]) 
        
        self.plot_num = 0 

        self.node_coords = np.array([[]])
        self.updated_node_coords = np.array([[]])
        
        self.measurement_points = dict()
        for agent_i in range(NUM_AGENTS):
            self.measurement_points[f"{agent_i}"] = []
        
        self.global_agent_pos = dict()
        for agent_i in range(NUM_AGENTS):
            self.global_agent_pos[f"{agent_i}"] = []
            self.global_agent_pos[f"{agent_i }"].append(self.start)
            

        # Start node & init RRT
        self.start = np.array([START_X, START_Y]) 

        # Find tree, initially all infos = 0.0

        self.current_node_index = 0 # start_node
        
        # Execution stuff
        self.dist_residual = dict()
        for agent_i in range(NUM_AGENTS):
            self.dist_residual[f"{agent_i}"] = 0
        
        self.start_node_index = dict()
        for agent_i in range(NUM_AGENTS):
            self.start_node_index[f"{agent_i}"] = int(0)

        self.pre_length = len(self.updated_node_coords.reshape(-1,2))
        
        self.agent_done = dict()
        for agent_i in range(NUM_AGENTS):
            self.agent_done[f"{agent_i}"] =  False
            
        self.all_done = False
        
        ti = time.time()
         
        while self.all_done == False:
            
            #print(f"current budget is {self.agent_budget}")
            agent_ID = max(self.agent_budget, key=self.agent_budget.get)
            
            covariance_trace = self.agent_replan(agent_ID=agent_ID)
            
            ## 改
            if self.agent_budget[f"{agent_ID}"] <= 0:
                self.agent_done[f"{agent_ID}"] = True
                
            for agent_ID in range(NUM_AGENTS):
                if self.agent_done[f"{agent_ID}"] == True:
                    self.all_done =  True
                else:
                    self.all_done = False
                    break

        tf = time.time()
        # generator.visualize_graph(self.Tree, self.path, self.i, self.gp, self.ground_truth, 'Tree', self.added_node_coord, self.budget, covariance_trace, tf-ti)

#        print('Time - ' + str(tf - ti))
#        print('Done!')
        #self.plot(agent_ID=agent_ID,  agent_pos=agent_pos)
        
        self.added_node_coord = self.added_node_coord.reshape(-1, 2)
        
        for i in range(len(self.added_node_coord)):
            if i+1!= len(self.added_node_coord):
                self.Tree.add_node(str(i))
                self.Tree.add_edge(str(i), str(i+1), np.linalg.norm(self.added_node_coord[i] - self.added_node_coord[i+1]))
                self.Tree.add_edge(str(i+1), str(i), np.linalg.norm(self.added_node_coord[i] - self.added_node_coord[i+1]))
        
        return covariance_trace, tf-ti

    def execute_path(self, each_step_pos, current_pos, agent_ID, sample_number, index_input=True):
#        print('steping from ' + str(prev_step) + ' to ' + str(each_step))
        #current_node_index = int(prev_step)
        #dist = np.linalg.norm(self.node_coords[current_node_index] - self.node_coords[int(each_step)])#next_node_index])
        #gp = GaussianProcessForIPP()
        
        dist = np.linalg.norm(each_step_pos - current_pos)#next_node_index])
        remain_length = dist
        next_length = self.step_sample - self.dist_residual[f"{agent_ID}"]
#        reward = 0

        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (each_step_pos - current_pos) * next_length / dist + current_pos
            else:
                self.sample = (each_step_pos - current_pos) * next_length / dist + self.sample
            
            self.measurement_points[f"{agent_ID}"].append(self.sample)
            
            for agent_i in range(NUM_AGENTS):
                if self.measurement_points[f"{agent_i}"] != []:
                    for j,sample in enumerate(self.measurement_points[f"{agent_i}"]):
                        if j < sample_number:
                            observed_value = self.underlying_distribution.distribution_function(
                                sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        
                        # 削弱把这儿修改以下
                        else:
                            observed_value = np.array([0])
                        self.gp.add_observed_point(sample, observed_value)
                
            
            #observed_value = self.underlying_distribution.distribution_function(
                #self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        
            #gp.add_observed_point(self.sample, observed_value)
            
            
            #print(np.array(gp.observed_points).shape)
            
            remain_length -= next_length
            next_length = self.step_sample
            no_sample = False

        self.gp.update_gp()
        self.high_info_area = self.gp.get_high_info_area()# if ADAPTIVE_AREA else None
        cov_trace = self.gp.evaluate_cov_trace(self.high_info_area)
        self.cov_trace = cov_trace

        self.dist_residual[f"{agent_ID}"] = self.dist_residual[f"{agent_ID}"] + remain_length if no_sample else remain_length
        self.agent_budget[f"{agent_ID}"] -= dist
        
        self.step_used_budget = dist
        
#        self.current_node_index = int(each_step)
        return cov_trace #done, self.node_info, self.node_std, self.budget #reward, done, self.node_info, self.node_std, self.budget


    def get_ground_truth(self):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth
    
    def plot(self, plot_num, global_agent_pos, agent_ID):
        # def plot(self, route, n, path, ground_truth, remain_budget, agent_ID, CMAES_route=False):
            # Plotting shorest path
        plt.switch_backend('agg')

        self.gp.plot(self.ground_truth)

        # plt.subplot(1,3,1)
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        # plt.scatter(self.node_coords[f"{agent_ID}"][1][0], self.node_coords[f"{agent_ID}"][1][1], c='r', marker='*',
                    #s=15 ** 2)

        for ID in range(NUM_AGENTS):

            pointsToDisplay = [path for path in global_agent_pos[f"{ID}"]]

            x = [item[0] for item in pointsToDisplay]
            y = [item[1] for item in pointsToDisplay]
            for i in range(len(x) - 1):
                plt.plot(x[i:i + 2], y[i:i + 2], c=colorlist[ID - 1], linewidth=4, zorder=5,
                         alpha=0.25 + 0.6 * i / len(x))

            # plt.scatter(self.node_coords[agent_ID][0], self.node_coords[agent_ID][1], c='r', s=35)

        plt.subplot(2, 2, 4)
        plt.title('Interesting area')
        x = self.high_info_area[:, 0]
        y = self.high_info_area[:, 1]
        plt.hist2d(x, y, bins=30, vmin=0, vmax=1)


        #plt.suptitle('Cov trace: {:.4g} of with budget {:.4g} '.format(self.cov_trace, BUDGET_SIZE-self.agent_budget[f"{agent_ID}"] ))
        
        plt.suptitle('Cov trace: {:.4g} of with step used budget {:.4g} '.format(self.cov_trace, self.step_used_budget))
        
        # plt.tight_layout()
        #plt.savefig('{}/{}.png'.format(path, n), dpi=150)
        #plt.savefig('./gifs/{agent_ID}.png')
        plt.savefig('./gifs/{}.png'.format(plot_num), dpi=150)




if __name__ == '__main__':
    NUM_REPEAT = 290  ## 10
    NUM_TEST = 1 ## 10
    SAVE_CSV_RESULT = True
    
    NUM_AGENTS = 3
    results = []
    all_results = []

    for j in range(NUM_REPEAT):
        for i in range(NUM_TEST):
            print('Loop:', i, j)
            rig = RIG_planner(i)
            rig.global_planner()
            #print('test successfully')
           
            cov_trace, time_used = rig.agent_planner()
            print(f"total usedtime is {time_used}, final cov_trace is {cov_trace}")
            
            results.append(cov_trace)
            #budget_history = np.array(rig.budget_history)
           # obj_history = np.array(rig.obj_history)
        
        if SAVE_CSV_RESULT:
            csv_filename = f'result/RIG_tree_results.csv'
            csv_data = np.array(results).reshape(-1, NUM_TEST)
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
        all_results.append(results)
        results = []
            
