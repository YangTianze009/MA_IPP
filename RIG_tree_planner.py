'''
Adaptive implementation of RRT_star
Replans after every budget expenditure of ~0.4

Written By - A. Vashisth (@accgen99)
'''
import csv
import os
import numpy as np
from copy import deepcopy
import time
from RRTGenerator import RRTGraph

from RIG_original import *
from classes.Gaussian2D import *
from classes.Graph import *
from gp_ipp import *
from predictor import predictor


class RIG_planner:
    def __init__(self, i):
        self.path = f'RRT_s_IPP/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.i = i
        self.env = None
        self.rrt = None
        self.budget = 8.0
        self.radius = 0.8
        self.step_sample = 0.2
        self.info = None
        self.budget_history = []
        self.obj_history = []

        self.gaussian = Gaussian2D(i)
        np.random.seed() # Remove seed set by Gaussian2D()
        self.node_coords = np.array([[]])

    def planner(self):
        ti = time.time()
        self.Tree = Graph()

        # Create copy of ground truth for future use
        self.underlying_distribution = deepcopy(self.gaussian)

        # Initialize GP
        self.gp = GaussianProcessForIPP()
        self.ground_truth = self.get_ground_truth()
        self.high_info_area = self.gp.get_high_info_area()

        # Start node & init RRT
        start = np.array([0.0, 0.0])
        generator = RRTGraph(self.step_sample)

        self.updated_node_coords = np.array([[]])
        # Find tree, initially all infos = 0.0
        counts = 0
        self.node_coords = np.array([[]])

        # Execution stuff
        self.dist_residual = 0
        self.added_node_coord = np.array([[0.0, 0.0]])

        while self.budget > 0:
            self.current_node_index = 0 # start_node
            start_node_index = 0
            pre_length = len(self.updated_node_coords.reshape(-1,2))

            if counts != 0:
                pre_length -= 1

            # Generate tree
            self.rrt = RRT(50, 1.0, 1.0, self.radius, self.step_sample, self.gp, self.underlying_distribution)

            nodes = self.rrt.RRT_planner(start, iterations=175, info = self.gp)
            for each_node in nodes:
                append_it = np.array([[each_node.x, each_node.y]])
                self.node_coords = np.array(np.append(self.node_coords, append_it))

            self.updated_node_coords = np.array(np.append(self.updated_node_coords, self.node_coords))
            print(f"node coords shape is {self.node_coords.shape}")
            self.node_coords = self.node_coords.reshape(-1,2)
            
            # Generating graph & predictor
            Predictor = predictor(self.node_coords, self.step_sample, self.gaussian, self.gp)
            graph = generator.create_graph(self.node_coords)

            cost = 0.0
            path = []
            while cost < 0.3:
                self.cov_trace = float('infinity')
                connected_edges = graph.edges[str(start_node_index)]
                for node, edge in connected_edges.items():
                    if edge.length != 0.0 and node not in path:
                        pred_copy = deepcopy(Predictor)
                        cov_new, dist = pred_copy.prediction(node, int(start_node_index), self.high_info_area)
                        if cov_new < self.cov_trace:
                            self.cov_trace = cov_new
                            best_node = node
                            length = dist

                start_node_index = best_node
                path.append(best_node)
                cost += length

            start_index = 0

            for each_step in path:
                if self.budget < 0.0:
                    break
                self.added_node_coord = np.append(self.added_node_coord, self.node_coords[int(each_step)])
                covariance_trace = self.execute_path(each_step, start_index)
                start_index = int(each_step)
                self.budget_history.append(8-self.budget)
                self.obj_history.append(covariance_trace)

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
        return covariance_trace, tf-ti

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
#        self.current_node_index = int(each_step)
        return cov_trace #done, self.node_info, self.node_std, self.budget #reward, done, self.node_info, self.node_std, self.budget


    def get_ground_truth(self):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth



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
    for j in range(NUM_REPEAT):
        for i in range(NUM_TEST):
            print('Loop:', i, j)
            rig = RIG_planner(i)
            #print('test successfully')
            c, t = rig.planner()
            print(f"current time is {t}, current cov_trace is {c}")
            sub_results.append(c)
            budget_history = np.array(rig.budget_history)
            obj_history = np.array(rig.obj_history)
            if SAVE_TRAJECTORY_HISTORY:
                csv_filename2 = f'result/CSV2/RIG_tree_1_results.csv'
                new_file = False if os.path.exists(csv_filename2) else True
                field_names = ['budget', 'obj']
                with open(csv_filename2, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.concatenate((budget_history.reshape(-1, 1), obj_history.reshape(-1, 1)), axis=-1)
                    writer.writerows(csv_data)
            results.append(sub_results)
            sub_results = []
        if SAVE_CSV_RESULT:
            csv_filename = f'result/CSV/RIG_tree_results.csv'
            csv_data = np.array(results).reshape(-1, NUM_TEST)
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
        all_results.append(results)
        results = []
