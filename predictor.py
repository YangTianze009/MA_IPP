import numpy as np
from itertools import product
from copy import deepcopy
from RIG_parameter import *

class predictor:
    def __init__(self, updated_node_coords, sample_size, gaussian, gprocess, sample_number, measurement_points, agent_ID):
        self.node_coords = updated_node_coords
        self.step_sample = sample_size
        self.dist_residual = 0
        self.underlying_distribution = deepcopy(gaussian)
        self.ground_truth = self.get_ground_truth()
        self.gp = deepcopy(gprocess)
        
        self.sample_number = sample_number
        self.measurement_points = measurement_points
        self.agent_ID = agent_ID
        
        #print(f"sample number is {self.sample_number}")
        for agent_i in range(NUM_AGENTS):
                if self.measurement_points[f"{agent_i}"] != []:
                    for j, sample in enumerate(self.measurement_points[f"{agent_i}"]):
                        #if j < self.sample_number:
                        observed_value = self.underlying_distribution.distribution_function(
                            sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        #else:
                            #observed_value = np.array([0])
                        
                        self.gp.add_observed_point(sample, observed_value)
                        
        self.gp.update_gp()
        
        high_info_area = self.gp.get_high_info_area()
        self.cov_trace = self.gp.evaluate_cov_trace(high_info_area)
        


    def get_ground_truth(self):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    def prediction(self, each_step, start, high_info_area):
    
        gp = deepcopy(self.gp)
        
        self.current_node_index = start
        dist = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[int(each_step)])#next_node_index])
        remain_length = dist
        next_length = self.step_sample - self.dist_residual

        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[int(each_step)] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.node_coords[self.current_node_index]
            else:
                self.sample = (self.node_coords[int(each_step)] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.sample
            
            observed_value = np.array([0])
            
            gp.add_observed_point(self.sample, observed_value)        
            
            ###### observed_value = 0
            #observed_value = self.underlying_distribution.distribution_function(
                #self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
            remain_length -= next_length
            next_length = self.step_sample
            no_sample = False
        
        #print(f"observed point is {gp.observed_points}")

        gp.update_gp()
        
        high_info_area = gp.get_high_info_area()# if ADAPTIVE_AREA else None
        cov_trace = gp.evaluate_cov_trace(high_info_area)
        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
 #       print(cov_trace)
        return cov_trace, dist #done, self.node_info, self.node_std, self.budget #reward, done, self.node_info, self.node_std, self.budget
