import cma
import imageio
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv
from copy import deepcopy
from matplotlib.pylab import *

from env import Env
from test_parameters import *


class CIPPWorker():
    def __init__(self, global_step, seed=None, save_image=False):
        np.random.seed(seed)
        self.env = Env(sample_size=10, k_size=1, start=(0,0), destination=(1,1), budget_range=BUDGET_RANGE, seed=seed)
        self.env.reset()
        np.random.seed(seed)
        self.horizon = 4
        self.dtr = 0.4
        self.budget_factor = 0.66

        self.local_budget = self.env.budget0
        self.start = self.env.start
        self.end = self.env.destination
        self.next_iter_end = None
        self.temp_observed_points = []
        self.temp_observed_value = []

        self.save_image = save_image
        self.global_step = global_step

        self.budget_history = []
        self.obj_history = []
        self.obj2_history = []
        self.planning_time = 0

    def object_function(self, X):
        control_points = X.reshape(-1,2)
        route = np.concatenate((self.start, control_points, self.end))
        self.env.gp_ipp.observed_points = deepcopy(self.temp_observed_points)
        self.env.gp_ipp.observed_value = deepcopy(self.temp_observed_value)
        cov_trace = self.env.route_step(route, SAMPLE_LENGTH, measurement=False)
        b_penalty, _ = self.budget_penalty(X)
        # p_penalty = self.progress_penalty(route)

        return cov_trace*b_penalty

    def budget_penalty(self, X):
        control_points = X.reshape(-1,2)
        complete_route = np.concatenate((self.start,control_points, self.end))
        dist = np.linalg.norm(complete_route[1:,:]-complete_route[:-1,:],axis=-1).sum()
        # print(complete_route[1:,:]-complete_route[:-1,:])
        # print(dist)
        if dist-self.local_budget <=0:
            return 1, dist
        else:
            return 1 / np.exp(-(dist+0.5-self.local_budget)**2), dist

    def progress_penalty(self, planned_route):
        travel_dist = 0
        curr = planned_route[0]
        for i in range(planned_route.shape[0]):
            travel_dist += np.linalg.norm(planned_route[i]-curr)
            curr = planned_route[i]
            if travel_dist >= self.dtr:
                p_tr = planned_route[i]
                break

        x = self.horizon-np.linalg.norm(p_tr-self.next_iter_end)
        if x <=0:
            return 1
        else:
            return 1 / np.exp(-x**2)

    def cmaes_global_plan(self):
        x1 = np.linspace(self.env.start[0,0], self.env.destination[0,0], 50).reshape(-1,1)
        x2 = np.linspace(self.env.start[0,1], self.env.destination[0,1], 50).reshape(-1,1)
        initial_route = np.concatenate((x1,x2),axis=-1).reshape(-1,2)
        control_points = initial_route[1:-1].reshape(1,-1)

        es = cma.fmin(self.object_function, control_points, 0.25, options={'maxiter': 500, 'bounds': [0,1], 'popsize': 50, 'tolfun':0.1})
        planned_points = es[0].reshape(-1,2)
        planned_route = np.concatenate((self.env.start, planned_points, self.env.destination))
        _, dist = self.budget_penalty(planned_points)

        high_info = self.env.high_info_area
        cov_trace = self.env.gp_ipp.evaluate_cov_trace(high_info)
        print(cov_trace)
        return cov_trace

    def cmaes_plan(self, start, end):
        start = start.reshape(-1,2)
        end = end.reshape(-1,2)
        x1 = np.linspace(start[0,0], end[0,0], 10)
        x2 = np.linspace(start[0,1], end[0,1], 10)

        self.temp_observed_points = self.env.gp_ipp.observed_points
        self.temp_observed_value = self.env.gp_ipp.observed_value

        initial_route = np.concatenate((x1, x2), axis=-1).reshape(-1,2)
        control_points = initial_route[1:-1].reshape(1, -1)
        es = cma.fmin(self.object_function, control_points, 0.25, options={'maxiter': 100, 'bounds':[0,1], 'popsize': 10})
        planned_points = es[0].reshape(-1, 2)
        planned_route = np.concatenate((start, planned_points, end))

        self.env.gp_ipp.observed_points = self.temp_observed_points
        self.env.gp_ipp.observed_value = self.temp_observed_value

        return planned_route

    def execute_path(self, planned_route):
        curr = planned_route[0]
        travel_dist = 0
        for i in range(planned_route.shape[0]):
            length = np.linalg.norm(planned_route[i]-curr, axis=-1)
            if travel_dist + length < self.dtr:
                travel_dist += length
                curr = planned_route[i]
                route = planned_route[:i+1]
                next_start = curr
            if travel_dist+length >= self.dtr:
                d = length - (travel_dist+length-self.dtr)
                next_start = curr + (planned_route[i] - curr)*d/length
                travel_dist = self.dtr
                break
        next_start = next_start.reshape(1,2)
        route = np.concatenate((route,next_start))

        self.env.route_step(route, SAMPLE_LENGTH)
        # for i in range(len(route)-1):
        #     node = route[i:i+1,:]
        #     self.env.route_step(node, SAMPLE_LENGTH)
        self.budget_history.append(self.env.budget0 - self.env.budget)
        self.obj_history.append(self.env.cov_trace)
        self.obj2_history.append(self.env.RMSE)
        executed_path = route[1:]

        return next_start, travel_dist, executed_path

    def adaptive_cmaes_plan(self):
        global_start = deepcopy(self.env.start)
        global_destination = deepcopy(self.env.destination)
        pg_length = np.linalg.norm(global_start-global_destination)

        budget_limit = self.horizon * (1 + self.budget_factor)
        dtr_g = self.dtr/(1+self.budget_factor)
        self.start = global_start
        travel_dist = 0
        self.local_budget = np.min([budget_limit, self.env.budget])
        real_path = global_start
        i = 0
        while True:
            i += 1
            t1 = time.time()
            if travel_dist/(1+self.budget_factor) + self.horizon > pg_length:
                self.end = global_destination
                self.next_iter_end = global_destination
            else:
                self.end = global_start+(global_destination-global_start)*(travel_dist/(1+self.budget_factor)+self.horizon)/pg_length
                self.next_iter_end = global_start + (global_destination - global_start) * (
                            travel_dist / (1 + self.budget_factor) + dtr_g + self.horizon) / pg_length
            d_pr = np.min([self.horizon, self.budget_factor*self.env.budget])
            planned_route = self.cmaes_plan(self.start, self.end)
            t2 = time.time()
            self.planning_time += t2-t1
            self.start, dist, executed_path = self.execute_path(planned_route)
            real_path = np.concatenate((real_path, executed_path))

            if self.save_image:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                self.env.plot(real_path, self.global_step, i, result_path, CMAES_route=True)

            travel_dist = self.env.budget0-self.env.budget
            print(self.env.budget)
            self.local_budget = np.min([budget_limit, self.env.budget])
            if self.local_budget <= dtr_g:
                travel_dist += np.linalg.norm(real_path[-1]-global_destination)
                self.execute_path(np.concatenate((real_path[-1].reshape(1,2),global_destination.reshape(1,2))))
                real_path = np.concatenate((real_path, global_destination))
                if self.save_image:
                    self.env.plot(real_path, self.global_step, i+1, result_path, CMAES_route=True)

                break
        high_info = self.env.gp_ipp.get_high_info_area()
        cov_tr = self.env.gp_ipp.evaluate_cov_trace(high_info)

        if self.save_image:
            self.make_gif(result_path, self.global_step)
        return cov_tr

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "8"
    result_path = f'result/CMAES'
    mean_cov_trace = 0
    results1 = []
    results2 = []
    sub_results1 = []
    sub_results2 = []
    for j in range(10):
        for i in range(NUM_TEST):
            seed = 1+100*(i+1)
            example = CIPPWorker(i, seed=seed, save_image=False)
            budget = int(deepcopy(example.env.budget0))+1
            # cov_trace = example.cmaes_global_plan()
            cov_trace = example.adaptive_cmaes_plan()
            sub_results1.append(cov_trace)
            sub_results2.append(example.planning_time)
            mean_cov_trace = (cov_trace + mean_cov_trace*i)/(i+1)
            print(mean_cov_trace)
            budget_history = np.array(example.budget_history)
            obj_history = np.array(example.obj_history)
            obj2_history = np.array(example.obj2_history)
            if SAVE_TRAJECTORY_HISTORY:
                csv_filename2 = f'result/CSV2/Budget_'+str(budget)+'_CMAES_trajectory_results.csv'
                new_file = False if os.path.exists(csv_filename2) else True
                field_names = ['budget', 'obj', 'obj2']
                with open(csv_filename2, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)

                    csv_data = np.concatenate((budget_history.reshape(-1, 1), obj_history.reshape(-1, 1), obj2_history.reshape(-1, 1)), axis=-1)
                    writer.writerows(csv_data)
            results1.append(sub_results1)
            results2.append(sub_results2)
            sub_results1 = []
            sub_results2 = []
        if SAVE_CSV_RESULT:
            csv_filename = f'result/CSV/Budget_'+str(budget)+'_CMAES_results.csv'
            csv_data = np.array(results1).reshape(-1, NUM_TEST)
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            results1 = []
            csv_filename = f'result/CSV3/Budget_'+str(budget)+'_CMAES_planning_time.csv'
            csv_data = np.array(results2).reshape(-1, NUM_TEST)
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            results2 = []