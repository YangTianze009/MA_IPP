import copy
import os
import time
import math
import imageio
import csv
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from env import Env
from raor_parameters import *
from parameter import ADAPTIVE_TH
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class TSPSolver:
    def __init__(self, env):
        self.env = env
        self.local_budget = self.env.budget0
        self.start = self.env.start
        self.end = self.env.destination

    def create_data_model(self, coords):
        """Stores the data for the problem."""
        data = dict()
        # Locations in block units
        data['locations'] = coords  # yapf: disable
        data['num_vehicles'] = 1
        data['starts'] = [1]
        data['ends'] = [0]
        return data

    def compute_euclidean_distance_matrix(self, locations):
        """Creates callback to return distance between points."""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    distances[from_counter][to_counter] = (int(math.hypot((from_node[0] - to_node[0]),(from_node[1] - to_node[1]))))
        return distances

    def print_solution(self, manager, routing, solution):
        """Prints solution on console."""
        route = [1]
        # print('Objective: {}'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        # plan_output = 'Route: '
        route_distance = 0
        while not routing.IsEnd(index):
            # plan_output += ' {} -'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            route.append(manager.IndexToNode(index))
        # plan_output += ' {}\n'.format(manager.IndexToNode(index))
        # print(plan_output)
        # plan_output += 'Objective: {}m\n'.format(route_distance)
        return route

    def run_solver(self, coords):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = self.create_data_model(coords)
        distance_matrix = self.compute_euclidean_distance_matrix(data['locations'])
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                               data['num_vehicles'], data['starts'], data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            route = self.print_solution(manager, routing, solution)
        return route


class RAOr:
    def __init__(self, sample_size=100, seed=None, show_img=False):
        self.sample_frac_init = 0.4
        self.magnify = 10000 # for numerical calc in ortools
        self.sample_length = 0.2
        self.iter_times = 3  # 3 by default
        self.seed = seed
        self.env = Env(sample_size=sample_size, k_size=1, budget_range=(7.99999,8), start=(0,0), destination=(1,1), seed=seed)
        self.env.reset(self.seed)
        self.tsp = TSPSolver(self.env)
        self.budget0 = self.env.budget0
        self.start, self.end = self.env.start.squeeze()*self.magnify, self.env.destination.squeeze()*self.magnify
        self.coords = self.env.node_coords * self.magnify
        self.graph = list(self.env.graph.values())
        self.tsp_coords = []
        self.sample_idx = []
        self.show_img = show_img
        self.frame_names = []
        self.budget_history = []
        self.obj_history = []
        self.rmse_history = []
        self.time_history = None
        self.path = 'result/ipp_raor_non'
        self.ADAPTIVE = True
        if self.ADAPTIVE:
            self.epsilon = 1
            self.dtr = 0.4
            self.horizon = 4
            self.budget_factor = 0.66
            self.path = f'result/ipp_raor_dtr{self.dtr}_eps{self.epsilon}_n{self.env.sample_size}_h{self.horizon}_complete'

    def sample_uniform(self, exist_route):
        num_nodes = len(self.coords) - 2
        num_sample_nodes = int(num_nodes * self.sample_frac_init)
        start_idx = exist_route[-1]
        start_coord = self.coords[start_idx] / self.magnify
        end_coord = self.coords[0] / self.magnify
        while True:
            random_idx = np.random.choice(range(1, len(self.coords)), size=num_sample_nodes, replace=False).tolist()
            random_idx = list(filter(start_idx.__ne__, random_idx))
            # for v in exist_route:
            #     random_idx = list(filter(v.__ne__, random_idx))
            random_idx_new = random_idx.copy()
            for i in random_idx:
                coord = self.coords[i] / self.magnify
                dist = self.env.prm.calcDistance(coord, start_coord) + self.env.prm.calcDistance(coord, end_coord)
                if dist > self.env.budget:
                    random_idx_new.remove(i)
            if random_idx_new.__len__() != 0:
                break
        self.sample_idx = [0, start_idx] + random_idx_new
        self.tsp_coords = []
        for idx in self.sample_idx:
            self.tsp_coords.append(self.coords[idx].squeeze())
        return

    def sample_curr2end(self, exist_route):
        start_idx = exist_route[-1]
        self.sample_idx = [0, start_idx]
        self.tsp_coords = []
        for idx in self.sample_idx:
            self.tsp_coords.append(self.coords[idx].squeeze())
        return


    def get_graph_idx(self, idx):
        graph_idx = []
        for i in idx:
            graph_idx.append(self.sample_idx[i])
        return graph_idx

    def clean_coords(self, idx_pre):
        self.tsp_coords = []
        start_idx = idx_pre[0]
        idx_pre = list(filter(start_idx.__ne__, idx_pre))
        self.sample_idx = [0, start_idx] + list(set(idx_pre))[1:] # set will start with 0
        for idx in self.sample_idx:
            self.tsp_coords.append(self.coords[idx].squeeze())
        return

    def new_sample(self, graph_idx):
        num_nodes = self.coords.__len__() - 2
        num_sample = 1
        v_new = np.random.choice(range(2, num_nodes + 2), size=num_sample, replace=False).tolist()
        for v in v_new:
            if v in graph_idx:
                graph_idx = list(filter(v.__ne__, graph_idx))
            else:
                graph_idx.insert(1, v)
        return graph_idx.copy()

    def calc_prob(self):
        self.prob = []
        coords = self.coords[1:].copy() / self.magnify
        y_pred, y_std = self.env.gp_ipp.gp.predict(coords, return_std=True)
        for i in range(y_std.size):
            p = y_pred[i] + y_std[i]*1
            self.prob.append(p.item())
        self.prob = np.asarray(self.prob)
        self.prob /= np.sum(self.prob)

    def new_sample_adaptive(self, graph_idx, exist_route):
        num_nodes = self.coords.__len__() - 2
        num_sample = 1
        graph_idx0 = graph_idx.copy()
        start_idx = exist_route[-1]
        while graph_idx0 == graph_idx:
            flag = True
            while flag:
                # v_new = np.random.choice(range(1, num_nodes + 2), p=self.prob, size=num_sample, replace=False).tolist()
                v_new = np.random.choice(range(1, num_nodes + 2), size=num_sample, replace=False).tolist()
                for v in v_new:
                    if v not in exist_route and v != start_idx: # TODO
                        flag = False
            for v in v_new:
                # coord = self.coords[v] / self.magnify
                # y_pred, y_std = self.env.gp_ipp.gp.predict(np.expand_dims(coord, axis=0), return_std=True)
                # interest = y_pred + y_std * 1
                if np.random.rand() < self.epsilon:
                    if v in graph_idx:
                        graph_idx = list(filter(v.__ne__, graph_idx))
                    else:
                        graph_idx.insert(1, v)
                # else:
                #     if interest >= ADAPTIVE_TH:
                #         if v in graph_idx:
                #             pass
                #         else:
                #             graph_idx.insert(1, v)
                #     else:
                #         if v in graph_idx:
                #             graph_idx = list(filter(v.__ne__, graph_idx))
                #         else:
                #             pass
        return graph_idx.copy()

    def calc_budget(self, graph_idx, used_budget):
        distance = 0
        for i in range(len(graph_idx) - 1):
            distance += self.length(graph_idx[i], graph_idx[i+1])
        budget_limit = min(self.budget0 - used_budget, self.horizon * (1 + self.budget_factor))
        return budget_limit-(distance-used_budget)

    def get_cov_trace(self, route, route_ahead=[], measurement=True):
        if measurement:
            env = self.env
        else:
            env = copy.deepcopy(self.env)
        env.reset(self.seed)
        true_coords = []
        for i in route:
            coord = self.coords[i].squeeze()
            true_coords.append(coord / self.magnify)
        cov_trace = env.route_step(true_coords, self.sample_length, measurement=True)
        if route_ahead:
            true_coords_ahead = []
            for j in route_ahead:
                coord = self.coords[j].squeeze()
                true_coords_ahead.append(coord / self.magnify)
            cov_trace = env.route_step(true_coords_ahead, self.sample_length, measurement=False)
        # print(len(env.high_info_area))
        return cov_trace

    def plot(self, route, n1, n2):
        pointsToDisplay = []
        for i in route:
            pointsToDisplay.append(self.coords[i] / self.magnify)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.env.gp_ipp.plot(self.env.ground_truth)
        # plt.subplot(1, 3, 1)
        plt.scatter(self.env.start[:, 0], self.env.start[:, 1], c='r', s=15)
        plt.scatter(self.env.destination[:, 0], self.env.destination[:, 1], c='r', s=15)

        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(n1 - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], c='black', linewidth=2, zorder=5, alpha=0.25 + 0.7 * i / len(x))
        for i in range(n1-1, len(x) - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], c='white', linewidth=2, zorder=5, alpha=0.25 + 0.7 * i / len(x))

        if self.ADAPTIVE:
            plt.subplot(2, 2, 4)
            plt.title('Interesting area')
            x = self.env.high_info_area[:, 0]
            y = self.env.high_info_area[:, 1]
            plt.hist2d(x, y, bins=30, vmin=0, vmax=1)
            plt.scatter(self.env.start[:, 0], self.env.start[:, 1], c='r', s=15)
            plt.scatter(self.env.destination[:, 0], self.env.destination[:, 1], c='r', s=15)

        plt.suptitle('Budget: {:.4g}/{:.4g},  RMSE: {:.4g}, F1score:{:.4g}, CovTrace:{:.4g}'.format(
            self.env.budget, self.budget0, self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth),
            self.env.gp_ipp.evaluate_F1score(self.env.ground_truth), self.env.cov_trace))
        figname = f'{self.path}/{self.seed}_{n1}_{n2}.png'
        plt.savefig(figname, dpi=150)
        self.frame_names.append(figname)
        # plt.show()
        plt.close()

    def make_gif(self, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(self.path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.frame_names:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')
        for filename in self.frame_names[:-1]:
            os.remove(filename)

    def run_RAOr(self, exist_route=[1], used_budget=0):
        # self.calc_prob()
        dtr_g = self.dtr / (1 + self.budget_factor)
        while True:
            self.sample_curr2end(exist_route)
            route_idx = self.tsp.run_solver(self.tsp_coords)
            route = self.get_graph_idx(route_idx)
            # route = self.check_seq(route_pre)  # init
            route_best = route.copy()
            cov_trace_best = self.get_cov_trace(exist_route, route, measurement=False)
            route_all = exist_route + route[1:]
            if self.env.budget >= 0:
                if self.show_img:
                    self.plot(route_all, len(exist_route), 0)
                break
            else:
                print('Overbudget at beginning. Error')
        for cnt in range((self.coords.__len__() - exist_route.__len__()) * self.iter_times):
        # for cnt in range((self.coords.__len__()) * self.iter_times):
            route_modified = self.new_sample_adaptive(route, exist_route)
            self.clean_coords(route_modified)
            route_idx = self.tsp.run_solver(self.tsp_coords)
            route = self.get_graph_idx(route_idx)
            # route = self.check_seq(route_pre)
            route_all = exist_route + route[1:]
            remain_budget = self.calc_budget(route_all, used_budget)
            if remain_budget < 0:
                # print('Overbudget.')
                # if remain_budget < - self.budget0 / 2:
                #     print(f'Remain budget: {remain_budget}, BREAK!')
                #     break
                continue
            cov_trace = self.get_cov_trace(exist_route, route, measurement=False)
            if cov_trace and cov_trace < cov_trace_best:
                route_best = route.copy()
                cov_trace_best = cov_trace
                if self.show_img:
                    self.plot(route_all, len(exist_route), cnt + 1)
        if self.show_img:
            self.make_gif(len(exist_route))
        return cov_trace_best, route_best

    def length(self, idx1, idx2):
        c1 = self.coords[idx1] / self.magnify
        c2 = self.coords[idx2] / self.magnify
        return np.linalg.norm(c1-c2)

    def run_RAOr_adaptive(self):
        cnt = 0
        final_route = [1]
        n_iter = 0
        dist = 0
        cov, route = self.run_RAOr()
        cov_list = [900]
        t0 = time.time()
        while True:
            cnt += 1
            self.frame_names = []
            for i in range(len(route) - 1):
                dist += self.length(route[i], route[i + 1])
                n_iter_new = dist // self.dtr
                if n_iter_new != n_iter:
                    n_iter = n_iter_new
                    final_route += route[1: i+2]
                    cov = self.get_cov_trace(final_route, measurement=True)
                    break
                elif self.env.budget0 - dist <= self.dtr:
                    final_route += route[1: i+2]
                    cov = self.get_cov_trace(final_route, measurement=True)
                    break
            cov1, route = self.run_RAOr(exist_route=final_route, used_budget=dist)
            cov_list.append(cov)
            rmse = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            self.budget_history.append(dist)
            self.obj_history.append([cov, rmse])
            print('Main loop info:', cnt, dist, cov, cov1, final_route)
            if 0 in final_route:
                break
        if self.show_img:
            self.plot(final_route, len(final_route), 100000)
        self.time_history = time.time() - t0
        print('Time span:', self.time_history)
        print('Cov list:', cov_list)
        print('Route:', final_route)
        print('Iter times:', cnt)
        return cov, final_route



if __name__ == '__main__':
    NUM_REPEAT = 10
    sub_results = []
    results = []
    all_results = []
    time_span = []
    t0 = time.time()
    for j in range(NUM_REPEAT):
        for i in range(NUM_TEST):
            print('Loop:', j, i)
            seed = (i+1) * 100 + 1
            raor = RAOr(sample_size=10, seed=seed, show_img=False)
            covtr, _ = raor.run_RAOr_adaptive()
            sub_results.append(covtr)
            budget_history = np.array(raor.budget_history)
            obj_history = np.array(raor.obj_history)
            time_span.append(raor.time_history)
            if SAVE_TRAJECTORY_HISTORY:
                csv_filename2 = f'result/CSV2/RAOr_2-8n.csv'
                new_file = False if os.path.exists(csv_filename2) else True
                field_names = ['budget', 'obj', 'obj2']
                with open(csv_filename2, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if new_file:
                        writer.writerow(field_names)
                    csv_data = np.concatenate((budget_history.reshape(-1, 1), obj_history.reshape(-1, 2)), axis=-1)
                    writer.writerows(csv_data)
            results.append(sub_results)
            sub_results = []
        if SAVE_CSV_RESULT:
            csv_filename = f'result/CSV/RAOr_1-8n.csv'
            csv_data = np.array(results).reshape(-1, NUM_TEST)
            with open(csv_filename, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
        all_results.append(results)
        results = []
    if SAVE_TIME_RESULT:
        csv_filename3 = f'result/CSV3/RAOr_3-8n.csv'
        csv_data = np.array(time_span).reshape(NUM_REPEAT, NUM_TEST)
        with open(csv_filename3, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
    print(time_span)
    print('Avg time:', np.mean(time_span))
    print('Results:', all_results)
    print('Mean:', np.mean(all_results))
    print('Stddev:', np.std(all_results))

