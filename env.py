import os
import copy
import numpy as np
from itertools import product
from classes import PRMController, Obstacle, Utils
from classes.Gaussian2D import Gaussian2D
from matplotlib import pyplot as plt
from gp_ipp import GaussianProcessForIPP
from parameter import *


class Env():
    def __init__(self, sample_size=500, k_size=10, start=None, destination=None, obstacle=[],
                 budget_range=None, save_image=False, seed=None):
        self.sample_size = sample_size
        self.k_size = k_size
        self.budget = []
        self.budget_range = budget_range
        budget = np.random.uniform(*self.budget_range)
        for i in range(0, NUM_THREADS + 1):
            self.budget.append(0)
        for i in range(1, NUM_THREADS + 1):
            self.budget[i] = budget
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])
        self.obstacle = obstacle
        self.seed = seed

        # generate PRM
        # self.prm = None
        # self.node_coords, self.graph = None, None
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)

        node_coords = dict()
        for i in range(1, NUM_THREADS + 1):
            node_coords[f"{i}"] = []

        graph = dict()
        for i in range(1, NUM_THREADS + 1):
            graph[f"{i}"] = []

        prm = dict()
        for i in range(1, NUM_THREADS + 1):
            prm[f"{i}"] = []


        # different graph is a random seed to ensure that all the agents start at the same node
        for i in range(1, NUM_THREADS + 1):
            prm[f"{i}"] = PRMController(self.sample_size, self.obstacle, self.start, self.destination,
                                        self.budget_range,
                                        self.k_size)
            agent_node_coords, agent_graph = prm[f"{i}"].runPRM(saveImage=False, seed=seed,
                                                                start_pos=np.random.rand(1, 2))
            node_coords[f"{i}"] = agent_node_coords
            graph[f"{i}"] = agent_graph

        # for i in range(1, NUM_THREADS + 1):
        #     print(node_coords[f"{i}"][1])
        #     print(node_coords[f"{i}"][10])
        # print(len(node_coords["1"]))
        # print(node_coords)
        # print(graph)
        self.node_coords = node_coords
        self.graph = graph
        self.prm = prm
        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None
        self.high_info_area = None

        # GP
        self.gp_ipp = GaussianProcessForIPP()
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy((self.node_info, self.node_std, self.budget))
        self.RMSE = None
        self.F1score = None
        self.MI = None
        self.MI0 = None

        ## !! make cov_trace to be list but not nontype
        self.cov_trace = None
        # self.cov_trace = []

        # start point
        self.current_node_index = None
        self.sample = self.start
        self.route = []
        self.dist_residual = 0

        self.save_image = save_image
        self.frame_files = []

    def reset(self, agent_ID, seed=None):
        # generate PRM
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        # self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range, self.k_size)
        # self.budget = np.random.uniform(*self.budget_range)
        # self.node_coords, self.graph = self.prm.runPRM(saveImage=False)
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # underlying distribution
        self.underlying_distribution = Gaussian2D()
        self.ground_truth = self.get_ground_truth()

        # initialize gp
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        self.node_info, self.node_std = self.gp_ipp.update_node(self.node_coords[f"{agent_ID}"])

        # initialize evaluations
        # self.F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
        self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        self.MI = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        self.cov_trace0 = 900

        # save initial state
        self.node_info0, self.node_std0, self.budget[agent_ID] = copy.deepcopy(
            (self.node_info, self.node_std, self.budget0[agent_ID]))

        # start point
        self.current_node_index = 1
        self.sample = self.start
        self.route = []
        np.random.seed(None)

        return self.node_coords[f"{agent_ID}"], self.graph[f"{agent_ID}"], self.node_info, self.node_std, self.budget[
            agent_ID]

    def step(self, route, next_node_index, all_samples, sample_numbers, done, agent_ID,
             measurement=True):
        reward = 0
        done = done
        # get all the observed positions of all agents after making decision
        # all the agents should have the same observation points
        for i in range(1, NUM_THREADS + 1):
            if all_samples[f"{i}"] != []:
                for j, sample in enumerate(all_samples[f"{i}"]):
                    if j < sample_numbers:
                        if measurement:
                            observed_value = self.underlying_distribution.distribution_function(
                                sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        else:
                            observed_value = np.array[0]
                        # print(f"sample is {sample}", f"observed value is {observed_value}")
                        self.gp_ipp.add_observed_point(sample, observed_value)
        # print(f"observation points are {len(self.gp_ipp.observed_points)}")
        # print(f"observation points are {self.gp_ipp.observed_points}")
        ground_truth = self.get_ground_truth()
        self.gp_ipp.update_gp()
        try:
            self.node_info, self.node_std = self.gp_ipp.update_node(self.node_coords[f"{agent_ID}"])
            print(f"node_info is {self.node_info.shape}")
            print(f"node_std is {self.node_std.shape}")
            if measurement:
                self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
                cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
                # print(f"cov_trace is {cov_trace}")
                RMSE = self.gp_ipp.evaluate_RMSE(ground_truth)
                self.RMSE = RMSE
        except:
            cov_trace = self.cov_trace
            RMSE = self.RMSE
            self.RMSE = RMSE
            print("bug occurs")
        # self.F1score = F1score
        if next_node_index in route[-2:]:
            reward += -0.1

        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        self.cov_trace = cov_trace

        if done:
            reward -= cov_trace / 50
            # print(f"reward is {reward}")
        self.gp_ipp.clear_observed_point()
        assert self.budget[agent_ID] >= 0
        return reward, done, self.node_info, self.node_std, ground_truth

    def observed_positions(self, current_node_index, next_node_index, sample_length, agent_ID, dist_residual):
        sample_coordinates = []
        dist = np.linalg.norm(
            self.node_coords[f"{agent_ID}"][current_node_index] - self.node_coords[f"{agent_ID}"][next_node_index])
        remain_length = dist
        next_length = sample_length - dist_residual
        no_sample = True
        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[f"{agent_ID}"][next_node_index] - self.node_coords[f"{agent_ID}"][
                    current_node_index]) * next_length / dist + self.node_coords[f"{agent_ID}"][current_node_index]
            else:
                self.sample = (self.node_coords[f"{agent_ID}"][next_node_index] - self.node_coords[f"{agent_ID}"][
                    current_node_index]) * next_length / dist + self.sample

            sample_coordinates.append(self.sample)
            remain_length -= next_length
            next_length = sample_length
            no_sample = False

        dist_residual = dist_residual + remain_length if no_sample else remain_length
        self.budget[agent_ID] -= dist
        # print(f"self.budget is {self.budget}", "\n", f"distance is {dist}")
        self.route.append(next_node_index)
        # assert self.budget >= 0  # Dijsktra filter

        return sample_coordinates, self.budget[agent_ID], dist_residual

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    def plot(self, route, n, path, ground_truth, remain_budget, agent_ID, CMAES_route=False):
        # Plotting shorest path
        plt.switch_backend('agg')

        self.gp_ipp.plot(ground_truth)

        # plt.subplot(1,3,1)
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        plt.scatter(self.node_coords[f"{agent_ID}"][1][0], self.node_coords[f"{agent_ID}"][1][1], c='r', marker='*', s=15 ** 2)

        for ID in range(1, NUM_THREADS + 1):
            if CMAES_route:
                pointsToDisplay = route[f"{ID}"]
            else:
                pointsToDisplay = [(self.prm[f"{ID}"].findPointsFromNode(path)) for path in route[f"{ID}"]]

            x = [item[0] for item in pointsToDisplay]
            y = [item[1] for item in pointsToDisplay]
            for i in range(len(x) - 1):
                plt.plot(x[i:i + 2], y[i:i + 2], c=colorlist[ID - 1], linewidth=4, zorder=5,
                         alpha=0.25 + 0.6 * i / len(x))

            # plt.scatter(self.node_coords[agent_ID][0], self.node_coords[agent_ID][1], c='r', s=35)

        '''
        if sampling_path:
            pointsToDisplay2 = [(self.prm.findPointsFromNode(path)) for path in sampling_path]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(x0[i:i + 2], y0[i:i + 2], c='white', linewidth=4, zorder=5, alpha=1 - 0.2 * i / len(x0))
            for i in range(len(x1) - 1):
                plt.plot(x1[i:i + 2], y1[i:i + 2], c='red', linewidth=4, zorder=6)
        '''

        plt.subplot(2, 2, 4)
        plt.title('Interesting area')
        x = self.high_info_area[:, 0]
        y = self.high_info_area[:, 1]
        plt.hist2d(x, y, bins=30, vmin=0, vmax=1)

        '''
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]

        for i in range(len(x) - 1):
            plt.plot(x[i:i + 2], y[i:i + 2], c='black', linewidth=4, zorder=5, alpha=0.25 + 0.6 * i / len(x))
        
        plt.suptitle('Budget: {:.4g}/{:.4g},  Cov trace: {:.4g}'.format(
            self.budget, self.budget0, self.cov_trace))       #### !!!!!!!!!
        
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size), dpi=150)
        #print('already saved') #### !!!!
        #plt.show() ## change here !! default commented
        frame = '{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size)
        self.frame_files.append(frame)
        '''

        plt.suptitle('Cov trace: {:.4g}  remain_budget: {:.4g}'.format(self.cov_trace, remain_budget))
        # plt.tight_layout()
        plt.savefig('{}/{}.png'.format(path, n), dpi=150)
        # plt.savefig('./gifs/samples.png')


if __name__ == '__main__':
    env = Env(100, save_image=True)
    nodes, graph, info, std, budget = env.reset()
    print(nodes)
    print(graph)
