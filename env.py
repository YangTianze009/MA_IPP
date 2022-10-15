import os
import copy
import numpy as np
from itertools import product
from classes import PRMController, Obstacle, Utils
from classes.Gaussian2D import Gaussian2D
from matplotlib import pyplot as plt
from matplotlib import cm
from gp_ipp import GaussianProcessForIPP
from parameter import *
from sklearn.cluster import KMeans
import math
from scipy.stats import multivariate_normal


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


class Env():
    def __init__(self, sample_size=500, k_size=10, start=None, destination=None, obstacle=[],
                 budget_range=None, save_image=False, seed=None):
        self.gp_intent_map = None
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

        start_pos = np.random.rand(1, 2)
        # start_pos = np.array([0, 0])
        # coordinates = np.random.rand(self.sample_size, 2)
        # different graph is a random seed to ensure that all the agents start at the same node
        for i in range(1, NUM_THREADS + 1):
            coordinates = np.random.rand(self.sample_size, 2)
            prm[f"{i}"] = PRMController(self.sample_size, self.obstacle, self.start, self.destination,
                                        self.budget_range,
                                        self.k_size)
            agent_node_coords, agent_graph = prm[f"{i}"].runPRM(saveImage=False, seed=seed,
                                                                start_pos=start_pos, coordinates=coordinates)

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
        self.gp_ipp_sampling_first = GaussianProcessForIPP()
        self.node_info, self.node_std = None, None
        self.intent_node_info, self.intent_node_std = None, None
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

    def step(self, cov_trace_before, route, next_node_index, done, agent_ID, all_samples_sampling, sampling,
             measurement=True):
        # print(f"all samples is {all_samples}")
        # print(f"all_samples_sampling is {all_samples_sampling}")
        reward = 0
        cov_trace = 0
        done = done
        # get all the observed positions of all agents after making decision
        # all the agents should have the same observation points
        ground_truth = self.get_ground_truth()
        if sampling:
            gp_ipp_reward = GaussianProcessForIPP()
            for i in range(1, NUM_THREADS + 1):
                if all_samples_sampling[f"{i}"] != []:
                    for sample in all_samples_sampling[f"{i}"]:
                        observed_value = self.underlying_distribution.distribution_function(
                            sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        # print(f"sample is {sample}", f"observed value is {observed_value}")
                        gp_ipp_reward.add_observed_point(sample, observed_value)

            # print(f"observed_points are {len(gp_ipp_reward.observed_points)}")
            gp_ipp_reward.update_gp()
            self.high_info_area = gp_ipp_reward.get_high_info_area() if ADAPTIVE_AREA else None
            cov_trace = gp_ipp_reward.evaluate_cov_trace(self.high_info_area)
            # print(f"cov_trace in step is {cov_trace}")

        if next_node_index in route[-2:]:
            reward += -0.1

        elif cov_trace_before > cov_trace:
            reward += (cov_trace_before - cov_trace) * 2 / cov_trace_before

        if done:
            reward -= cov_trace / 50
            # print(f"reward is {reward}")
        self.gp_ipp.clear_observed_point()
        assert self.budget[agent_ID] >= 0
        return reward, done, ground_truth, cov_trace

    # plot the intent map using Gaussian distribution and gain the intent value
    def construct_intent_map(self, gaussian_mean, gaussian_cov, agent_ID, node_coordinates):
        intent_info = np.zeros((len(node_coordinates), 1))
        for i in range(1, NUM_THREADS + 1):
            if gaussian_mean[f"{i}"] != [] and i != agent_ID:
                Gaussian = multivariate_normal(mean=gaussian_mean[f"{i}"], cov=gaussian_cov[f"{i}"])
                for i in range(len(node_coordinates)):
                    X, Y = np.array(node_coordinates[i][0]), np.array(node_coordinates[i][1])
                    d = np.dstack([X, Y])
                    intent_info[i] += Gaussian.pdf(d) / 124.8
        return intent_info

    def get_node_information(self, all_samples, sample_numbers, agent_ID, agent_position):
        gp_ipp_info = GaussianProcessForIPP()
        if not PARTIAL_GP:
            for i in range(1, NUM_THREADS + 1):
                if all_samples[f"{i}"] != []:
                    for j, sample in enumerate(all_samples[f"{i}"]):
                        if j < sample_numbers:
                            # print(f"j is {j}", f"sample is {sample}", type(sample))
                            observed_value = self.underlying_distribution.distribution_function(
                                sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                        else:
                            observed_value = np.array([0])
                        # print(f"sample is {sample}", f"observed value is {observed_value}")
                        gp_ipp_info.add_observed_point(sample, observed_value)
        else:
            if all_samples[f"{agent_ID}"] != []:
                for j, sample in enumerate(all_samples[f"{agent_ID}"]):
                    if j < sample_numbers:
                        # print(f"j is {j}", f"sample is {sample}", type(sample))
                        observed_value = self.underlying_distribution.distribution_function(
                            sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                    else:
                        observed_value = np.array([0])
                    # print(f"sample is {sample}", f"observed value is {observed_value}")
                    gp_ipp_info.add_observed_point(sample, observed_value)

        gp_ipp_info.update_gp()
        # cov1 = gp_ipp_info.evaluate_cov_trace()
        # print(f"cov1 is {cov1}")
        node_info, node_std = gp_ipp_info.update_node(self.node_coords[f"{agent_ID}"])

        for i in range(1, NUM_THREADS + 1):
            if agent_position[f"{i}"] != []:
                if i != agent_ID:
                    for j, sample in enumerate(agent_position[f"{i}"]):
                        observed_value = np.array([0])
                        gp_ipp_info.add_observed_point(sample, observed_value)

        gp_ipp_info.update_gp()
        _, node_std = gp_ipp_info.update_node(self.node_coords[f"{agent_ID}"])
        # cov2 = gp_ipp_info.evaluate_cov_trace()
        # print(f"cov 2 is {cov2}")
        return node_info, node_std

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
            self.sample = np.array(self.sample)
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

    def plot(self, gaussian_mean, gaussian_cov, sampling_end_nodes,
             route, n, path, all_samples, sample_numbers, remain_budget, agent_ID, agent_position, CMAES_route=False):
        # Plotting path
        plt.switch_backend('agg')
        print("plot start")
        for i in range(1, NUM_THREADS + 1):
            if all_samples[f"{i}"] != []:
                for j, sample in enumerate(all_samples[f"{i}"]):
                    if j < sample_numbers:
                        observed_value = self.underlying_distribution.distribution_function(
                            sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)

                        # print(f"sample is {sample}", f"observed value is {observed_value}")
                        self.gp_ipp.add_observed_point(sample, observed_value)
        # print(f"observation points are {len(self.gp_ipp.observed_points)}")
        # print(f"observation points are {self.gp_ipp.observed_points}")
        ground_truth = self.get_ground_truth()
        self.gp_ipp.update_gp()
        self.gp_ipp.plot(ground_truth)

        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)

        # plot the others route
        for agent_ID in range(1, NUM_THREADS + 1):
            if agent_position[f"{agent_ID}"] != []:
                for j, sample in enumerate(agent_position[f"{agent_ID}"]):
                    observed_value = np.array([0])
                    self.gp_ipp.add_observed_point(sample, observed_value)
        self.gp_ipp.update_gp()

        print(f"cov_trace in plot is {cov_trace}", "\n")
        self.gp_ipp.plot_std()

        plt.subplot(2, 3, 1)
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        plt.scatter(self.node_coords[f"{agent_ID}"][1][0], self.node_coords[f"{agent_ID}"][1][1], c='r', marker='*',
                    s=15 ** 2)

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

        plt.subplot(2, 3, 5)
        plt.title('Interesting area')
        x = self.high_info_area[:, 0]
        y = self.high_info_area[:, 1]
        plt.hist2d(x, y, bins=30, vmin=0, vmax=1)

        # plot the intent map
        # plt.subplot(2, 3, 3)
        M = 1000  # sample numbers in gaussian distribution
        gaussian_value = np.zeros((M, M))
        for i in range(1, NUM_THREADS + 1):
            if gaussian_mean[f"{i}"] != [] and i != agent_ID:
                Gaussian = multivariate_normal(mean=gaussian_mean[f"{i}"], cov=gaussian_cov[f"{i}"])
                X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
                d = np.dstack([X, Y])
                Z = Gaussian.pdf(d).reshape(M, M) / 124.8

                gaussian_value += Z

        # print(gaussian_value.shape)
        gaussian_value = gaussian_value
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title('intent map')
        X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
        levels = [0.01 * i for i in range(101)]

        # plt.contourf(X, Y, gaussian_value, levels, cmap=cm.jet)
        # plt.colorbar()

        for i in range(1, NUM_THREADS + 1):
            if i != agent_ID and gaussian_mean[f"{i}"] != []:
                plt.scatter(gaussian_mean[f"{i}"][0], gaussian_mean[f"{i}"][1], c=colorlist[i - 1], marker='*',
                            s=15 ** 2)
                for j in range(SAMPLING_TIMES):
                    plt.scatter(sampling_end_nodes[f"{i}"][j][0], sampling_end_nodes[f"{i}"][j][1], c=colorlist[i - 1],
                                marker='o', s=8 ** 2)
        plt.suptitle('Cov trace: {:.4g}  remain_budget: {:.4g}'.format(cov_trace, remain_budget))
        # plt.tight_layout()
        plt.savefig('{}/{}.png'.format(path, n), dpi=150)
        plt.cla()
        plt.close("all")

    def plot_each(self, gaussian_mean, gaussian_cov, sampling_end_nodes,
                  route, n, path, all_samples, sample_numbers, remain_budget, agent_ID, CMAES_route=False):
        # Plotting path
        plt.switch_backend('agg')
        print("plot start")
        for i in range(1, NUM_THREADS + 1):
            if all_samples[f"{i}"] != []:
                for j, sample in enumerate(all_samples[f"{i}"]):
                    if j < sample_numbers:
                        observed_value = self.underlying_distribution.distribution_function(
                            sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)

                        # print(f"sample is {sample}", f"observed value is {observed_value}")
                        self.gp_ipp.add_observed_point(sample, observed_value)
        # print(f"observation points are {len(self.gp_ipp.observed_points)}")
        # print(f"observation points are {self.gp_ipp.observed_points}")
        ground_truth = self.get_ground_truth()
        self.gp_ipp.update_gp()
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        print(f"cov_trace in plot is {cov_trace}", "\n")

        self.gp_ipp.plot_each(ground_truth, n, path)

        plt.figure(1)
        plt.axis('off')
        # plt.subplot(1,3,1)
        colorlist = ['black', 'darkred', 'darkolivegreen', "purple", "gold"]
        plt.scatter(self.node_coords[f"{agent_ID}"][1][0], self.node_coords[f"{agent_ID}"][1][1], c='r', marker='*',
                    s=15 ** 2)

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
        plt.savefig('{}/Predict mean_{}.eps'.format(path, n), dpi=300)

        plt.figure(4)
        plt.axis('off')
        x = self.high_info_area[:, 0]
        y = self.high_info_area[:, 1]
        plt.hist2d(x, y, bins=30, vmin=0, vmax=1)
        plt.savefig('{}/high_interest_area_{}.eps'.format(path, n), dpi=300)

        # plot the intent map
        plt.figure(5)
        plt.axis('off')
        M = 1000  # sample numbers in gaussian distribution
        gaussian_value = np.zeros((M, M))
        for i in range(1, NUM_THREADS + 1):
            if gaussian_mean[f"{i}"] != [] and i != agent_ID:
                Gaussian = multivariate_normal(mean=gaussian_mean[f"{i}"], cov=gaussian_cov[f"{i}"])
                X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
                d = np.dstack([X, Y])
                Z = Gaussian.pdf(d).reshape(M, M)

                gaussian_value += Z

        # print(gaussian_value.shape)
        gaussian_value = gaussian_value / 124.8
        plt.xlabel("X")
        plt.ylabel("Y")
        X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
        levels = [0.01 * i for i in range(101)]

        plt.contourf(X, Y, gaussian_value, levels, cmap=cm.jet)
        plt.colorbar()

        for i in range(1, NUM_THREADS + 1):
            if i != agent_ID and gaussian_mean[f"{i}"] != []:
                plt.scatter(gaussian_mean[f"{i}"][0], gaussian_mean[f"{i}"][1], c=colorlist[i - 1], marker='*',
                            s=15 ** 2)
                for j in range(len(sampling_end_nodes[f"{i}"])):
                    plt.scatter(sampling_end_nodes[f"{i}"][j][0], sampling_end_nodes[f"{i}"][j][1], c=colorlist[i - 1],
                                marker='o', s=8 ** 2)
        # plt.tight_layout()
        plt.subplots_adjust(top=1, bottom=0.02, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('{}/intent map_{}.pdf'.format(path, n), dpi=300)
        plt.cla()
        plt.close("all")


if __name__ == '__main__':
    env = Env(100, save_image=True)
    nodes, graph, info, std, budget = env.reset()
    print(nodes)
    print(graph)
