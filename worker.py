import copy
import os
from sklearn.cluster import KMeans
import imageio
import numpy as np
import torch
from env import Env
from AttentionNet import AttentionNet
from parameter import *
import scipy.signal as signal
import math
import time
from threading import Lock
from scipy.stats import multivariate_normal
import random

overall_reward = []
overall_delta_cov_trace = []
for i in range(0, NUM_THREADS):
    overall_reward.append(0)
    overall_delta_cov_trace.append(0)
# store all the agents route for communication
agent_route = dict()
agent_position = dict()
for i in range(1, NUM_THREADS + 1):
    agent_route[f"{i}"] = []
    agent_position[f"{i}"] = []

# store all the samples position for update the GP when making decision
all_samples = dict()
for i in range(1, NUM_THREADS + 1):
    all_samples[f"{i}"] = []

all_done = dict()
for i in range(1, NUM_THREADS + 1):
    all_done[f"{i}"] = False

all_reset = dict()
for i in range(1, NUM_THREADS + 1):
    all_reset[f"{i}"] = False

sampling_end_nodes = dict()
gaussian_mean = dict()
gaussian_cov = dict()
for i in range(1, NUM_THREADS + 1):
    sampling_end_nodes[f"{i}"] = []
    gaussian_mean[f"{i}"] = []
    gaussian_cov[f"{i}"] = []

# show when the sampling is finished
sampling_finish = []
for i in range(NUM_THREADS + 1):
    sampling_finish.append(True)
episode = 0
lock = Lock()
agent_step = 0
first_sampling_agent = None
cov_trace = 900
first_step_seq = []


def calculate_intent_difference_KL(cov, cov_before, mean, mean_before):
    intent_difference_KL_1 = 0
    intent_difference_KL_2 = 0
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    Gaussian_before = multivariate_normal(mean=mean_before, cov=cov_before)
    M = 30
    X, Y = np.meshgrid(np.linspace(0, 1, M), np.linspace(0, 1, M))
    d = np.dstack([X, Y])
    Z1 = Gaussian.pdf(d).reshape(M, M)
    Z2 = Gaussian_before.pdf(d).reshape(M, M)
    Z1 = Z1 / np.max(Z1)
    Z2 = Z2 / np.max(Z2)

    for i in range(M):
        for j in range(M):
            if Z1[i][j] < 1e-5:
                Z1[i][j] = 1e-5
            if Z2[i][j] < 1e-5:
                Z2[i][j] = 1e-5
            intent_difference_KL_1 += Z1[i][j] * np.log(Z1[i][j] / Z2[i][j])
            intent_difference_KL_2 += Z2[i][j] * np.log(Z2[i][j] / Z1[i][j])
            # if Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]) > 5:
            #     print(Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]))
            # #     print(Z1[i][j] * np.log(Z1[i][j] / Z2[i][j]), Z2[i][j] * np.log(Z2[i][j] / Z1[i][j]))
            #     print(Z1[i][j], Z2[i][j], np.log(Z1[i][j] / Z2[i][j]), "\n")
    intent_difference_KL = [intent_difference_KL_1, intent_difference_KL_2]

    return intent_difference_KL


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


class Worker:
    def __init__(self, metaAgentID, agent_ID, localNetwork, global_step, env,
                 sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):
        global agent_route, all_samples, all_done, all_reset, sampling_end_nodes, agent_step, agent_position
        self.node_coords = []
        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size
        self.agent_ID = agent_ID
        self.env = env
        # self.local_net = AttentionNet(2, 128, device=self.device)
        # self.local_net.to(device)
        self.local_net = localNetwork
        self.perf_metrics = None
        self.if_done = False
        self.curren_step = 0
        self.experience = None

    def cal_agent_distance(self, agent_ID):
        agent_pos = agent_route[f"{agent_ID}"]
        distance = 0
        for i, t in enumerate(agent_pos):
            if i > 0:
                distance += cal_distance(self.env.node_coords[f"{agent_ID}"][t],
                                         self.env.node_coords[f"{agent_ID}"][agent_pos[i - 1]])
        # print(f"coordinate is {self.env.node_coords[0]}")
        return distance

    def cal_relative_coord(self, another_node_index, current_node_index):
        x = self.node_coords[another_node_index][0] - self.node_coords[current_node_index][0]
        y = self.node_coords[another_node_index][1] - self.node_coords[current_node_index][1]
        return [x, y]

    def cal_agent_input(self, another_node_index, current_node_index, self_ID, others_ID):
        x = self.env.node_coords[f"{others_ID}"][another_node_index][0] - \
            self.env.node_coords[f"{self_ID}"][current_node_index][0]
        y = self.env.node_coords[f"{others_ID}"][another_node_index][1] - \
            self.env.node_coords[f"{self_ID}"][current_node_index][1]
        return [x, y]

    def run_episode(self, currEpisode):
        global episode, agent_step, all_samples, cov_trace, gaussian_mean, gaussian_cov, first_step_seq, node_info, node_std, overall_reward, overall_delta_cov_trace
        # reset the agent route and all samples in new episode
        if episode != currEpisode:
            episode = currEpisode
            for i in range(1, NUM_THREADS + 1):
                agent_route[f"{i}"] = []
                agent_position[f"{i}"] = []
            for i in range(1, NUM_THREADS + 1):
                all_samples[f"{i}"] = []
                sampling_end_nodes[f"{i}"] = []
                gaussian_mean[f"{i}"] = []
                gaussian_cov[f"{i}"] = []

        for i in range(1, NUM_THREADS + 1):
            all_done[f"{i}"] = False

        overall_reward = []
        overall_delta_cov_trace = []
        for i in range(0, NUM_THREADS + 1):
            overall_reward.append(0)
            overall_delta_cov_trace.append(0)

        for i in range(NUM_THREADS + 1):
            sampling_finish.append(True)

        first_step_seq = []
        for i in range(NUM_THREADS + 1):
            first_step_seq.append(False)
        first_step_seq[0] = True

        episode_buffer = []
        dist_residual = 0
        agent_step = 0
        perf_metrics = dict()
        cov_trace = 900
        done = False
        time.sleep(0.5)
        mean_before = None
        cov_before = None
        intent_difference = []
        for i in range(14):
            episode_buffer.append([])
        node_coords, graph, node_info, node_std, budget = self.env.reset(agent_ID=self.agent_ID)
        # make sure that all the agents finish the reset process
        all_reset[f"{self.agent_ID}"] = True

        for i in range(1, NUM_THREADS + 1):
            while not all_reset[f"{i}"]:
                time.sleep(0.5)
                pass

        # print(f"node_coords is {node_coords.shape}")
        self.node_coords = node_coords
        # print(f"agent_id is {self.agent_ID}", self.node_coords[32])
        # reset initial each agent's route in a dictionary(eg. agent 7 has the initial position which is node 7)
        # agent_ID = initial node index for this agent
        flag = 0
        for t in range(1, NUM_THREADS + 1):
            if agent_route[f"{t}"] != []:
                flag = 1
        if flag == 0:
            for t in range(1, NUM_THREADS + 1):
                agent_route[f"{t}"].append(1)
                agent_position[f"{t}"].append(self.node_coords[1])

        n_nodes = node_coords.shape[0]
        current_index = torch.tensor([1]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        next_node_index = None
        action_index = None
        value = None
        reward = None
        ground_truth_sampling = None
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1, agent_ID=self.agent_ID)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 1)

        graph = list(graph.values())
        edge_inputs = []

        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        try:
            pos_encoding = self.calculate_position_embedding(edge_inputs)

        except:
            while len(edge_inputs) != self.sample_size + 2:
                edge_inputs.append(edge_inputs[0])
            print(f"bug occurs in edge inputs")

            pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)  # (1, sample_size+2, 32)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, k_size)

        relative_node_coords = []

        LSTM_h = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

        # choose the first step for each agent
        while not first_step_seq[self.agent_ID - 1]:
            # print(f"sequence is {first_step_seq}")
            time.sleep(0.5)
            if first_step_seq[self.agent_ID - 1]:
                break
        # print(f"self.agent_ID is {self.agent_ID}")
        agent_input = [index for index in range(1, NUM_THREADS + 1)]
        for agent_ID in range(1, NUM_THREADS + 1):
            agent_input[agent_ID - 1] = self.cal_agent_input(agent_route[f"{agent_ID}"][-1],
                                                             current_index.item(), self_ID=self.agent_ID,
                                                             others_ID=agent_ID)

        # print(f"mean is {gaussian_mean}")
        # print(f"cov is {gaussian_cov}")
        agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)
        # get the intent input
        intent_input = self.env.construct_intent_map(gaussian_mean, gaussian_cov, self.agent_ID, self.node_coords)
        if max(intent_input)[0] != 0:
            intent_input = intent_input / np.max(intent_input)
        else:
            pass
        # get the node input before making the policy to make sure that information is updating
        node_info, node_std = self.env.get_node_information(all_samples, 0, self.agent_ID, agent_position)
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes, 1))

        for i in range(0, self.sample_size + 2):
            relative_node_coords.append([])
        for i in range(0, self.sample_size + 2):
            relative_node_coords[i] = self.cal_relative_coord(i, current_index)

        if USE_INTENT:
            node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs, intent_input),
                                         axis=1)

        else:
            node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs), axis=1)

        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 5)

        route = agent_route[f"{self.agent_ID}"]
        remain_budget = self.env.budget[self.agent_ID]

        # sampling for the first step
        current_agent_dis = self.cal_agent_distance(self.agent_ID)
        sample_numbers = current_agent_dis // self.sample_length
        reward_first_step = []
        edge_inputs_first_step = []
        budget_inputs_first_step = []
        next_index_first_step = []
        LSTM_h_first_step = []
        LSTM_c_first_step = []
        pos_encoding_first_step = []
        agent_input_first_step = []
        dist_residual_first_step = []
        mask_first_step = []
        route_first_step = []
        all_samples_first_step = []
        relative_node_coords_first_step = []
        remain_budget_first_step = []
        cov_trace_first_step = []
        action_index_first_step = []
        value_first_step = []
        cov_trace_sampling_final = 900
        best_trajectory = 0
        cov_trace_before = cov_trace

        # Sampling process
        if SAMPLING and not done:
            end_nodes = []
            for sampling_time in range(SAMPLING_TIMES):
                node_inputs_sampling = copy.deepcopy(node_inputs)
                edge_inputs_sampling = copy.deepcopy(edge_inputs)
                budget_inputs_sampling = copy.deepcopy(budget_inputs)
                current_index_sampling = copy.deepcopy(current_index)
                LSTM_h_sampling = copy.deepcopy(LSTM_h)
                LSTM_c_sampling = copy.deepcopy(LSTM_c)
                pos_encoding_sampling = copy.deepcopy(pos_encoding)
                agent_input_sampling = copy.deepcopy(agent_input)
                dist_residual_sampling = copy.deepcopy(dist_residual)
                mask_sampling = copy.deepcopy(mask)
                route_sampling = copy.deepcopy(route)
                all_samples_sampling = copy.deepcopy(all_samples)
                relative_node_coords_sampling = copy.deepcopy(relative_node_coords)
                remain_budget_before_sampling = copy.deepcopy(remain_budget)
                sampling_done = copy.deepcopy(done)
                cov_trace_sampling = copy.deepcopy(cov_trace)
                self.env.budget[self.agent_ID] = remain_budget_before_sampling
                intent_input_sampling = copy.deepcopy(intent_input)

                for sampling_step in range(SAMPLING_STEPS):
                    with torch.no_grad():
                        logp_list_sampling, value_sampling, LSTM_h_sampling, LSTM_c_sampling = self.local_net(
                            node_inputs_sampling, edge_inputs_sampling,
                            budget_inputs_sampling,
                            current_index_sampling, LSTM_h_sampling,
                            LSTM_c_sampling, pos_encoding_sampling, mask_sampling)

                    # next_node (1), logp_list (1, 10), value (1,1,1)
                    if self.greedy:
                        action_index_sampling = torch.argmax(logp_list_sampling, dim=1).long()
                    else:
                        action_index_sampling = torch.multinomial(logp_list_sampling.exp(), 1).long().squeeze(1)
                    pass
                    next_node_index_sampling = edge_inputs_sampling[:, current_index_sampling.item(),
                                               action_index_sampling.item()]
                    if sampling_step == 0:
                        next_index_first_step.append(copy.deepcopy(next_node_index_sampling))
                        action_index_first_step.append(copy.deepcopy(action_index_sampling))

                    sample_coords_sampling, remain_budget_sampling, dist_residual_sampling = self.env.observed_positions(
                        current_index_sampling.item(),
                        next_node_index_sampling.item(),
                        self.sample_length,
                        agent_ID=self.agent_ID,
                        dist_residual=dist_residual_sampling)

                    for sample in sample_coords_sampling:
                        all_samples_sampling[f"{self.agent_ID}"].append(sample)

                    reward_sampling, sampling_done, ground_truth_sampling, cov_trace_sampling = self.env.step(
                        cov_trace_sampling,
                        route_sampling,
                        next_node_index_sampling.item(),
                        sampling_done,
                        agent_ID=self.agent_ID,
                        all_samples_sampling=all_samples_sampling,
                        sampling=True)
                    node_info_sampling, node_std_sampling = self.env.get_node_information(all_samples_sampling,
                                                                                          sample_numbers, self.agent_ID,
                                                                                          agent_position)
                    route_sampling.append(next_node_index_sampling.item())
                    # print(f"route sampling is {route_sampling}", f"reward is {reward_sampling}",
                    #       f"cov_trace is {cov_trace_sampling}")
                    # decide the destination in sampling process
                    budget_inputs_sampling = self.calc_estimate_budget(remain_budget_sampling,
                                                                       current_idx=next_node_index_sampling.item(),
                                                                       agent_ID=self.agent_ID)
                    budget_inputs_sampling = torch.FloatTensor(budget_inputs_sampling).unsqueeze(0).to(self.device)
                    next_edge_sampling = torch.gather(edge_inputs_sampling, 1,
                                                      next_node_index_sampling.repeat(1, 1, K_SIZE))
                    next_edge_sampling = next_edge_sampling.permute(0, 2, 1)
                    connected_nodes_budget_sampling = torch.gather(budget_inputs_sampling, 1, next_edge_sampling)
                    connected_nodes_budget_sampling = connected_nodes_budget_sampling.squeeze(0).squeeze(0)
                    connected_nodes_budget_sampling = connected_nodes_budget_sampling.tolist()
                    sampling_done = True
                    for i in connected_nodes_budget_sampling[1:]:
                        if i[0] > 0:
                            sampling_done = False

                    current_index_sampling = next_node_index_sampling.unsqueeze(0).unsqueeze(0)
                    node_info_inputs_sampling = node_info_sampling.reshape(n_nodes, 1)
                    node_std_inputs_sampling = node_std_sampling.reshape(n_nodes, 1)
                    # calculate the relative coordinates of all nodes
                    for t in range(0, self.sample_size + 2):
                        relative_node_coords_sampling[t] = []
                    for t in range(0, self.sample_size + 2):
                        relative_node_coords_sampling[t] = self.cal_relative_coord(t, current_index_sampling)

                    node_inputs_sampling = np.concatenate(
                        (relative_node_coords_sampling, node_info_inputs_sampling, node_std_inputs_sampling,
                         intent_input_sampling),
                        axis=1)
                    node_inputs_sampling = torch.FloatTensor(node_inputs_sampling).unsqueeze(0).to(
                        self.device)  # (1, sample_size+2, 4)

                    # mask last five node
                    mask_sampling = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(
                        self.device)

                    if sampling_step == 0:
                        edge_inputs_first_step.append(copy.deepcopy(edge_inputs_sampling))
                        budget_inputs_first_step.append(copy.deepcopy(budget_inputs_sampling))
                        value_first_step.append(copy.deepcopy(value_sampling))
                        LSTM_h_first_step.append(copy.deepcopy(LSTM_h_sampling))
                        LSTM_c_first_step.append(copy.deepcopy(LSTM_c_sampling))
                        pos_encoding_first_step.append(copy.deepcopy(pos_encoding_sampling))
                        agent_input_first_step.append(copy.deepcopy(agent_input_sampling))
                        dist_residual_first_step.append(copy.deepcopy(dist_residual_sampling))
                        mask_first_step.append(copy.deepcopy(mask_sampling))
                        route_first_step.append(copy.deepcopy(route_sampling))
                        all_samples_first_step.append(copy.deepcopy(all_samples_sampling))
                        relative_node_coords_first_step.append(copy.deepcopy(relative_node_coords_sampling))
                        remain_budget_first_step.append(copy.deepcopy(remain_budget_sampling))
                        reward_first_step.append(reward_sampling)
                        cov_trace_first_step.append(cov_trace_sampling)

                    end_nodes.append(self.node_coords[next_node_index_sampling.item()])
                    if sampling_done or sampling_step == SAMPLING_STEPS - 1:
                        if cov_trace_sampling < cov_trace_sampling_final:
                            best_trajectory = sampling_time
                            cov_trace_sampling_final = cov_trace_sampling
                        # print(f"end_nodes are {end_nodes}", f"sampling step is {sampling_step}")
                        self.env.budget[self.agent_ID] = remain_budget_before_sampling
                        # route_sampling.append(next_node_index_sampling.item())
                        break

            sampling_end_nodes[f"{self.agent_ID}"] = end_nodes
            estimator = KMeans(n_clusters=1)
            data = end_nodes
            estimator.fit(data)
            centroids = estimator.cluster_centers_
            mean = centroids[0]
            cov = np.cov(data, rowvar=False) + np.array([[1 / 28 ** 2, 0], [0, 1 / 28 ** 2]])
            gaussian_mean[f"{self.agent_ID}"] = mean
            mean_before = mean
            gaussian_cov[f"{self.agent_ID}"] = cov
            cov_before = cov

            if not BEST_TRAJ:
                best_trajectory = np.random.randint(0, SAMPLING_TIMES)
            # copy the input of first step
            edge_inputs = copy.deepcopy(edge_inputs_first_step[best_trajectory])
            budget_inputs = copy.deepcopy(budget_inputs_first_step[best_trajectory])
            next_node_index = copy.deepcopy(next_index_first_step[best_trajectory])
            action_index = copy.deepcopy(action_index_first_step[best_trajectory])
            LSTM_h = copy.deepcopy(LSTM_h_first_step[best_trajectory])
            LSTM_c = copy.deepcopy(LSTM_c_first_step[best_trajectory])
            value = copy.deepcopy(value_first_step[best_trajectory])
            pos_encoding = copy.deepcopy(pos_encoding_first_step[best_trajectory])
            dist_residual = copy.deepcopy(dist_residual_first_step[best_trajectory])
            route = copy.deepcopy(route_first_step[best_trajectory])
            # print(f"route is {route}")
            all_samples = copy.deepcopy(all_samples_first_step[best_trajectory])
            relative_node_coords = copy.deepcopy(relative_node_coords_first_step[best_trajectory])

            self.env.budget[self.agent_ID] = copy.deepcopy(remain_budget_first_step[best_trajectory])
            remain_budget = copy.deepcopy(remain_budget_first_step[best_trajectory])
            reward = copy.deepcopy(reward_first_step[best_trajectory])
            first_step_seq[self.agent_ID] = True
            cov_trace = copy.deepcopy(cov_trace_first_step[best_trajectory])
            ground_truth = ground_truth_sampling
            overall_delta_cov_trace[self.agent_ID - 1] += cov_trace_before - cov_trace
            overall_reward[self.agent_ID - 1] += reward

        agent_route[f"{self.agent_ID}"].append(next_node_index.item())
        agent_position[f"{self.agent_ID}"][0] = self.node_coords[next_node_index.item()]
        current_index = next_node_index.unsqueeze(0).unsqueeze(0)

        # print("stuck here 2")
        for agent_ID in range(1, NUM_THREADS + 1):
            while len(agent_route[f"{agent_ID}"]) < 2:
                time.sleep(0.5)
                if len(agent_route[f"{agent_ID}"]) == 2:
                    break

        time.sleep(1 * self.agent_ID)
        # print(f"agent_id is {self.agent_ID}", self.node_coords[32])
        # print("stuck here 3")
        for step_i in range(steps):

            # print(f"self.agent_ID is {self.agent_ID}")
            current_agent_dis = self.cal_agent_distance(self.agent_ID)
            # print(f"current distance is {current_agent_dis}")
            sample_numbers = current_agent_dis // self.sample_length
            # get the relative agent position with current agent
            # if this agent has a longer distance from the start node than other agents, then wait for other agents

            for agent_ID in range(1, NUM_THREADS + 1):
                if agent_ID != self.agent_ID:
                    if not all_done[f"{agent_ID}"]:
                        if SAMPLING:
                            while current_agent_dis >= self.cal_agent_distance(agent_ID):
                                if current_agent_dis == self.cal_agent_distance(agent_ID):
                                    # print(self.agent_ID)
                                    if self.agent_ID < agent_ID:
                                        break
                                # print(f"self.agentID is {self.agent_ID}", f"stuck is here in distance")
                                time.sleep(0.5)
                                # print("stuck in here")
                                if all_done[f"{agent_ID}"]:
                                    break
                    if not SAMPLING:
                        while current_agent_dis > self.cal_agent_distance(agent_ID):
                            # print(f"self.agentID is {self.agent_ID}", f"stuck is here in distance")
                            time.sleep(0.5)
                            if all_done[f"{agent_ID}"]:
                                break
                else:
                    pass
            sampling_finish[self.agent_ID] = False
            agent_step += 1
            self.curren_step += 1

            agent_input = [index for index in range(1, NUM_THREADS + 1)]
            for agent_ID in range(1, NUM_THREADS + 1):
                agent_input[agent_ID - 1] = self.cal_agent_input(agent_route[f"{agent_ID}"][-1],
                                                                 current_index.item(), self_ID=self.agent_ID,
                                                                 others_ID=agent_ID)
            # agent_input[agent_ID - 1] = np.random.rand(2)
            agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)
            intent_input = self.env.construct_intent_map(gaussian_mean, gaussian_cov, self.agent_ID, self.node_coords)
            intent_input = intent_input / np.max(intent_input)

            # print(f"agent route is {agent_route}")
            # print(f"agent_input is {agent_input}")
            node_info, node_std = self.env.get_node_information(all_samples, sample_numbers, self.agent_ID, agent_position)
            node_info_inputs = node_info.reshape((n_nodes, 1))
            node_std_inputs = node_std.reshape((n_nodes, 1))
            relative_node_coords = []
            for i in range(0, self.sample_size + 2):
                relative_node_coords.append([])
            for i in range(0, self.sample_size + 2):
                relative_node_coords[i] = self.cal_relative_coord(i, current_index)

            if USE_INTENT:
                node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs, intent_input),
                                             axis=1)
            else:
                node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs), axis=1)

            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 5)

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[8] += budget_inputs
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding
            episode_buffer[13] += agent_input

            # store the input of first step
            cov_trace_before = cov_trace
            reward_first_step = []
            edge_inputs_first_step = []
            budget_inputs_first_step = []
            next_index_first_step = []
            LSTM_h_first_step = []
            LSTM_c_first_step = []
            pos_encoding_first_step = []
            agent_input_first_step = []
            dist_residual_first_step = []
            mask_first_step = []
            route_first_step = []
            all_samples_first_step = []
            relative_node_coords_first_step = []
            remain_budget_first_step = []
            cov_trace_first_step = []
            action_index_first_step = []
            value_first_step = []
            best_trajectory = 0
            cov_trace_sampling_final = 900

            # Sampling process
            if SAMPLING and not done:
                end_nodes = []
                for sampling_time in range(SAMPLING_TIMES):
                    node_inputs_sampling = copy.deepcopy(node_inputs)
                    edge_inputs_sampling = copy.deepcopy(edge_inputs)
                    budget_inputs_sampling = copy.deepcopy(budget_inputs)
                    current_index_sampling = copy.deepcopy(current_index)
                    LSTM_h_sampling = copy.deepcopy(LSTM_h)
                    LSTM_c_sampling = copy.deepcopy(LSTM_c)
                    pos_encoding_sampling = copy.deepcopy(pos_encoding)
                    agent_input_sampling = copy.deepcopy(agent_input)
                    dist_residual_sampling = copy.deepcopy(dist_residual)
                    mask_sampling = copy.deepcopy(mask)
                    route_sampling = copy.deepcopy(route)
                    all_samples_sampling = copy.deepcopy(all_samples)
                    relative_node_coords_sampling = copy.deepcopy(relative_node_coords)
                    remain_budget_before_sampling = copy.deepcopy(remain_budget)
                    sampling_done = copy.deepcopy(done)
                    cov_trace_sampling = copy.deepcopy(cov_trace)
                    self.env.budget[self.agent_ID] = remain_budget_before_sampling
                    intent_input_sampling = copy.deepcopy(intent_input)

                    for sampling_step in range(SAMPLING_STEPS):
                        # print(f"agent input is {agent_input_sampling}")
                        with torch.no_grad():
                            logp_list_sampling, value_sampling, LSTM_h_sampling, LSTM_c_sampling = self.local_net(
                                node_inputs_sampling, edge_inputs_sampling,
                                budget_inputs_sampling,
                                current_index_sampling, LSTM_h_sampling,
                                LSTM_c_sampling, pos_encoding_sampling, mask_sampling)
                        # print(f"policy is {torch.exp(logp_list_sampling)}")
                        # print(f"policy max is {torch.max(torch.exp(logp_list_sampling), dim=1)}")
                        # next_node (1), logp_list (1, 10), value (1,1,1)
                        if self.greedy:
                            action_index_sampling = torch.argmax(logp_list_sampling, dim=1).long()
                        else:
                            action_index_sampling = torch.multinomial(logp_list_sampling.exp(), 1).long().squeeze(1)
                        pass
                        next_node_index_sampling = edge_inputs_sampling[:, current_index_sampling.item(),
                                                   action_index_sampling.item()]

                        if sampling_step == 0:
                            next_index_first_step.append(copy.deepcopy(next_node_index_sampling))
                            action_index_first_step.append(copy.deepcopy(action_index_sampling))

                        sample_coords_sampling, remain_budget_sampling, dist_residual_sampling = self.env.observed_positions(
                            current_index_sampling.item(),
                            next_node_index_sampling.item(),
                            self.sample_length,
                            agent_ID=self.agent_ID,
                            dist_residual=dist_residual_sampling)

                        for sample in sample_coords_sampling:
                            all_samples_sampling[f"{self.agent_ID}"].append(sample)
                        # print(f"all_samples_sampling is {all_samples_sampling}")
                        # print(f"cov_trace_sampling is {cov_trace_sampling}")
                        reward_sampling, sampling_done, ground_truth_sampling, cov_trace_sampling = self.env.step(
                            cov_trace_sampling,
                            route_sampling,
                            next_node_index_sampling.item(),
                            sampling_done,
                            agent_ID=self.agent_ID,
                            all_samples_sampling=all_samples_sampling,
                            sampling=True)
                        node_info_sampling, node_std_sampling = self.env.get_node_information(all_samples_sampling,
                                                                                              sample_numbers,
                                                                                              self.agent_ID, agent_position)
                        route_sampling.append(next_node_index_sampling.item())

                        # print(f"cov_trace_sampling is {cov_trace_sampling}")

                        # decide the destination in sampling process
                        budget_inputs_sampling = self.calc_estimate_budget(remain_budget_sampling,
                                                                           current_idx=next_node_index_sampling.item(),
                                                                           agent_ID=self.agent_ID)
                        budget_inputs_sampling = torch.FloatTensor(budget_inputs_sampling).unsqueeze(0).to(self.device)
                        next_edge_sampling = torch.gather(edge_inputs_sampling, 1,
                                                          next_node_index_sampling.repeat(1, 1, K_SIZE))
                        next_edge_sampling = next_edge_sampling.permute(0, 2, 1)
                        connected_nodes_budget_sampling = torch.gather(budget_inputs_sampling, 1, next_edge_sampling)
                        connected_nodes_budget_sampling = connected_nodes_budget_sampling.squeeze(0).squeeze(0)
                        connected_nodes_budget_sampling = connected_nodes_budget_sampling.tolist()
                        sampling_done = True
                        for i in connected_nodes_budget_sampling[1:]:
                            if i[0] > 0:
                                sampling_done = False
                        # print(f"sampling done is {sampling_done}", f"agent_id is {self.agent_ID}")
                        current_index_sampling = next_node_index_sampling.unsqueeze(0).unsqueeze(0)
                        node_info_inputs_sampling = node_info_sampling.reshape(n_nodes, 1)
                        node_std_inputs_sampling = node_std_sampling.reshape(n_nodes, 1)
                        # calculate the relative coordinates of all nodes
                        for t in range(0, self.sample_size + 2):
                            relative_node_coords_sampling[t] = []
                        for t in range(0, self.sample_size + 2):
                            relative_node_coords_sampling[t] = self.cal_relative_coord(t, current_index_sampling)

                        node_inputs_sampling = np.concatenate(
                            (relative_node_coords_sampling, node_info_inputs_sampling, node_std_inputs_sampling,
                             intent_input_sampling),
                            axis=1)
                        node_inputs_sampling = torch.FloatTensor(node_inputs_sampling).unsqueeze(0).to(
                            self.device)  # (1, sample_size+2, 4)

                        # mask last five node
                        mask_sampling = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(
                            self.device)

                        if sampling_step == 0:
                            edge_inputs_first_step.append(copy.deepcopy(edge_inputs_sampling))
                            budget_inputs_first_step.append(copy.deepcopy(budget_inputs_sampling))
                            value_first_step.append(copy.deepcopy(value_sampling))
                            LSTM_h_first_step.append(copy.deepcopy(LSTM_h_sampling))
                            LSTM_c_first_step.append(copy.deepcopy(LSTM_c_sampling))
                            pos_encoding_first_step.append(copy.deepcopy(pos_encoding_sampling))
                            agent_input_first_step.append(copy.deepcopy(agent_input_sampling))
                            dist_residual_first_step.append(copy.deepcopy(dist_residual_sampling))
                            mask_first_step.append(copy.deepcopy(mask_sampling))
                            route_first_step.append(copy.deepcopy(route_sampling))
                            all_samples_first_step.append(copy.deepcopy(all_samples_sampling))
                            relative_node_coords_first_step.append(copy.deepcopy(relative_node_coords_sampling))
                            remain_budget_first_step.append(copy.deepcopy(remain_budget_sampling))
                            reward_first_step.append(reward_sampling)
                            cov_trace_first_step.append(cov_trace_sampling)

                        end_nodes.append(self.node_coords[next_node_index_sampling.item()])

                        if sampling_done or sampling_step == SAMPLING_STEPS - 1:
                            if cov_trace_sampling < cov_trace_sampling_final:
                                best_trajectory = sampling_time
                                cov_trace_sampling_final = cov_trace_sampling

                            # print(f"next_node_index_sampling is {next_node_index_sampling.item()}")
                            # print(f"end_nodes are {end_nodes}", f"sampling step is {sampling_step}")
                            self.env.budget[self.agent_ID] = remain_budget_before_sampling
                            # route_sampling.append(next_node_index_sampling.item())
                            # print(f"route sampling is {route_sampling}")
                            break

                sampling_end_nodes[f"{self.agent_ID}"] = end_nodes
                # print(f"len of end nodes are {len(end_nodes)}")
                estimator = KMeans(n_clusters=1)
                data = end_nodes
                estimator.fit(data)
                centroids = estimator.cluster_centers_
                mean = centroids[0]
                cov = np.cov(data, rowvar=False) + np.array([[1 / 28 ** 2, 0], [0, 1 / 28 ** 2]])
                # current_intent_difference_KL = calculate_intent_difference_KL(mean=mean, mean_before=mean_before,
                #                                                               cov=cov, cov_before=cov_before)
                gaussian_mean[f"{self.agent_ID}"] = mean
                gaussian_cov[f"{self.agent_ID}"] = cov
                current_intent_difference_KL = calculate_intent_difference_KL(mean=mean, mean_before=mean_before,
                                                                              cov=cov, cov_before=cov_before)
                # print(f"KL is {current_intent_difference_KL}")
                intent_difference.append(current_intent_difference_KL)
                # print(f"intent difference is {intent_difference_abs}")
                mean_before = mean
                cov_before = cov
                # print(f"gaussian mean is {gaussian_mean}", f"gaussian cov is {gaussian_cov}")
                # copy the input of first step

                if not BEST_TRAJ:
                    best_trajectory = np.random.randint(0, SAMPLING_TIMES)

                edge_inputs = copy.deepcopy(edge_inputs_first_step[best_trajectory])
                budget_inputs = copy.deepcopy(budget_inputs_first_step[best_trajectory])
                ground_truth = ground_truth_sampling
                next_node_index = copy.deepcopy(next_index_first_step[best_trajectory])
                action_index = copy.deepcopy(action_index_first_step[best_trajectory])
                LSTM_h = copy.deepcopy(LSTM_h_first_step[best_trajectory])
                LSTM_c = copy.deepcopy(LSTM_c_first_step[best_trajectory])
                value = copy.deepcopy(value_first_step[best_trajectory])
                pos_encoding = copy.deepcopy(pos_encoding_first_step[best_trajectory])
                dist_residual = copy.deepcopy(dist_residual_first_step[best_trajectory])
                route = copy.deepcopy(route_first_step[best_trajectory])
                all_samples = copy.deepcopy(all_samples_first_step[best_trajectory])
                relative_node_coords = copy.deepcopy(relative_node_coords_first_step[best_trajectory])
                # print(f"remain budget first step is {remain_budget_first_step}")
                self.env.budget[self.agent_ID] = copy.deepcopy(remain_budget_first_step[best_trajectory])
                remain_budget = copy.deepcopy(remain_budget_first_step[best_trajectory])
                reward = copy.deepcopy(reward_first_step[best_trajectory])
                sampling_finish[self.agent_ID] = True
                cov_trace = copy.deepcopy(cov_trace_first_step[best_trajectory])
                overall_delta_cov_trace[self.agent_ID - 1] += cov_trace_before - cov_trace
                overall_reward[self.agent_ID - 1] += reward
                # print(f"reward is {reward}", f"agentID is {self.agent_ID}", "\n")
                # print(f"all samples is {all_samples}", "\n")
                # print(f"sampling_end_nodes is {sampling_end_nodes}")
                # print(f"agent id is {self.agent_ID}")
                # print(f"route is {route}", "\n")

            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            # print(f"cov is {cov_trace}")
            # print(f"reward is {reward}", "\n")
            next_edge = torch.gather(edge_inputs, 1, next_node_index.repeat(1, 1, K_SIZE))
            next_edge = next_edge.permute(0, 2, 1)
            connected_nodes_budget = torch.gather(budget_inputs, 1, next_edge)
            connected_nodes_budget = connected_nodes_budget.squeeze(0).squeeze(0)
            connected_nodes_budget = connected_nodes_budget.tolist()
            done = True
            for i in connected_nodes_budget[1:]:
                if i[0] > 0:
                    done = False
            all_done[f"{self.agent_ID}"] = done

            # print(f"agent1 = {agent1}, agent2 = {agent2}, agent3 = {agent3}")
            if SAVE_IMAGE:
                print(f"agent ID is {self.agent_ID}")

                agent1 = len(agent_route["1"])
                agent2 = len(agent_route["2"])
                agent3 = len(agent_route["3"])
                n = agent1 + agent2 + agent3
                if n > 2:
                    self.gifs_path = '{}/{}'.format(gifs_path, self.global_step)
                    if self.save_image:
                        if not os.path.exists(self.gifs_path):
                            os.makedirs(self.gifs_path)
                        self.env.plot(gaussian_mean, gaussian_cov, sampling_end_nodes, agent_route, n,
                                           self.gifs_path,
                                           all_samples=all_samples, sample_numbers=sample_numbers,
                                           remain_budget=remain_budget, agent_ID=self.agent_ID)

                # store the next_node_index in a global variable
            agent_route[f"{self.agent_ID}"].append(next_node_index.item())

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)
            # print(f"reward is {reward}", f"agentID is {self.agent_ID}", "\n")
            agent_position[f"{self.agent_ID}"][0] = self.node_coords[next_node_index.item()]
            # print(f"agent position is {agent_position}")
            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                perf_metrics['remain_budget'] = remain_budget / budget
                # perf_metrics['collect_info'] = 1 - remain_info.sum()
                # perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                perf_metrics['RMSE'] = self.env.RMSE
                perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(ground_truth)
                perf_metrics['delta_cov_trace'] = 900 - cov_trace
                # print(perf_metrics['delta_cov_trace'])
                perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                perf_metrics['cov_trace'] = cov_trace
                perf_metrics['success_rate'] = True
                intent_difference_final = np.array(intent_difference)
                intent_difference_final = np.mean(intent_difference_final, axis=0)
                perf_metrics["intent_difference_KL"] = intent_difference_final


                break

        if not done:
            episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index,
                                                          LSTM_h, LSTM_c, pos_encoding, mask)
            episode_buffer[6].append(value.squeeze(0))
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(ground_truth)
            perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = cov_trace
            perf_metrics['success_rate'] = False

        # print(f'agent {self.agent_ID}: route is {route}')

        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])

        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward, dtype=object).reshape(-1)
        print(f"reward is {reward_plus}")
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)
        print(f"target v is {target_v}")
        for i in range(target_v.size()[0]):
            episode_buffer[7].append(target_v[i, :, :])

        '''
        # save gif
        if self.save_image:
            if self.greedy:
                from test_driver import result_path as path
            else:
                path = gifs_path
            self.make_gif(path, currEpisode)
        
        if self.save_image:
            path = gifs_path
            self.make_gif(path, currEpisode)
        '''
        self.experience = episode_buffer
        self.perf_metrics = perf_metrics
        return perf_metrics

    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)

    def calc_estimate_budget(self, budget, current_idx, agent_ID):
        all_budget = []
        current_coord = self.node_coords[current_idx]
        # print(f"current_coord is {current_coord}")
        for i, point_coord in enumerate(self.node_coords):
            dist_current2point = self.env.prm[f"{agent_ID}"].calcDistance(current_coord, point_coord)
            estimate_budget = (budget - dist_current2point) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i + 1, 1)

    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        D_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        for i in range(self.sample_size + 2):
            for j in range(self.sample_size + 2):
                # print(f"i is {i}")
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size + 2):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(self.sample_size + 2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:, 1:32 + 1]
        return eigen_vector

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == '__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, 3, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
