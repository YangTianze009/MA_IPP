import copy
import os

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
import random

# store all the agents route for communication
agent_route = dict()
for i in range(1, NUM_THREADS + 1):
    agent_route[f"{i}"] = []

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

episode = 0
lock = Lock()


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


class Worker:
    def __init__(self, metaAgentID, agent_ID, localNetwork, global_step, env,
                 sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):
        global agent_route, all_samples, all_done, all_reset
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
        global episode
        # reset the agent route and all samples in new episode
        if episode != currEpisode:
            episode = currEpisode
            for i in range(1, NUM_THREADS + 1):
                agent_route[f"{i}"] = []
            for i in range(1, NUM_THREADS + 1):
                all_samples[f"{i}"] = []

        episode_buffer = []
        dist_residual = 0
        perf_metrics = dict()
        done = False
        for i in range(14):
            episode_buffer.append([])
        node_coords, graph, node_info, node_std, budget = self.env.reset(agent_ID=self.agent_ID)
        # make sure that all the agents finish the reset process
        all_reset[f"{self.agent_ID}"] = True
        for i in range(1, NUM_THREADS+1):
            while not all_reset[f"{i}"]:
                pass

        # print(f"node_coords is {node_coords.shape}")
        self.node_coords = node_coords

        # reset initial each agent's route in a dictionary(eg. agent 7 has the initial position which is node 7)
        # agent_ID = initial node index for this agent
        flag = 0
        for t in range(1, NUM_THREADS + 1):
            if agent_route[f"{t}"] != []:
                flag = 1
        if flag == 0:
            for t in range(1, NUM_THREADS + 1):
                agent_route[f"{t}"].append(1)

        n_nodes = node_coords.shape[0]
        current_index = torch.tensor([1]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes, 1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1, agent_ID=self.agent_ID)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 1)

        agent_input = [index for index in range(1, NUM_THREADS + 1)]

        for agent_ID in range(1, NUM_THREADS + 1):
            agent_input[agent_ID - 1] = self.cal_agent_input(1, 1, self.agent_ID, agent_ID)
        agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)

        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device)  # (1, sample_size+2, 32)
        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, k_size)

        relative_node_coords = []

        # calculate the relative coordinates of all nodes
        for i in range(0, self.sample_size + 2):
            relative_node_coords.append([])
        for i in range(0, self.sample_size + 2):
            relative_node_coords[i] = self.cal_relative_coord(i, current_index)
        node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 4)

        LSTM_h = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1, 1, EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

        # choose the first step for each agent

        with torch.no_grad():
            logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs,
                                                              current_index, LSTM_h, LSTM_c, pos_encoding,
                                                              agent_input, mask)
        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        current_agent_dis = self.cal_agent_distance(self.agent_ID)
        next_node_index = edge_inputs[:, current_index.item(), action_index.item()]

        agent_route[f"{self.agent_ID}"].append(next_node_index.item())  # default
        sample_coords, remain_budget, dist_residual = self.env.observed_positions(current_index.item(),
                                                                                  next_node_index.item(),
                                                                                  self.sample_length,
                                                                                  agent_ID=self.agent_ID,
                                                                                  dist_residual=dist_residual)
        for sample in sample_coords:
            all_samples[f"{self.agent_ID}"].append(sample)

        sample_numbers = current_agent_dis // self.sample_length
        # print(sample_numbers)

        # agent_input = [index for index in range(1, NUM_THREADS + 1)]

        for agent_ID in range(1, NUM_THREADS + 1):
            while len(agent_route[f"{agent_ID}"]) < 2:
                if len(agent_route[f"{agent_ID}"]) == 2:
                    break
            # agent_input[agent_ID - 1] = agent_route[f"{agent_ID}"][-1]

        '''
        for agent_ID in range(1, NUM_THREADS + 1):
            agent_input[agent_ID - 1] = self.cal_relative_coord(agent_input[agent_ID - 1], self.agent_ID)

        agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)
        '''

        '''
        for agent_ID in range(1, NUM_THREADS + 1):
            if agent_route[f"{agent_ID}"][-1] != 0:
                while len(all_samples[f"{agent_ID}"]) < sample_numbers:
                    time.sleep(0.5)

            else:
                pass
        '''
        route = agent_route[f"{agent_ID}"]
        reward, done, node_info, node_std, ground_truth = self.env.step(route, next_node_index.item(), all_samples,
                                                                        sample_numbers, done,
                                                                        agent_ID=self.agent_ID)
        current_index = next_node_index.unsqueeze(0).unsqueeze(0)

        for step_i in range(steps):

            all_done = dict()
            for i in range(1, NUM_THREADS + 1):
                all_done[f"{i}"] = False

            self.curren_step += 1
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding
            # episode_buffer[13] += agent_input

            current_agent_dis = self.cal_agent_distance(self.agent_ID)
            print(f"current distance is {current_agent_dis}", f"self.agent_id is {self.agent_ID}")
            # get the relative agent position with current agent
            # if this agent has a longer distance from the start node than other agents, then wait for other agents
            for agent_ID in range(1, NUM_THREADS + 1):
                if not all_done[f"{agent_ID}"]:
                    while current_agent_dis > self.cal_agent_distance(agent_ID):
                        # print(f"self.agentID is {self.agent_ID}", f"stuck is here in distance")
                        time.sleep(0.5)
                       # print(f"self.agent_ID is stuck in distance{self.agent_ID}", f"current distance is {current_agent_dis}",
                               #f"agent ID is {agent_ID}", "\n", f"other distance is: {self.cal_agent_distance(agent_ID)}")
                        if all_done[f"{agent_ID}"]:
                            break
                else:
                    pass

            agent_input = [index for index in range(1, NUM_THREADS + 1)]
            for agent_ID in range(1, NUM_THREADS + 1):
                agent_input[agent_ID - 1] = self.cal_agent_input(agent_route[f"{agent_ID}"][-1],
                                                                 current_index.item(), self_ID=self.agent_ID,
                                                                 others_ID=agent_ID)
            agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)

            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs,
                                                                  current_index, LSTM_h, LSTM_c, pos_encoding,
                                                                  agent_input, mask)
            # next_node (1), logp_list (1, 10), value (1,1,1)
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            episode_buffer[8] += budget_inputs
            episode_buffer[13] += agent_input

            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            sample_coords, remain_budget, dist_residual = self.env.observed_positions(current_index.item(),
                                                                                      next_node_index.item(),
                                                                                      self.sample_length,
                                                                                      agent_ID=self.agent_ID,
                                                                                      dist_residual=dist_residual)
            # decide the destination
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=next_node_index.item(),
                                                      agent_ID=self.agent_ID)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
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

            # store the sample coordinates in all_samples to decide which samples point should be used to update GP
            for sample in sample_coords:
                all_samples[f"{self.agent_ID}"].append(sample)
            # print(f"next node index is {next_node_index.item()}")

            # calculate the observation points in each agent
            sample_numbers = current_agent_dis // self.sample_length
            # print(f"sample numbers are {sample_numbers}")
            route = agent_route[f"{self.agent_ID}"]
            reward, done, node_info, node_std, ground_truth = self.env.step(route, next_node_index.item(), all_samples,
                                                                            sample_numbers, done,
                                                                            agent_ID=self.agent_ID)

            # print(f"agent1 = {agent1}, agent2 = {agent2}, agent3 = {agent3}")
            if SAVE_IMAGE:

                print(f"agent ID is {self.agent_ID}")

                agent1 = len(agent_route["1"])
                agent2 = len(agent_route["2"])
                agent3 = len(agent_route["3"])
                n = agent1 + agent2 + agent3
                self.gifs_path = '{}/{}'.format(gifs_path, self.global_step)
                if self.save_image:
                    if not os.path.exists(self.gifs_path):
                        os.makedirs(self.gifs_path)
                    self.env.plot(agent_route, n, self.gifs_path, ground_truth=ground_truth,
                                  remain_budget=remain_budget, agent_ID=self.agent_ID)

                    # print(f"self.agentID is {self.agent_ID}"); time.sleep(8)
                    # print(agent_route)
                    # print(f"self.agentID is over {self.agent_ID}")

            # store the next_node_index in a global variable
            agent_route[f"{self.agent_ID}"].append(next_node_index.item())

            print(f"agent route is {agent_route}")

            '''
            for agent_ID in range(1, NUM_THREADS + 1):
                agent_input[agent_ID - 1] = self.cal_agent_input(agent_route[f"{agent_ID}"][-1],
                                                                    next_node_index.item(), self_ID=self.agent_ID, others_ID=agent_ID)

            agent_input = torch.FloatTensor(agent_input).unsqueeze(0).to(self.device)  # (1, num_threads, 2)
            '''

            # if (not done and i==127):
            # reward += -np.linalg.norm(self.env.node_coords[self.env.current_node_index,:]-self.env.node_coords[0,:])

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)

            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)

            # calculate the relative coordinates of all nodes
            for t in range(0, self.sample_size + 2):
                relative_node_coords[t] = []
            for t in range(0, self.sample_size + 2):
                relative_node_coords[t] = self.cal_relative_coord(t, current_index)
            node_inputs = np.concatenate((relative_node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, sample_size+2, 4)
            # budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            # budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            # current_edge = torch.gather(edge_inputs, 1, next_node_index.repeat(1, 1, K_SIZE))
            # current_edge = current_edge.permute(0, 2, 1)
            # connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge)
            # connected_nodes_budget = connected_nodes_budget.squeeze(0).squeeze(0)
            # connected_nodes_budget = connected_nodes_budget.tolist()
            # done = True
            # for i in connected_nodes_budget[1:]:
            #     if i[0] > 0:
            #         done = False
            # all_done[f"{self.agent_ID}"] = done

            # mask last five node
            mask = torch.zeros((1, self.sample_size + 2, K_SIZE), dtype=torch.int64).to(self.device)

            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                perf_metrics['remain_budget'] = remain_budget / budget
                # perf_metrics['collect_info'] = 1 - remain_info.sum()
                # perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                perf_metrics['RMSE'] = self.env.RMSE
                perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(ground_truth)
                perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                # print(perf_metrics['delta_cov_trace'])
                perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                perf_metrics['cov_trace'] = self.env.cov_trace
                perf_metrics['success_rate'] = True
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
            perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['success_rate'] = False

        # print(f'agent {self.agent_ID}: route is {route}')

        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward, dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)

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
