import threading

import torch
import numpy as np
import ray
import os
from AttentionNet import AttentionNet
from worker import Worker
from parameter import *
import random
import time
import env


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM)
        self.localNetwork.to(self.device)
        self.env = None
        # self.env = env.Env( sample_size=SAMPLE_SIZE, k_size=K_SIZE,
        # budget_range=BUDGET_RANGE, save_image=SAVE_IMAGE)

    def get_weights(self):
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def multiThreadedJob(self, episodeNumber, budget_range, sample_size, sample_length):
        save_img = True  # False if episodeNumber % SAVE_IMG_GAP == 0 else False
        jobResults = []
        for i in range(14):
            jobResults.append([])
        rollouts = []
        perf_metrics = {}
        workers = []
        worker_threads = []
        workerNames = ["worker_" + str(i + 1) for i in range(NUM_THREADS)]

        self.env = env.Env(sample_size=SAMPLE_SIZE, k_size=K_SIZE,
                           budget_range=BUDGET_RANGE, save_image=SAVE_IMAGE)

        # save_img = False
        for agentID in range(1, NUM_THREADS + 1):
            workers.append(
                Worker(self.metaAgentID, agentID, self.localNetwork, episodeNumber, self.env,
                       sample_size, sample_length, self.device, save_image=save_img, greedy=GREEDY))
        for i, w in enumerate(workers):
            worker_work = lambda: w.work(episodeNumber)
            t = threading.Thread(target=worker_work, name=workerNames[i])
            t.start()

            worker_threads.append(t)


        # print(f"rollout size is: {np.array(rollouts).size}")
        cov_trace = 900
        for w in workers:
            wait_step = 0
            while w.experience == None:
                # print(w.if_done, w.curren_step, w.agent_ID)
                time.sleep(0.5)
            # else:
            #     print(f"Game Over {w.if_done, w.curren_step}")
            rollouts.append(w.experience)
            if cov_trace > w.perf_metrics["cov_trace"]:
                cov_trace = w.perf_metrics["cov_trace"]
            print(f'cov_trace is {w.perf_metrics["cov_trace"]}')
            # print(np.array(w.experience).shape)
            perf_metrics = w.perf_metrics
        perf_metrics["cov_trace"] = cov_trace
        print(f"final cov_trace is {cov_trace}")

        for b in range(len(rollouts)):
            for a in range(14):
                for c in range(len(rollouts[b][a])):
                    jobResults[a].append(rollouts[b][a][c])

        randnum = random.randint(0, 100)
        for i in range(14):
            random.seed(randnum)
            random.shuffle(jobResults[i])

        # print(f"this finish")
        return jobResults, perf_metrics

    def job(self, global_weights, episodeNumber, budget_range, sample_size=SAMPLE_SIZE, sample_length=None):
        print("starting episode {} on metaAgent {}".format(episodeNumber, self.metaAgentID))
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)

        jobResults, metrics = self.multiThreadedJob(episodeNumber, budget_range, sample_size, sample_length)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return jobResults, metrics, info


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
