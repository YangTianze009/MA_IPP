import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from AttentionNet import AttentionNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to MA-IPP!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
global_step = None


def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr, intent_difference = tensorboardData
    else:
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr, intent_difference = tensorboardData

    if USE_WANDB:
        import wandb
        wandb.init(project="intent_map", entity="ma_ipp")

        wandb.config.NUM_Runner = NUM_META_AGENT
        wandb.config.NUM_Worker = NUM_THREADS
        wandb.config.BATCH_SIZE = BATCH_SIZE
        wandb.config.BUDGET_RANGE = BUDGET_RANGE
        wandb.config.GAUSSIAN_NUM = GAUSSIAN_NUM
        wandb.config.GREEDY = GREEDY

        wandb.log({"Losses/Value": value, 'episode': curr_episode})
        wandb.log({"Losses/Policy Loss": policyLoss, 'episode': curr_episode})
        wandb.log({"Losses/Value Loss": valueLoss, 'episode': curr_episode})
        wandb.log({"Losses/Entropy": entropy, 'episode': curr_episode})
        wandb.log({"Losses/Grad Norm": gradNorm, 'episode': curr_episode})
        wandb.log({"Perf/Reward": reward, 'episode': curr_episode})
        wandb.log({"Perf/Returns": returns, 'episode': curr_episode})
        wandb.log({"Perf/Remain Budget": remain_budget, 'episode': curr_episode})
        wandb.log({"Perf/Success Rate": success_rate, 'episode': curr_episode})
        wandb.log({"Perf/RMSE": RMSE, 'episode': curr_episode})
        # wandb.log({"Perf/F1 Score": F1, 'episode':curr_episode})
        # wandb.log({"GP/MI": MI, 'episode':curr_episode})
        wandb.log({"GP/Delta Cov Trace": dct, 'episode': curr_episode})
        wandb.log({"GP/Cov Trace": cov_tr, 'episode': curr_episode})


    else:
        writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
        writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
        writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
        writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
        writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
        writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
        writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)
        writer.add_scalar(tag='Perf/Remain Budget', scalar_value=remain_budget, global_step=curr_episode)
        writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
        writer.add_scalar(tag='Perf/RMSE', scalar_value=RMSE, global_step=curr_episode)
        writer.add_scalar(tag='Perf/F1 Score', scalar_value=F1, global_step=curr_episode)
        writer.add_scalar(tag='GP/MI', scalar_value=MI, global_step=curr_episode)
        writer.add_scalar(tag='GP/Delta Cov Trace', scalar_value=dct, global_step=curr_episode)
        writer.add_scalar(tag='GP/Cov Trace', scalar_value=cov_tr, global_step=curr_episode)


def main():
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    # global_network.share_memory()
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)
    # Automatically logs gradients of pytorch model
    # wandb.watch(global_network, log_freq = SUMMARY_WINDOW)

    best_perf = 900
    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_network.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

        best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth')
        best_perf = best_model_checkpoint['best_perf']
        print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)
    else:
        weights = global_network.state_dict()

    # launch the first job on each runner
    dp_model = nn.DataParallel(global_network)

    jobList = []
    # sample_size = np.random.randint(200,400)
    sample_size = 200
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
        curr_episode += 1
    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace',
                   "intent_difference_abs"]
    tensorboardData = []
    trainingData = []
    experience_buffer = []
    for i in range(14):
        experience_buffer.append([])

    ## start
    try:
        while True:
            # wait for any job to be completed
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            # print(done_id)
            # get the results
            # jobResults, metrics, info = ray.get(done_id)[0]
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)
            # done_jobs = list(reversed(done_jobs))
            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(14):
                    experience_buffer[i] += jobResults[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            if np.mean(perf_metrics['cov_trace']) < best_perf and curr_episode % 32 == 0:
                best_perf = np.mean(perf_metrics['cov_trace'])
                print('Saving best model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
                              "best_perf": best_perf}
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

            update_done = False
            print(f"experience buffer[0] is {len(experience_buffer[0])}")
            while len(experience_buffer[0]) >= BATCH_SIZE:
                rollouts = copy.deepcopy(experience_buffer)
                for i in range(len(rollouts)):
                    rollouts[i] = rollouts[i][:BATCH_SIZE]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = []
                    for i in range(14):
                        experience_buffer.append([])
                    # sample_size = np.random.randint(200, 400)

                print(f"training start")
                node_inputs_batch = torch.stack(rollouts[0], dim=0)  # (batch,sample_size+2,2)
                edge_inputs_batch = torch.stack(rollouts[1], dim=0)  # (batch,sample_size+2,k_size)
                current_inputs_batch = torch.stack(rollouts[2], dim=0)  # (batch,1,1)
                action_batch = torch.stack(rollouts[3], dim=0)  # (batch,1,1)
                value_batch = torch.stack(rollouts[4], dim=0)  # (batch,1,1)
                reward_batch = torch.stack(rollouts[5], dim=0)  # (batch,1,1)
                value_prime_batch = torch.stack(rollouts[6], dim=0)  # (batch,1,1)
                target_v_batch = torch.stack(rollouts[7])
                budget_inputs_batch = torch.stack(rollouts[8], dim=0)
                LSTM_h_batch = torch.stack(rollouts[9])
                LSTM_c_batch = torch.stack(rollouts[10])
                mask_batch = torch.stack(rollouts[11])
                pos_encoding_batch = torch.stack(rollouts[12])
                agent_input_batch = torch.stack(rollouts[13])

                if device != local_device:
                    node_inputs_batch = node_inputs_batch.to(device)
                    edge_inputs_batch = edge_inputs_batch.to(device)
                    current_inputs_batch = current_inputs_batch.to(device)
                    action_batch = action_batch.to(device)
                    value_batch = value_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    value_prime_batch = value_prime_batch.to(device)
                    target_v_batch = target_v_batch.to(device)
                    budget_inputs_batch = budget_inputs_batch.to(device)
                    LSTM_h_batch = LSTM_h_batch.to(device)
                    LSTM_c_batch = LSTM_c_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    pos_encoding_batch = pos_encoding_batch.to(device)
                    agent_input_batch = agent_input_batch.to(device)

                # PPO
                with torch.no_grad():
                    logp_list, value, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch,
                                                            current_inputs_batch, LSTM_h_batch, LSTM_c_batch,
                                                            pos_encoding_batch, mask_batch)
                old_logp = torch.gather(logp_list, 1, action_batch.squeeze(1)).unsqueeze(1)  # (batch_size,1,1)
                advantage = (reward_batch + GAMMA * value_prime_batch - value_batch)  # (batch_size, 1, 1)
                # advantage = target_v_batch - value_batch

                entropy = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                scaler = GradScaler()

                for i in range(8):
                    with autocast():
                        logp_list, value, _, _ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch,
                                                          current_inputs_batch, LSTM_h_batch, LSTM_c_batch,
                                                          pos_encoding_batch, mask_batch)
                        logp = torch.gather(logp_list, 1, action_batch.squeeze(1)).unsqueeze(1)
                        ratios = torch.exp(logp - old_logp.detach())
                        surr1 = ratios * advantage.detach()
                        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantage.detach()
                        policy_loss = -torch.min(surr1, surr2)
                        policy_loss = policy_loss.mean()

                        # value_clipped = value + (target_v_batch - value).clamp(-0.2, 0.2)
                        # value_clipped_loss = (value_clipped-target_v_batch).pow(2)
                        # value_loss =(value-target_v_batch).pow(2).mean()
                        # value_loss = torch.max(value_loss, value_clipped_loss).mean()

                        mse_loss = nn.MSELoss()
                        value_loss = mse_loss(value, target_v_batch).mean()

                        entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                        loss = policy_loss + 0.5 * value_loss + 0 * entropy_loss
                    global_optimizer.zero_grad()
                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                    # global_optimizer.step()
                    scaler.step(global_optimizer)
                    scaler.update()
                lr_decay.step()

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                        entropy.item(), grad_norm.item(), target_v_batch.mean().item(), *perf_data]
                trainingData.append(data)
                print(f"len of the training data is {len(trainingData)}")

                # experience_buffer = []
                # for i in range(8):
                #    experience_buffer.append([])

            if len(trainingData) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []
                if USE_WANDB:
                    print("Write to Wandb")
                else:
                    print("Write to Tensorboard")

            # get the updated global weights
            if update_done == True:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    global_network.to(device)
                else:
                    weights = global_network.state_dict()

            jobList = []
            for i, meta_agent in enumerate(meta_agents):
                jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
                curr_episode += 1

            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict()}
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')


    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
