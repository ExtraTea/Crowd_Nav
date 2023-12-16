import gymnasium as gym
import numpy as np
import torch
from crowd_sim import *
import copy

# def run_environment(executions):
#     env = gym.make('CrowdSim-v0', render_mode='human')
#     all_observations = []
#     all_actions = []
#     all_dones = []
#     all_infos = []

#     for _ in range(executions):
#         state, info = env.reset()
#         done = False
#         observations_temp = []
#         actions_temp = []
#         done_temp = []
#         info_temp = []
#         while not done:
#             last_state = copy.deepcopy(state)
#             state, rew, done, truncated, info = env.step(None)
#             observations_temp.append(last_state)
#             actions_temp.append(info['action'])
#             done_temp.append(done)
#             info_temp.append(info['event'])

#             if done:    
#                 # ステップ数が50未満の場合、データを埋める
#                 # if len(done_temp) < 100:
#                 #     remaining_steps = 100 - len(done_temp)
#                 #     first_observation = observations_temp[0]
#                 #     first_info = info_temp[0]
#                 #     # 配列の最初にデータを埋める
#                 #     observations_temp = [first_observation] * remaining_steps + observations_temp
#                 #     actions_temp = [np.array([0, 0])] * remaining_steps + actions_temp
#                 #     done_temp = [False] * remaining_steps + done_temp
#                 #     info_temp = [first_info] * remaining_steps + info_temp
#                 all_observations.append(observations_temp)
#                 all_actions.append(actions_temp)
#                 all_dones.append(done_temp)
#                 all_infos.append(info_temp)
# env = gym.make('CrowdSim-v0', render_mode='human')
# state, info = env.reset()
# done = False
# observations = []
# actions = []
# dones = []
# infos = []
# poipoi = 0
# observations_temp = []
# actions_temp = []
# done_temp = []
# info_temp = []
# while True:
#     env.render()
#     last_state = copy.deepcopy(state)
#     state, rew, done, truncated, info = env.step(None)
#     observations_temp.append(last_state)
#     actions_temp.append(info['action'])
#     done_temp.append(done)
#     info_temp.append(info['event'])
#     if truncated:
#         # print("truncated")
#         state, info = env.reset()
        
#     if done:
#         print(info['event'])
#         if info['event'] == 'reaching_goal':
#             observations.append(observations_temp)
#             actions.append(actions_temp)
#             dones.append(done_temp)
#             infos.append(info_temp)
#             observations_temp = []
#             actions_temp = []
#             done_temp = []
#             info_temp = []
#             poipoi +=1
#             if poipoi == 1024*32:
#                 break
#         state, info = env.reset()
# torch.save({'observations': observations, 'actions': actions, 'dones': dones, 'infos': info}, "expert_dataset_wall.pt")


import concurrent.futures
import gymnasium as gym
import numpy as np
import torch
from crowd_sim import *
import copy

def run_environment(executions):
    env = gym.make('CrowdSim-v0', render_mode='human')
    all_observations = []
    all_actions = []
    all_dones = []
    all_infos = []

    for _ in range(executions):
        state, info = env.reset()
        done = False
        observations_temp = []
        actions_temp = []
        done_temp = []
        info_temp = []
        while not done:
            last_state = copy.deepcopy(state)
            state, rew, done, truncated, info = env.step(None)
            observations_temp.append(last_state)
            actions_temp.append(info['action'])
            done_temp.append(done)
            info_temp.append(info['event'])
            if done and info['event'] == 'reaching_goal':
                all_observations.append(observations_temp)
                all_actions.append(actions_temp)
                all_dones.append(done_temp)
                all_infos.append(info_temp)

    return all_observations, all_actions, all_dones, all_infos

def main():
    number_of_processes = 16
    total_executions = 1024 * 4

    executions_per_process = total_executions // number_of_processes
    extra_executions = total_executions % number_of_processes

    all_observations = []
    all_actions = []
    all_dones = []
    all_infos = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_environment, executions_per_process + (1 if i < extra_executions else 0))
            for i in range(number_of_processes)
        ]
        for future in concurrent.futures.as_completed(futures):
            observations, actions, dones, infos = future.result()
            all_observations.extend(observations)
            all_actions.extend(actions)
            all_dones.extend(dones)
            all_infos.extend(infos)

    # 結果をまとめて保存
    dataset = {'observations': all_observations, 'actions': all_actions, 'dones': all_dones, 'infos': all_infos}
    torch.save(dataset, "expert_dataset_dis.pt")

if __name__ == "__main__":
    main()