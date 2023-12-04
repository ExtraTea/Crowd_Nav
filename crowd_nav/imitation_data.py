import gymnasium as gym
import numpy as np
import torch
from crowd_sim import *
import copy
env = gym.make('CrowdSim-v0', render_mode='human')
state, info = env.reset()
done = False
observations = []
actions = []
dones = []
infos = []
poipoi = 0
observations_temp = []
actions_temp = []
done_temp = []
info_temp = []
while True:
    env.render()
    last_state = copy.deepcopy(state)
    state, rew, done, truncated, info = env.step(None)
    observations_temp.append(last_state)
    actions_temp.append(info['action'])
    done_temp.append(done)
    info_temp.append(info['event'])
    if truncated:
        # print("truncated")
        pass
    if done:
        # print(info['event'])
        if info['event'] == 'reaching_goal':
            observations.append(observations_temp)
            actions.append(actions_temp)
            dones.append(done_temp)
            infos.append(info_temp)
            observations_temp = []
            actions_temp = []
            done_temp = []
            info_temp = []
            poipoi +=1
            if poipoi == 10:
                break
        state, info = env.reset()
torch.save({'observations': observations, 'actions': actions, 'dones': dones, 'infos': info}, "expert_dataset.pt")