import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm

def build_occupancy_maps(self, human_states):
    """

    :param human_states:
    :return: tensor of shape (# human - 1, self.cell_num ** 2)
    """
    occupancy_maps = []
    for human in human_states:
        other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                        for other_human in human_states if other_human != human], axis=0)
        other_px = other_humans[:, 0] - human.px
        other_py = other_humans[:, 1] - human.py
        # new x-axis is in the direction of human's velocity
        human_velocity_angle = np.arctan2(human.vy, human.vx)
        other_human_orientation = np.arctan2(other_py, other_px)
        rotation = other_human_orientation - human_velocity_angle
        distance = np.linalg.norm([other_px, other_py], axis=0)
        other_px = np.cos(rotation) * distance
        other_py = np.sin(rotation) * distance

        # compute indices of humans in the grid
        other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
        other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
        other_x_index[other_x_index < 0] = float('-inf')
        other_x_index[other_x_index >= self.cell_num] = float('-inf')
        other_y_index[other_y_index < 0] = float('-inf')
        other_y_index[other_y_index >= self.cell_num] = float('-inf')
        grid_indices = self.cell_num * other_y_index + other_x_index
        occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
        if self.om_channel_size == 1:
            occupancy_maps.append([occupancy_map.astype(int)])
        else:
            # calculate relative velocity for other agents
            other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
            rotation = other_human_velocity_angles - human_velocity_angle
            speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
            other_vx = np.cos(rotation) * speed
            other_vy = np.sin(rotation) * speed
            dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
            for i, index in np.ndenumerate(grid_indices):
                if index in range(self.cell_num ** 2):
                    if self.om_channel_size == 2:
                        dm[2 * int(index)].append(other_vx[i])
                        dm[2 * int(index) + 1].append(other_vy[i])
                    elif self.om_channel_size == 3:
                        dm[3 * int(index)].append(1)
                        dm[3 * int(index) + 1].append(other_vx[i])
                        dm[3 * int(index) + 2].append(other_vy[i])
                    else:
                        raise NotImplementedError
            for i, cell in enumerate(dm):
                dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
            occupancy_maps.append([dm])

    return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.robot_feature = nn.Sequential

        #robot_channel
    
        #pedestrian channel
        self.pedestrian_extractor1 = mlp(13+16*3, [150, 100], last_relu=True)
        self.pedestrian_extractor2 = mlp(50, [100, 50])
        self.attention = mlp(50*2, [100, 100, 1])
        self.cell_size = 1.0
        self.cell_num = 4
        # self.pedestrian_extractor3 = mlp(50+6, [150, 100, 100])

        #concat channel
        self.concat_extractor1 = mlp(100+6, [100 , 100], last_relu=True)
    
    def forward(self, obs):
        robot_input = obs['robot_node']
        pedestrian_input = obs['pedestrian_node']
        size = pedestrian_input.shape
        pedestrian_input = [human_state for human_state in pedestrian_input]
        pedestrian_input = build_occupancy_maps(self, pedestrian_input)
        
        pedestrian_feature1 = self.pedestrian_extractor1(pedestrian_input)
        pedestrian_feature2 = self.pedestrian_extractor2(pedestrian_feature1)
        global_state = torch.mean(pedestrian_feature1.view(5, -1), 1, keepdim=True)
        global_state = global_state.expand((5, 100)).contiguous().view(-1, 100)
        attention_input = torch.cat(pedestrian_feature1, global_state, dim=1)

        scores = self.attention(attention_input).view(5, 1).squeeze(dim=2)

        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = pedestrian_feature2.view(5, -1)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        joint_state = torch.cat([robot_input, weighted_feature], dim=1)
        concat_feature = self.concat_extractor1(joint_state)
        return concat_feature

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super(ProgressBarCallback, self).__init__()
        self.pbar = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from crowd_sim import *
from crowd_sim.envs.utils.robot import Robot
import argparse
import configparser
import logging
import time

timesteps = 2000
# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
progress_bar = ProgressBarCallback(total_timesteps=timesteps)
n_envs=2
env = make_vec_env('CrowdSim-v0', n_envs=n_envs)

model = PPO("MultiInputPolicy", env, verbose=1)
start_time = time.time()

model.learn(total_timesteps=timesteps, callback=progress_bar)
end_time = time.time()
model.save("ppo_crowdnav")
# 所要時間を計算
elapsed_time = end_time - start_time

# 所要時間をログに記録
logging.info(f"Elapsed Time: {elapsed_time} seconds")
