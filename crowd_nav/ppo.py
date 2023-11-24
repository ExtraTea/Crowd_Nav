import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
def build_occupancy_maps(self, human_states):
    occupancy_maps = []
    for i, human in enumerate(human_states):
        # other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
        #                                 for other_human in human_states if other_human != human], axis=0)
        
        # other_humans = torch.cat(torch.from_numpy[np.array([(human_states[j][0],human_states[j][1],human_states[j][2],human_states[j][3])]) for j in range(len(human_states)) if j != i], axis=0)
        other_humans = torch.stack([torch.tensor([human_states[j][0], human_states[j][1], human_states[j][2], human_states[j][3]]) for j in range(len(human_states)) if j != i]).to("cuda:0")
        # human = human.to("cuda:0")
        # print(other_humans.size())
        other_px = other_humans[:, 0] - human[0]
        other_py = other_humans[:, 1] - human[1]
        # new x-axis is in the direction of human's velocity
        human_velocity_angle = torch.atan2(human[2], human[3])
        other_human_orientation = torch.atan2(other_py, other_px)
        rotation = other_human_orientation - human_velocity_angle
        distance = torch.norm(torch.stack([other_px, other_py]), dim=0)
        other_px = torch.cos(rotation) * distance
        other_py = torch.sin(rotation) * distance

        # compute indices of humans in the grid
        cell_size_temp = torch.tensor(self.cell_size).to("cuda:0")
        cell_num_temp = torch.tensor(self.cell_num).to("cuda:0")
        other_x_index = torch.floor(other_px / cell_size_temp + cell_num_temp / 2)
        other_y_index = torch.floor(other_py / cell_size_temp + cell_num_temp / 2)
        other_x_index[other_x_index < 0] = torch.tensor(float('-inf')).to("cuda:0")
        other_x_index[other_x_index >= self.cell_num] = torch.tensor(float('-inf')).to("cuda:0")
        other_y_index[other_y_index < 0] = torch.tensor(float('-inf')).to("cuda:0")
        other_y_index[other_y_index >= self.cell_num] = torch.tensor(float('-inf')).to("cuda:0")
        grid_indices = cell_num_temp * other_y_index + other_x_index
        # occupancy_map = torch.isin(range(int(cell_num_temp.item() ** 2)), grid_indices)
        occupancy_map = torch.isin(torch.arange(int(cell_num_temp.item() ** 2)).to("cuda:0"), grid_indices)
        if self.om_channel_size == 1:
            occupancy_maps.append([occupancy_map.astype(int)])
        else:
            # calculate relative velocity for other agents
            other_human_velocity_angles = torch.arctan2(other_humans[:, 3], other_humans[:, 2])
            rotation = other_human_velocity_angles - human_velocity_angle
            speed = torch.norm(other_humans[:, 2:4], dim=1)
            other_vx = torch.cos(rotation) * speed
            other_vy = torch.sin(rotation) * speed
            # dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
            dm = torch.stack([torch.tensor([0]) for _ in range(int(cell_num_temp.item() ** 2 * self.om_channel_size))]).to("cuda:0")
            for i, index in enumerate(grid_indices):
                if index.item() in range(int(cell_num_temp.item() ** 2)):
                    if self.om_channel_size == 2:
                        dm[2 * int(index.item())].append(other_vx[i])
                        dm[2 * int(index.item()) + 1].append(other_vy[i])
                    elif self.om_channel_size == 3:
                        dm[3 * int(index.item())] += 1
                        dm[3 * int(index.item()) + 1] = other_vx[i] + dm[3 * int(index.item()) + 1]
                        dm[3 * int(index.item()) + 2] = other_vy[i] + dm[3 * int(index.item()) + 2]

            for i, cell in enumerate(dm):
                dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                if i%self.om_channel_size != 0:
                    continue
                dm[i+1] = dm[i+1] / dm[i]
                dm[i+2] = dm[i+2] / dm[i]
                dm[i] = 1
            occupancy_maps.append(dm)
    
    occupancy_maps = [occupancy_map.squeeze(1) for occupancy_map in occupancy_maps]
    # occupancy_maps = torch.cat([occupancy_map for occupancy_map in occupancy_maps], dim=2)
    # occupancy_maps = torch.stack([torch.cat([occupancy_map for occupancy_map in occupancy_maps], dim=1)]).to("cuda:0")
    occupancy_maps = torch.stack(occupancy_maps).to("cuda:0")
    # print(occupancy_maps.size())
    return occupancy_maps

def pedestrian_intput_concat(occupancy_maps, pedestrian_inputs, robot_rotated_input, robot_input):
    def rotate(state, robot_input):
        # print(human_states.size())
        # print(robot_input.size())
        human_states_selected = human_states[:, :, [0,1]]
        robot_input_selected = robot_input[:,:,[0,1]]
        # print(len(robot_input))
        robot_input_selected = robot_input_selected.repeat(1,5,1)
        # print(human_states_selected.size())
        # print(robot_input_selected.size())
        
        da = torch.norm(human_states_selected - robot_input_selected, dim=2, keepdim=True)
        
        human_states_selected = human_states[:, :, [3]]
        robot_input_selected = robot_input[:,:,[4]]
        robot_input_selected = robot_input_selected.repeat(1,5,1)
        radius_sum = human_states_selected + robot_input_selected
        # print(radius_sum.size())
        state[:,:,[11]] = da
        return torch.cat((state, radius_sum), dim=2)
    # print(robot_rotated_input.size())
    robot_rotated_input = robot_rotated_input.unsqueeze(1)
    stacked_robot_rotated_input = torch.cat([robot_rotated_input for _ in range(5)], dim=1).to("cuda:0")
    # print(stacked_robot_rotated_input.size())
    # print(pedestrian_inputs.size())
    human_states = torch.cat((stacked_robot_rotated_input, pedestrian_inputs), dim=2)
    rotated_human_states = rotate(human_states, robot_input)
    return torch.cat([occupancy_maps, rotated_human_states], dim=2)

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
        self.om_channel_size = 3

        #robot_channel
    
        #pedestrian channel
        self.pedestrian_extractor1 = mlp(13+16*3, [150, 100], last_relu=True)
        self.pedestrian_extractor2 = mlp(100, [100, 50])
        self.attention = mlp(100+100, [100, 100, 1])
        self.cell_size = 1.0
        self.cell_num = 4
        # self.pedestrian_extractor3 = mlp(50+6, [150, 100, 100])

        #obstacle channel
        self.obstacle_extractor1 = mlp(72, [128, 100], last_relu=True)

        #concat channel
        self.concat_extractor1 = mlp(100+6+50, [100 , 100], last_relu=True)
        
    
    def forward(self, obs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        robot_rotated_input = obs['robot_rotated_node'].float().to(device)
        # robot_rotated_input = robot_rotated_input.squeeze(0)
        pedestrian_input = obs['pedestrian_node'].float().to(device)
        obstacle_input = obs['angular_map'].float().to(device)
        robot_input = obs['robot_node'].float().to(device)
        occupancy_maps = torch.stack([build_occupancy_maps(self, pedestrian_input_each_env) for pedestrian_input_each_env in pedestrian_input], dim=0).float().to(device)
        pedestrian_input.squeeze(0)
        # print(occupancy_maps.size())
        # print(pedestrian_input.size())
        # print(robot_rotated_input.size())
        # print(robot_input.size())
        pedestrian_input = pedestrian_intput_concat(occupancy_maps, pedestrian_input, robot_rotated_input, robot_input)
        pedestrian_feature1 = self.pedestrian_extractor1(pedestrian_input)
        pedestrian_feature2 = self.pedestrian_extractor2(pedestrian_feature1)
        # global_state = torch.mean(pedestrian_feature1.view(5, -1), 1, keepdim=True).float().to(device)
        global_state = torch.mean(pedestrian_feature1, keepdim=True, dim=1)
        global_state = global_state.expand((len(global_state),5, 100)).contiguous()
        attention_input = torch.cat((pedestrian_feature1, global_state), dim=2)

        scores = self.attention(attention_input).view(len(attention_input),5,1).squeeze(dim=2)
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        weights = F.softmax(scores, dim=1)
        # print(weights.size())   
        # self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        features = pedestrian_feature2.view(5, -1)
        # weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        weighted_feature = torch.sum(torch.mul(weights.unsqueeze(2), pedestrian_feature2), dim=1)
        # print(weighted_feature.size())
        obstacle_feature = self.obstacle_extractor1(obstacle_input)
        
        robot_rotated_input = robot_rotated_input.squeeze(0).squeeze(0)

        weighted_feature = weighted_feature.squeeze(0)
        # print(robot_rotated_input.size())
        # print(weighted_feature.size())
        # print(obstacle_feature.size())
        joint_state = torch.cat([robot_rotated_input, weighted_feature, obstacle_feature], dim=1)
        # print(joint_state.size())
        concat_feature = self.concat_extractor1(joint_state)
        # print(concat_feature.size())
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
from stable_baselines3.common.vec_env import SubprocVecEnv
import logging
import time
if __name__ == '__main__':
    # 訓練の設定
    timesteps = 1024*16
    n_envs=8

    # ロギングの設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    progress_bar = ProgressBarCallback(total_timesteps=timesteps/n_envs)

    env = make_vec_env('CrowdSim-v0', vec_env_cls=SubprocVecEnv, n_envs=n_envs)

    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=100),
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # model = PPO("MultiInputPolicy", env, verbose=1)

    start_time = time.time()

    model.learn(total_timesteps=timesteps, callback=progress_bar)
    end_time = time.time()
    model.save("ppo_crowdnav")
    # 所要時間を計算
    elapsed_time = end_time - start_time

    # 所要時間をログに記録
    logging.info(f"Elapsed Time: {elapsed_time} seconds")
