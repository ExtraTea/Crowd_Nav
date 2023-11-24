import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
def build_occupancy_maps_old(self, human_states):
    occupancy_maps = []
    for i, human in enumerate(human_states):
        other_humans = torch.stack([torch.tensor([human_states[j][0], human_states[j][1], human_states[j][2], human_states[j][3]]) for j in range(len(human_states)) if j != i]).to("cuda:0")
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

def build_occupancy_maps(self, human_states):
    num_humans = human_states.size(0)

    # 相対位置と速度のベクトル化計算
    rel_positions = human_states[:, None, :2] - human_states[None, :, :2]
    rel_velocities = human_states[:, None, 2:] - human_states[None, :, 2:]

    # 自己除外マスク
    self_mask = ~torch.eye(num_humans, dtype=torch.bool).to(self.device)

    # 角度と距離の計算
    human_velocity_angle = torch.atan2(human_states[:, 2], human_states[:, 3])
    other_human_orientation = torch.atan2(rel_positions[..., 1], rel_positions[..., 0])
    rotation = other_human_orientation - human_velocity_angle[:, None]
    distance = torch.norm(rel_positions, dim=-1)

    # 回転後の相対位置
    other_px = torch.cos(rotation) * distance
    other_py = torch.sin(rotation) * distance

    # グリッドインデックスの計算
    other_x_index = torch.floor(other_px / self.cell_size + self.cell_num / 2)
    other_y_index = torch.floor(other_py / self.cell_size + self.cell_num / 2)

    # 範囲外インデックスの処理
    valid_indices = (other_x_index >= 0) & (other_x_index < self.cell_num) & \
                    (other_y_index >= 0) & (other_y_index < self.cell_num)
    valid_indices &= self_mask

    # 占有マップの計算
    grid_indices = self.cell_num * other_y_index + other_x_index
    grid_indices[~valid_indices] = -1  # 無効なインデックスのマーク

    occupancy_maps = torch.zeros((num_humans, 3, self.cell_num ** 2), dtype=torch.float, device=self.device)
    for i in range(num_humans):
        valid_grid_indices = grid_indices[i][valid_indices[i]]
        occupancy_maps[i, 0, valid_grid_indices.long()] = 1
        occupancy_maps[i, 1, valid_grid_indices.long()] = rel_velocities[i, :, 0][valid_indices[i]]
        occupancy_maps[i, 2, valid_grid_indices.long()] = rel_velocities[i, :, 1][valid_indices[i]]

    # 各チャネルの最初の16要素を選択して平滑化
    occupancy_maps = occupancy_maps[:, :, :16].reshape(num_humans, -1)

    return occupancy_maps

def build_occupancy_maps_batched(self, all_pedestrian_inputs):
    num_envs, num_humans, _ = all_pedestrian_inputs.size()

    # 全ての環境に対して相対位置と速度を計算
    rel_positions = all_pedestrian_inputs[:, :, None, :2] - all_pedestrian_inputs[:, None, :, :2]
    rel_velocities = all_pedestrian_inputs[:, :, None, 2:] - all_pedestrian_inputs[:, None, :, 2:]

    # 自己除外マスク
    self_mask = ~torch.eye(num_humans, dtype=torch.bool).to(self.device)
    self_mask = self_mask.unsqueeze(0).repeat(num_envs, 1, 1)

    # 角度と距離の計算
    human_velocity_angle = torch.atan2(all_pedestrian_inputs[:, :, 2], all_pedestrian_inputs[:, :, 3])
    other_human_orientation = torch.atan2(rel_positions[..., 1], rel_positions[..., 0])
    rotation = other_human_orientation - human_velocity_angle[:, :, None]
    distance = torch.norm(rel_positions, dim=-1)

    # 回転後の相対位置
    other_px = torch.cos(rotation) * distance
    other_py = torch.sin(rotation) * distance

    # グリッドインデックスの計算
    other_x_index = torch.floor(other_px / self.cell_size + self.cell_num / 2)
    other_y_index = torch.floor(other_py / self.cell_size + self.cell_num / 2)

    # 範囲外インデックスの処理
    valid_indices = (other_x_index >= 0) & (other_x_index < self.cell_num) & \
                    (other_y_index >= 0) & (other_y_index < self.cell_num)
    valid_indices &= self_mask

    # 占有マップの計算
    grid_indices = self.cell_num * other_y_index + other_x_index
    grid_indices[~valid_indices] = -1  # 無効なインデックスのマーク

    occupancy_maps_batched = torch.zeros((num_envs, num_humans, 3, self.cell_num ** 2), dtype=torch.float, device=self.device)
    for env_idx in range(num_envs):
        for human_idx in range(num_humans):
            valid_grid_indices = grid_indices[env_idx, human_idx][valid_indices[env_idx, human_idx]]
            occupancy_maps_batched[env_idx, human_idx, 0, valid_grid_indices.long()] = 1
            occupancy_maps_batched[env_idx, human_idx, 1, valid_grid_indices.long()] = rel_velocities[env_idx, human_idx, :, 0][valid_indices[env_idx, human_idx]]
            occupancy_maps_batched[env_idx, human_idx, 2, valid_grid_indices.long()] = rel_velocities[env_idx, human_idx, :, 1][valid_indices[env_idx, human_idx]]

    # 各チャネルの最初の16要素を選択して平滑化
    occupancy_maps_batched = occupancy_maps_batched[:, :, :, :16].reshape(num_envs, num_humans, -1)

    return occupancy_maps_batched


def pedestrian_intput_concat_old(occupancy_maps, pedestrian_inputs, robot_rotated_input, robot_input):
    def rotate(state, robot_input):
        human_states_selected = human_states[:, :, [0,1]]
        robot_input_selected = robot_input[:,:,[0,1]]
        robot_input_selected = robot_input_selected.repeat(1,5,1)
        
        da = torch.norm(human_states_selected - robot_input_selected, dim=2, keepdim=True)
        
        human_states_selected = human_states[:, :, [3]]
        robot_input_selected = robot_input[:,:,[4]]
        robot_input_selected = robot_input_selected.repeat(1,5,1)
        radius_sum = human_states_selected + robot_input_selected
        state[:,:,[11]] = da
        return torch.cat((state, radius_sum), dim=2)

    robot_rotated_input = robot_rotated_input.unsqueeze(1)
    stacked_robot_rotated_input = torch.cat([robot_rotated_input for _ in range(5)], dim=1).to("cuda:0")
    human_states = torch.cat((stacked_robot_rotated_input, pedestrian_inputs), dim=2)
    rotated_human_states = rotate(human_states, robot_input)
    return torch.cat([occupancy_maps, rotated_human_states], dim=2)


def pedestrian_intput_concat(occupancy_maps, pedestrian_inputs, robot_rotated_input, robot_input):
    def rotate(state, robot_input):
        # 位置と半径の計算をベクトル化
        human_positions = state[:, :, [0, 1]]
        robot_positions = robot_input[:, :, [0, 1]].expand(-1, 5, -1)
        da = torch.norm(human_positions - robot_positions, dim=2, keepdim=True)

        human_radius = state[:, :, [3]]
        robot_radius = robot_input[:, :, [4]].expand(-1, 5, -1)
        radius_sum = human_radius + robot_radius

        state[:, :, [11]] = da
        return torch.cat((state, radius_sum), dim=2)

    # ロボットの回転された入力を拡張
    stacked_robot_rotated_input = robot_rotated_input.unsqueeze(1).repeat(1, 5, 1).to("cuda:0")
    human_states = torch.cat((stacked_robot_rotated_input, pedestrian_inputs), dim=2)
    rotated_human_states = rotate(human_states, robot_input)
    return torch.cat([occupancy_maps, rotated_human_states], dim=2)

def mlp(input_dim, mlp_dims, last_relu=False, batch_norm=False, dropout=0.0):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    net = nn.Sequential(*layers)
    return net

# 例: He初期化の適用
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        pedestrian_input = obs['pedestrian_node'].float().to(device)
        obstacle_input = obs['angular_map'].float().to(device)
        robot_input = obs['robot_node'].float().to(device)

        occupancy_maps = build_occupancy_maps_batched(self, pedestrian_input)
        pedestrian_input.squeeze(0)
        pedestrian_input = pedestrian_intput_concat(occupancy_maps, pedestrian_input, robot_rotated_input, robot_input)
        pedestrian_feature1 = self.pedestrian_extractor1(pedestrian_input)
        pedestrian_feature2 = self.pedestrian_extractor2(pedestrian_feature1)
        global_state = torch.mean(pedestrian_feature1, keepdim=True, dim=1)
        global_state = global_state.expand((len(global_state),5, 100)).contiguous()
        attention_input = torch.cat((pedestrian_feature1, global_state), dim=2)

        scores = self.attention(attention_input).view(len(attention_input),5,1).squeeze(dim=2)
        weights = F.softmax(scores, dim=1)
        features = pedestrian_feature2.view(5, -1)
        weighted_feature = torch.sum(torch.mul(weights.unsqueeze(2), pedestrian_feature2), dim=1)
        obstacle_feature = self.obstacle_extractor1(obstacle_input)
        
        robot_rotated_input = robot_rotated_input.squeeze(0).squeeze(0)

        weighted_feature = weighted_feature.squeeze(0)
        # print(weighted_feature.size())
        # print(robot_rotated_input.size())
        # print(obstacle_feature.size())
        obstacle_feature = obstacle_feature.squeeze(0)
        if weighted_feature.dim() == 1:
            joint_state = torch.cat([robot_rotated_input, weighted_feature, obstacle_feature], dim=0)
        else :
            joint_state = torch.cat([robot_rotated_input, weighted_feature, obstacle_feature], dim=1)
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

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from crowd_sim import *
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import logging
import time
if __name__ == '__main__':
    
    # 訓練の設定
    timesteps = 1024*4
    n_envs=2

    # ロギングの設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    progress_bar = ProgressBarCallback(total_timesteps=timesteps/n_envs)

    env = make_vec_env('CrowdSim-v0', vec_env_cls=DummyVecEnv, n_envs=n_envs)
    # env = gym.make('CrowdSim-v0')
    # env = Monitor(env, filename='logs/')
    eval_env = gym.make('CrowdSim-v0')
    # tensorboard_log = 'logs/'
    # callback = EvalCallback(eval_env, best_model_save_path='logs/',
    #                      log_path='logs/', eval_freq=10000,
    #                      callback_on_new_best=None, verbose=1, deterministic=True)
    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=100),
    )
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,  tensorboard_log="./ppo_tensorboard/")
    print(prof)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    # model = PPO("MultiInputPolicy", env, verbose=1)

    start_time = time.time()

    model.learn(total_timesteps=timesteps, tb_log_name="first_run", callback=progress_bar)
    end_time = time.time()
    model.save("ppo_crowdnav")
    # 所要時間を計算
    elapsed_time = end_time - start_time

    # 所要時間をログに記録
    logging.info(f"Elapsed Time: {elapsed_time} seconds")
