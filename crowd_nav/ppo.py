import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from crowd_sim import *
import numpy as np
from tqdm import tqdm
import logging
import time
from imitation.data import rollout
from imitation.algorithms.bc import BC
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.data.types import Transitions
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

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
        hardcoded_original_shapes = [
            (6,),       # array1の形状
            (5, 6),     # array2の形状(human_num, 6)
            (72,),      # array3の形状
            (1, 6),     # array4の形状
        ]

        def reshape_to_original(combined_array, original_shapes):
            arrays = []
            start = 0
            for shape in original_shapes:
                size = np.prod(shape)
                array = combined_array[start:start+size].reshape(shape)
                arrays.append(array)
                start += size
            return arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # robot_rotated_input = obs['robot_rotated_node'].float().to(device)
        # pedestrian_input = obs['pedestrian_node'].float().to(device)
        # obstacle_input = obs['angular_map'].float().to(device)
        # robot_input = obs['robot_node'].float().to(device)
        
        numpy_array = obs.cpu().numpy()
        reshaped_batches = []
        for batch in numpy_array:
            reshaped_batches.append(reshape_to_original(batch, hardcoded_original_shapes))

        # 各形状ごとに配列をスタック
        final_arrays = []
        for i in range(len(hardcoded_original_shapes)):
            stacked_array = np.stack([batch[i] for batch in reshaped_batches])
            final_arrays.append(stacked_array)

        # print(final_arrays[0].shape)

        # 形状の確認
        final_shapes = [arr.shape for arr in final_arrays]
        robot_rotated_input = torch.from_numpy(final_arrays[0]).to(device)
        pedestrian_input = torch.from_numpy(final_arrays[1]).to(device)
        # print(pedestrian_input.size())
        obstacle_input = torch.from_numpy(final_arrays[2]).to(device)
        robot_input = torch.from_numpy(final_arrays[3]).to(device)

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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 専門家のデータセットをロード
    expert_dataset = torch.load("expert_dataset.pt")
    expert_obs, expert_act, expert_done, expert_info = expert_dataset['observations'], expert_dataset['actions'], expert_dataset['dones'], expert_dataset['infos']

    # 環境の設定
    env = make_vec_env('CrowdSim-v0', vec_env_cls=SubprocVecEnv, n_envs=8)

    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=100),
    )

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    all_obs = np.concatenate(expert_obs)
    all_acts = np.concatenate(expert_act)
    all_dones = np.concatenate(expert_done)
    # print(all_obs.shape)
    # print(all_acts.shape)
    # print(all_dones.shape)
    # print(all_dones)
    all_next_obs = np.concatenate([all_obs[1:], np.zeros_like(all_obs[0:1])])
    all_transitions = Transitions(
        obs=torch.tensor(all_obs).to('cpu'),
        acts=torch.tensor(all_acts).to('cpu'),
        next_obs=torch.tensor(all_next_obs).to('cpu'),
        dones=all_dones,
        infos=[{} for _ in range(len(all_obs))]
    )
    reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
    )

    gail_trainer = GAIL(
        demonstrations=all_transitions,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=model,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    timesteps = 1024 * 1024 * 16
    checkpoint_callback = CheckpointCallback(
                        save_freq = max(1024*128 // 8, 1),
                        save_path="./logs/",
                        name_prefix="rl_model",
                        save_replay_buffer=True,
                        save_vecnormalize=True,
                        )
    progress_bar_callback = ProgressBarCallback(timesteps=timesteps / 8) 
    callbacks = [checkpoint_callback, progress_bar_callback]

    gail_trainer.policy.to(device)
    gail_trainer.train(timesteps, callbacks=callbacks)
    trained_model = gail_trainer.gen_algo
    model = trained_model
    model.save("ppo_crowdnav_imitation")

    trained_model = gail_trainer.gen_algo
    model = trained_model
    model.save("ppo_crowdnav_imitation")
    print("imitation learning done")
    # PPOでの追加トレーニング
    

    env = make_vec_env('CrowdSim-v0', vec_env_cls=SubprocVecEnv, n_envs=8)
    model = PPO.load("ppo_crowdnav_imitation.zip", env=env)
    
    timesteps = 1024 * 1024 * 16
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    checkpoint_callback = CheckpointCallback(
                        save_freq = max(1024*128 // 8, 1),
                        save_path="./logs/",
                        name_prefix="rl_model",
                        save_replay_buffer=True,
                        save_vecnormalize=True,
                        )
    progress_bar_callback = ProgressBarCallback(total_timesteps=timesteps / 8) 
    callbacks = [checkpoint_callback, progress_bar_callback]
    
    start_time = time.time()
    model.learn(total_timesteps=int(timesteps), tb_log_name="first_run", callback=callbacks)
    model.save("ppo_crowdnav")

        
    end_time = time.time()
    

    # 所要時間の計算とログ記録
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed Time: {elapsed_time} seconds")

