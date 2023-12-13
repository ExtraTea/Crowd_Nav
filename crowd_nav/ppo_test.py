import gymnasium as gym
from stable_baselines3 import PPO
from crowd_sim import *
from matplotlib import pyplot as plt
import time
# 保存したモデルのファイルパス
# model_path = "ppo_crowdnav.zip"
model_path = "ppo_crowdnav_imitation.zip"

# モデルをロード
env = gym.make('CrowdSim-v0', render_mode='human')
model = PPO.load(model_path, env = env)

state, info  = env.reset()
done = False
tot_rew = 0
rews = 0
success = 0
for hoge in range(10):
    while True:
        env.render()
        action, _ = model.predict(state)
        state, rew, done,truncated, info = env.step(action)
        rews += rew
        if done:
            print(info, " reward: ",  rews)
            tot_rew += rews
            if info['event'] == 'reaching_goal':
                success += 1
            rews = 0
            state, info  = env.reset()
            break

print("success rate: ", success/500)
print("average reward: ", tot_rew/500)