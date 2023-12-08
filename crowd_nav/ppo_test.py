import gymnasium as gym
from stable_baselines3 import PPO
from crowd_sim import *
from matplotlib import pyplot as plt
import time
# 保存したモデルのファイルパス
model_path = "ppo_crowdnav.zip"
model_path = "ppo_crowdnav_imitation.zip"

# モデルをロード
model = PPO.load(model_path)
env = gym.make('CrowdSim-v0', render_mode='human')
state, info  = env.reset()
done = False
while True:
    
    env.render()
    action, _ = model.predict(state)
    state, rew, done,truncated, info = env.step(action)
    if done:
        print(info)
        time.sleep(3)
        state, info  = env.reset()