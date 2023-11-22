import gymnasium as gym
from stable_baselines3 import PPO
from crowd_sim import *
from matplotlib import pyplot as plt
# 保存したモデルのファイルパス
model_path = "ppo_crowdnav.zip"

# モデルをロード
model = PPO.load(model_path)
env = gym.make('CrowdSim-v0', render_mode='human')
state, info  = env.reset()
done = False
while True:
    env.render()
    action, _ = model.predict(state)
    state, rew, done,truncated, info = env.step(action)
    plt.close()
    if done:
        state, info  = env.reset()
    
        

env.close()