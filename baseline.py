import gym
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from taxi_env import *
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy


# Create log dir
log_dir = "chk/"
os.makedirs(log_dir, exist_ok=True)


# env = gym.make('CartPole-v1')
env = gym.make('Taxi-v3')

env = Monitor(env, log_dir, allow_early_resets=True)

#env = TaxiEnv()
episode_log = []
reward_log = []
time_steps = 2e5


model = DQN(MlpPolicy, env, verbose=2)
model.learn(total_timesteps=int(time_steps))
# model.save("deepq_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")

obs = env.reset()
reward = 0
for i in range(50):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    reward += rewards
    env.render()
    if dones:
        break
print(reward)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DQN")
plt.show()