
import taxi_env
import taxi_env_L
import taxi_env_state_array
import taxi_env_L_state_array

import gym
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import DQN, ACKTR
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 64, 64],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def demo(env, model):
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

# Create log dir
log_dir_s = "chk/source/"
log_dir_t = "chk/target/"
os.makedirs(log_dir_s, exist_ok=True)
os.makedirs(log_dir_t, exist_ok=True)

env_s = taxi_env_state_array.TaxiEnv(max_step=50)

env_s = Monitor(env_s, log_dir_s, allow_early_resets=True)

episode_log = []
reward_log = []
time_steps = 2e5

model_s = DQN(MlpPolicy, env_s, verbose=2)
model_s.learn(total_timesteps=int(time_steps), log_interval=100)

demo(env_s, model_s)

# model_s.save("model_state_source")

# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")

env_t = taxi_env_L_state_array.TaxiEnv(max_step = 50)
env_t = Monitor(env_t, log_dir_t, allow_early_resets=True)
tr_policy = model_s.policy

model_t = DQN(tr_policy, env_t, verbose=2)
model_t.learn(total_timesteps=int(5e5))

demo(env_t, model_t)

results_plotter.plot_results([log_dir_s,log_dir_t], int(3e5), results_plotter.X_EPISODES, "DQN")
plt.show()