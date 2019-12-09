
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
# from stable_baselines.results_plotter import load_results, ts2xy
from plot_result import *

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[32,32],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def demo(env, model, episodes = 10, max_step = 50, render = False):
    
    sum_reward = 0
    for e in range(episodes):
        ep_reward = 0
        obs = env.reset()
        for s in range(max_step):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            ep_reward += rewards
            if render:
                env.render()
            if dones:
                break
        sum_reward += ep_reward
        # print(ep_reward,'\n')
    mean_reward  = sum_reward/episodes
    return mean_reward

# Create log dir
log_dir_s = "run2/source/"
log_dir_t = "run2/target/"
log_dir_t_scratch = "run2/target_scratch/"
os.makedirs(log_dir_s, exist_ok=True)
os.makedirs(log_dir_t, exist_ok=True)
os.makedirs(log_dir_t_scratch, exist_ok=True)

#############################
#Learning Taxi_L from Scratch
env_t = taxi_env_L_state_array.TaxiEnv(max_step = 50)
env_t = Monitor(env_t, log_dir_t_scratch, allow_early_resets=True)

time_steps_t = 20e5
model_t = DQN(MlpPolicy, env_t, exploration_fraction=0.3, exploration_final_eps=0.05, verbose=2)
model_t.learn(total_timesteps=int(time_steps_t))

################################
#Learning the Source Environment

env_s = taxi_env_state_array.TaxiEnv(max_step=50)
env_s = Monitor(env_s, log_dir_s, allow_early_resets=True)
time_steps = 20e5

model_s = DQN(MlpPolicy, env_s, exploration_fraction=0.3, exploration_final_eps=0.05, verbose=2)
model_s.learn(total_timesteps=int(time_steps), log_interval=100)

# demo(env_s, model_s, episodes=100)

# model_s.save("model_state_source")

# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")
#####################################
#Learning the Target Env with Source policy

env_t = taxi_env_L_state_array.TaxiEnv(max_step = 50)
env_t = Monitor(env_t, log_dir_t, allow_early_resets=True)
tr_policy = model_s.policy
time_steps_t = 20e5
model_t = DQN(tr_policy, env_t, exploration_fraction=0.3, exploration_final_eps=0.1, verbose=2)
model_t.learn(total_timesteps=int(time_steps_t))
 
mean_rwd = demo(env_t, model_t, episodes=1, render=True)
print(mean_rwd)

plot_results([log_dir_t, log_dir_t_scratch], int(20e5), results_plotter.X_EPISODES, "DQN", ['Transfered from Source Task', 'Learning From Scratch'])
plt.show()
plt.savefig('source_to_target_learning_curve.png')


