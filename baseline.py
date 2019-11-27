import gym
import tensorflow as tf

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from taxi_env import *

# env = gym.make('CartPole-v1')
# env = gym.make('Taxi-v3')
env = TaxiEnv()

model = DQN(MlpPolicy, env, verbose=2)
model.learn(total_timesteps=200000)
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