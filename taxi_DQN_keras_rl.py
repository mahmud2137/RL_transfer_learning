from tensorflow import keras
### hack tf-keras to appear as top level keras
import sys
sys.modules['keras'] = keras

import gym
import random
import numpy as np
from keras.layers import Dense, Flatten, Embedding, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from taxi_env import *
import tensorflow as tf
print(tf.__version__)


from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = TaxiEnv()
action_size = env.action_space.n
state_size = env.observation_space.n

model = Sequential()
model.add(Embedding(500, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(action_size, activation='linear'))
print(model.summary())



memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=action_size, memory=memory, nb_steps_warmup=500, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=100000)