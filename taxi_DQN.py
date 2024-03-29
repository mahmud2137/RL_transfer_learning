import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from taxi_env import *
from taxi_env_L import *
from collections import deque
import matplotlib.pyplot as plt
import time

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        # state_shape  = self.env.observation_space.shape
        # state_shape = 4
        model.add(Dense(4, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation='softmax'))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 16
        if len(self.memory) < batch_size: 
            return
        
        samples = random.sample(self.memory, batch_size)
        X_batch, y_batch = [], []
        for sample in samples:
            
            state, action, reward, new_state, done = sample
            
            
            target = self.target_model.predict(state.reshape(1,-1))
            
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape(1,-1))[0])
                target[0][action] = reward + Q_future * self.gamma
            X_batch.append(list(state))
            y_batch.append(target[0])
            # self.model.fit(state.reshape(1,-1), target, epochs=1, verbose=0)

        
        self.model.fit(np.array(X_batch), np.array(y_batch), batch_size=len(X_batch), verbose=0)


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


    
if __name__ == "__main__":

    env = TaxiEnv()
    
    # env  = gym.make('MsPacman-ram-v0')
    # gamma   = 0.9
    # epsilon = .95

    episodes  = 100
    trial_len = 200
    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    rewards = []
    
    for trial in range(episodes):
        # cur_state = env.reset().reshape(1,2)
        start = time.time()
        cur_state = np.array(list(env.decode(env.reset())))
        
        trial_reward = 0
        for step in range(trial_len):
        
            action = dqn_agent.act(cur_state.reshape(1,-1))
            
            
            new_state, reward, done, _ = env.step(action)
            
            trial_reward += reward
            # reward = reward if not done else -20
            
            new_state = np.array(list(env.decode(new_state)))
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model
            cur_state = new_state
            if done:
                break

        rewards.append(trial_reward)

        if step >= 199:
            # if trial % 100 == 0:
            print("Failed to complete in episode {}".format(trial))
            # if step % 10 == 0:
            #     dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} episode".format(trial))
            dqn_agent.save_model("success.model")
            break

        total_time = time.time() - start
        print('Total Time:', total_time)
    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    size = 5
    chunks = list(chunk_list(rewards, size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]
    plt.Figure(figsize=(15,10))
    plt.plot(range(0, len(rewards), size), averages)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.show()
    plt.savefig('reward_curve.png')