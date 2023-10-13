import numpy as np
import gym
from tqdm import tqdm
import random

#random.seed(0)

class Qlearning:
    def __init__(self, env, gamma, learning_rate, epsilon):
        self.env = env
        self.Q_table = np.zeros([len(env.states), len(env.actions)])
        self.nactions = len(env.actions)
        self.gamma = gamma
        self.lr = learning_rate
        self.epsilon = epsilon

    def take_action(self, state):
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.nactions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self, reward, state, action, next_state):
        TD_error = reward + self.gamma * self.Q_table[next_state].max() - self.Q_table[state, action]
        self.Q_table[state, action]+= self.lr * TD_error

    def learning(self, epochs):
        return_list = []
        for i in range(10):
            with tqdm(total=int(epochs/10), desc='Iteration %d' % i) as pbar:
                for j in range(int(epochs/10)):
                    episode_reture = 0
                    observation, info = self.env.reset()
                    terminate = False
                    while not terminate:
                        action = self.take_action(observation)
                        next_state, reward, terminate, done, info = self.env.step(action)
                        self.update(reward, observation, action, next_state)
                        observation = next_state
                        episode_reture+= reward
                    
                    return_list.append(episode_reture)

                    if (j + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                            'episode':
                            '%d' % (epochs / 10 * i + j + 1),
                            'return':
                            '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list





                
                
