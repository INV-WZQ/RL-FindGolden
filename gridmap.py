import logging
import numpy as np
import random
from gym import spaces
import gym

class GridMap(gym.Env):

    def __init__(self):
        #4*4=16 states
        self.states=[i for i in range(16)]
        self.actions=['up', 'down', 'right', 'left']
        
        #位置
        self.x = []
        self.y = []
        for x in [150, 250, 350, 450]:
            for y in [150, 250, 350, 450]:
                self.x.append(x)
                self.y.append(y)

        #终止状态
        self.terminate_states = dict()
        for i in [0, 2, 7, 9]:
            self.terminate_states[i] = 1

        #Define Rewards
        self.rewards=dict()
        for i in range(4):
            for j in range(4):
                pos = i*4+j         
                if pos in [0, 2, 7, 9]: 
                    continue

                if i>0:
                    self.rewards[f'{pos}_up'] = -1.
                    if (pos in [4, 11, 13]):
                        self.rewards[f'{pos}_up'] = -10.
                if i<3:
                    self.rewards[f'{pos}_down'] = -1.
                    if (pos in [3, 5]):
                        self.rewards[f'{pos}_down'] = -10.
                if j<3:
                    self.rewards[f'{pos}_right'] = -1.
                    if (pos in [6, 8]):
                        self.rewards[f'{pos}_right'] = -10.
                if j>0:
                    self.rewards[f'{pos}_left'] = -1.
                    if (pos in [1, 10]):
                        self.rewards[f'{pos}_left'] = -10.
        
        self.rewards['1_right'] = 1.
        self.rewards['3_left'] = 1.
        self.rewards['6_up'] = 1.

        #转移概率



        self.gamma = 0.9
        self.viewer = None#?
        self.state = None #?

    

    def info_print(self):
        print("states:",self.states)
        print("actions:",self.actions)
        print("rewards:",self.rewards)
        print(self.x)
        print(self.y)


test=GridMap()
test.info_print()