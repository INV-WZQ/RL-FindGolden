import logging
import numpy as np
import random
from gym import spaces
import gym
from gym.envs.classic_control import utils
from gym.envs.classic_control import rendering

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

        self.rewards = dict() #不同状态下采取动作的奖励
        self.t = dict()       #采取动作的下一个位置
        for i in range(4):
            for j in range(4):
                pos = i*4+j         
                if pos in [0, 2, 7, 9]: 
                    continue

                if i>0:
                    self.t[f'{pos}_up'] = pos-4
                    self.rewards[f'{pos}_up'] = -1.
                    if (pos in [4, 11, 13]):
                        self.rewards[f'{pos}_up'] = -10.
                if i<3:
                    self.t[f'{pos}_down'] = pos+4
                    self.rewards[f'{pos}_down'] = -1.
                    if (pos in [3, 5]):
                        self.rewards[f'{pos}_down'] = -10.
                if j<3:
                    self.t[f'{pos}_right'] = pos+1
                    self.rewards[f'{pos}_right'] = -1.
                    if (pos in [6, 8]):
                        self.rewards[f'{pos}_right'] = -10.
                if j>0:
                    self.t[f'{pos}_left'] = pos-1
                    self.rewards[f'{pos}_left'] = -1.
                    if (pos in [1, 10]):
                        self.rewards[f'{pos}_left'] = -10.
        
        self.rewards['1_right'] = 1.
        self.rewards['3_left'] = 1.
        self.rewards['6_up'] = 1.

        self.gamma = 0.9
        self.viewer = None#?
        self.state = None #?

    def reset(self):
        self.state = self.states[int(random.random() * len(self.states))]
        return self.state
    
    def setAction(self, s):
        self.state = s
    
    def step(self, action):    #return observation, reward, done, info
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}
        key = f'{state}_{action}'

        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False
        if next_state in self.terminate_states:
            is_terminal = True
        if key not in self.rewards:
            r = 0
        else :
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #创建地图
            self.line1 = rendering.Line((100,100),(500,100))
            self.line2 = rendering.Line((100,200),(500,200))
            self.line3 = rendering.Line((100,300),(500,300))
            self.line4 = rendering.Line((100,400),(500,400))
            self.line5 = rendering.Line((100,500),(500,500))
            self.line6 = rendering.Line((100,100),(100,500))
            self.line7 = rendering.Line((200,100),(200,500))
            self.line8 = rendering.Line((300,100),(300,500))
            self.line9 = rendering.Line((400,100),(400,500))
            self.line10= rendering.Line((500,100),(500,500))
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            #创建block
            self.block1 = rendering.FilledPolygon([
                (100,100), (200,100),(200,200),(100,200)
            ])
            self.block2 = rendering.FilledPolygon([
                (400,200),(500,200),(500,300),(400,300)
            ])
            self.block3 = rendering.FilledPolygon([
                (200,300),(300,300),(300,400),(200,400)
            ])
            self.golden = rendering.FilledPolygon([
                (300,100),(400,100),(400,200),(300,200)
            ])
            self.block1.set_color(0, 0, 0)
            self.block2.set_color(0, 0, 0)
            self.block3.set_color(0, 0, 0)
            self.golden.set_color(255, 215, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.block1)
            self.viewer.add_geom(self.block2)
            self.viewer.add_geom(self.block3)
            self.viewer.add_geom(self.golden)

            if self.state is None: 
                return None

            self
            return self.viewer.render(return_rgb_array=mode == 'human')

    def info_print(self):
        print("states:",self.states)
        print("actions:",self.actions)
        print("rewards:",self.rewards)
        print(self.x)
        print(self.y)

#https://zhuanlan.zhihu.com/p/485631527
try:
    test=GridMap()
    test.info_print()
    test.reset()
    test.render()
finally:
    import time
    time.sleep(2)
    test.close()