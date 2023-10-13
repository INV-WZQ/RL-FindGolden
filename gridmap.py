
import random
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pygame
from pygame import gfxdraw as gfx
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils, rendering
from gym.error import DependencyNotInstalled



class GridMap(gym.Env[np.ndarray, Union[int, np.ndarray]]):# CartPoleEnv类的观测空间是一个NumPy数组，动作空间可以是整数或NumPy数组。
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    '''
    metadata字典包含有关环境的信息，例如渲染模式和视频渲染的每秒帧数。
    human模式用于在屏幕上渲染环境，而rgb_array模式用于将环境渲染为NumPy数组的图像。
    video.frames_per_second参数指定了视频渲染的每秒帧数。
    '''
    
    def __init__(self, render_mode: Optional[str] = None):
        
        self.render_mode = render_mode
        #4*4=16 states
        self.states=[i for i in range(16)]
        self.actions=['up', 'down', 'right', 'left']
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Discrete(len(self.states))
        #位置
        self.x = []
        self.y = []
        for y in [450, 350, 250, 150]:
            for x in [150, 250, 350, 450]:
                self.x.append(x)
                self.y.append(y)

        #终止状态
        self.terminate_states = dict()
        for i in [5, 11, 12 ,14]:
            self.terminate_states[i] = 1

        self.rewards = dict() #不同状态下采取动作的奖励
        self.t = dict()       #采取动作的下一个位置
        for i in range(4):
            for j in range(4):
                pos = i*4+j         
                if pos in [5, 11, 12 ,14]: 
                    continue

                if i>0:
                    self.t[f'{pos}_up'] = pos-4
                    self.rewards[f'{pos}_up'] = -1.
                    if (pos in [9, 15]):
                        self.rewards[f'{pos}_up'] = -10.
                else :
                    self.t[f'{pos}_up'] = pos
                    self.rewards[f'{pos}_up'] = -10.
                
                if i<3:
                    self.t[f'{pos}_down'] = pos+4
                    self.rewards[f'{pos}_down'] = -1.
                    if (pos in [1, 7, 8]):
                        self.rewards[f'{pos}_down'] = -10.
                else:
                    self.t[f'{pos}_down'] = pos
                    self.rewards[f'{pos}_down'] = -10.

                if j<3:
                    self.t[f'{pos}_right'] = pos+1
                    self.rewards[f'{pos}_right'] = -1.
                    if (pos in [4, 10]):
                        self.rewards[f'{pos}_right'] = -10.
                else:
                    self.t[f'{pos}_right'] = pos
                    self.rewards[f'{pos}_right'] = -10.

                if j>0:
                    self.t[f'{pos}_left'] = pos-1
                    self.rewards[f'{pos}_left'] = -1.
                    if (pos in [6, 13]):
                        self.rewards[f'{pos}_left'] = -10.
                else:
                    self.t[f'{pos}_left'] = pos
                    self.rewards[f'{pos}_left'] = -10.
        #到达目标奖励
        self.rewards['13_right'] = 1.
        self.rewards['15_left'] = 1.
        self.rewards['10_down'] = 1.

        self.state = None 
        self.screen = None 
        self.clock = None
    
    #初始化
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        while True:#初始化不能在障碍和目标处
            self.state = self.states[int(random.random() * len(self.states))]#随机找状态
            if self.state not in self.terminate_states:
                break
        
        if self.render_mode == "human":
            self.render()
        
        return self.state, {}
    
    def setAction(self, s):
        self.state = s
    
    def step(self, action):    #return observation, reward, terminate, done, info
        action = self.actions[int(action)]          #action是数字，得对应到具体的actions
        err_msg = f"{action!r} ({type(action)}) invalid" 
        assert action in self.actions, err_msg
        assert self.state is not None, 'Call reset before using step method'
        
        state = self.state      
        if state in self.terminate_states:
            return state, 0, True, {}
        key = f'{state}_{action}'
        #转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = bool(
            next_state in self.terminate_states
        )
        #奖励
        reward = 0
        if key in self.rewards:
            reward = self.rewards[key]

        if self.render_mode == "human":
            self.render()
        
        return next_state, reward, is_terminal, False, {}
        #False是在返回结果中的倒数第二个元素，表示当前时间步骤是否处于"done"状态，即是否已经完成了一个回合
    
    #画图
    def render(self, mode='human', close=False):
        
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        #显示窗口高宽
        screen_width = 600
        screen_height = 600
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else :
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
            #通过使用Clock对象，可以确保游戏在每秒钟渲染的帧数保持一致，从而使游戏画面更加平滑。

        #创建环境
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        #水平线hline(surface, x1, x2, y, color)
        gfx.hline(self.surf, 100, 500, 100, (0,0,0))
        gfx.hline(self.surf, 100, 500, 200, (0,0,0))
        gfx.hline(self.surf, 100, 500, 300, (0,0,0))
        gfx.hline(self.surf, 100, 500, 400, (0,0,0))
        gfx.hline(self.surf, 100, 500, 500, (0,0,0))
        #垂直线vline(surface, x, y1, y2, color)
        gfx.vline(self.surf, 100, 100, 500, (0,0,0))
        gfx.vline(self.surf, 200, 100, 500, (0,0,0))
        gfx.vline(self.surf, 300, 100, 500, (0,0,0))
        gfx.vline(self.surf, 400, 100, 500, (0,0,0))
        gfx.vline(self.surf, 500, 100, 500, (0,0,0))
        
        #绘制填充矩形box(surface, rect, color)
        #rec=(x, y, width, height)
        gfx.box(self.surf, (100,100,100,100), (100,0,0))
        gfx.box(self.surf, (400,200,100,100), (0,100,0))
        gfx.box(self.surf, (200,300,100,100), (0,0,100))
        gfx.box(self.surf, (300,100,100,100), (255,215,0))
        
        #绘画圆circle(surface, x, y, r, color)
        gfx.filled_circle(self.surf, self.x[self.state], self.y[self.state], 50, (222,222,100))
        
        
        #self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        import time
        time.sleep(0.5)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def info_print(self):
        print("states:",self.states)
        print("actions:",self.actions)
        print("rewards:",self.rewards)
        print(self.x)
        print(self.y)
