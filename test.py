import gym
#CartPole-v0
env=gym.make('Find-Golden', render_mode="human")
observation, info = env.reset()
env.render()

import time
time.sleep(5)
env.close()