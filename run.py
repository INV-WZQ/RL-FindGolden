import gym
import qlearning
import matplotlib.pyplot as plt
import random
import numpy as np
#random.seed(0)

env = gym.make('Find-Golden', render_mode="rgb_array")
Q = qlearning.Qlearning(env, 0.9, 0.25, 0.15)
return_list = Q.learning(20000)

sample = [min(return_list[i:i+20]) for i in range(0,20000,20)]
episodes_list = list(range(len(sample)))
plt.plot(episodes_list, sample)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))
plt.show()

print("-------5个test--------")
for i in range(5):
    env = gym.make('Find-Golden', render_mode="human")
    observation, info = env.reset()
    sum = 0
    for _ in range(100):
        env.render()
        next_state, reward, terminate, done, info = env.step(np.argmax(Q.Q_table[observation]))
        #print(observation,next_state, np.argmax(Q.Q_table[observation]), reward)
        observation = next_state
        sum+=reward

        if terminate:
            break
    import time
    time.sleep(1)
    print(f"test{i} 的 reward：",sum)
    
