import gym
import time
env = gym.make('FrozenLake-v0')
ob = env.reset()
print(ob)

actions = [0, 0, 0]
for i in range(9999):
    # for a in actions:
    # time.sleep(2)
    print('='*30)
    env.render()
    a = i % 4
    r = env.step(0)
    env.render()
    print('='*30)
    print(r)
    input()
