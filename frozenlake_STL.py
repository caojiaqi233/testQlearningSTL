"""
reward: didn't use the reward from the evn
action: LEFT = 0  DOWN = 1  RIGHT = 2  UP = 3
state:    '0  1  2  3  4',
          '5  6  7  8  9',
          '10 11 12 13 14',
          '15 16 17 18 19',
          '20 21 22 23 24'

episode ends : when reach the goal or fall in a hole
"""

# loading the Gym library
import gym     

# to change the map
custom_map = [
    'FHFFG',
    'FHFFF',
    'FHFHF',
    'FFFHF',
    'SFFHF'
]

# open the environment with deterministic mode
env = gym.make("FrozenLake-v0", is_slippery=False, desc=custom_map)

# print the information
env.reset()                  
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)
env.render()                             # print its state

# run the environment with random action
MAX_ITERATIONS = 10
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    env.render()                         # print its state as well as the chosen action
    print("new_state:",new_state)
    print("reward:",reward)
    if done:
        break

"""      
#env.step(env.action_space.sample())     # take a random action
"""

# close the environment
env.close()




