# File: run_agent.py
# Description: Running algorithm
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899



# Importing classes
# from env import Environment
"""
Environment
"""
import gym
custom_map = [
    'FHFFG',
#    'FHFFF',
    'FHFFF',
    'FHFHF',
    'FFFHF',
    'SFFHF'
]
env = gym.make("FrozenLake-v0", is_slippery=False, desc=custom_map)


"""
Iteration

    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    env.render()                         # print its state as well as the chosen action
    print("new_state:",new_state)
    print("reward:",reward)
    if done:
        break
"""


"""
Demo
"""
phi1 = "G[0,T](d_obs[t]>=1)";
d1 = [20,15,16,17,12,7,2,3,4]
rob1 = []
d2 = [20,15,16,11,12,7,8,3,4]
rob1 = []
num_d = 2

"""
initial Reward
"""

import numpy as np


from frozenlake_agent_brain import QLearningTable

def update():
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []
    
    # reward initialization from STL
    #reward_table = np.array([[-1, -2, 6, 7, 800], [0, -2,6,7,7],[1,-2,5,-2,6],[2,3,4,-2,5],[1,2,3,-2,4]])
    #reward_table = np.array([[0, 0, 3, 3.5, 4], [0, 0,2.5,0,0],[0,-1,2,0,0],[0.5,1,1.5,0,0],[0,0,0,0,0]])
    reward_table = np.array([[-9,-100,-1,0,1],[-8,-100,-2,-1,0],[-7,-100,-3,-100,-2],[-6,-5,-4,-100,-3],[-7,-6,-5,-100,-4]])
    rob1 = np.array([[-9,-100,-1,0,1],[-8,-100,-2,-1,0],[-7,-100,-3,-100,-2],[-6,-5,-4,-100,-3],[-7,-6,-5,-100,-4]])
    
    all_reward_sum = []

    for episode in range(5000):
        # Initial Observation
        observation = env.reset()
        rob1_pre = rob1[observation//5][observation%5]

        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0
        reward_sum = 0

        while i<100:
            # Refreshing environment
            env.render()

            # RL chooses action based on observation
            current_step = i+1
            action = RL.choose_action(str(observation),current_step)

            # RL takes an action and get the next observation and reward
            observation_, reward, done, info = env.step(action)
            
            # access new reward from reward_table
            hang = observation_ // 5
            lie = observation_ % 5
            
            rob1_now = rob1[hang][lie]
            rob1_now = min(rob1_now,rob1_pre)
            rob1_pre = rob1_now
            reward_table[hang][lie] = reward_table[hang][lie] + rob1_now
            
            reward_new = reward_table[hang][lie]
            reward_sum = reward_sum +reward_new
            

            # RL learns from this transition and calculating the cost
            cost += RL.learn(str(observation), action, reward_new, str(observation_))

            # Swapping the observations - current and next
            observation = observation_

            # Calculating number of Steps in the current Episode
            i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                all_reward_sum += [reward_sum]
                break

    # Showing the final route
    # env.final()

    # Showing the Q-table with values for each action
    # RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs,all_reward_sum)
    
    # Print the evaluation
    print()
    print("Final result:")
    
    custom_map2 = [
        'FHFFG',
        'FHFFF',
        'FHFHF',
        'FFFHF',
        'SFFHF'
    ]
    env2 = gym.make("FrozenLake-v0", is_slippery=False, desc=custom_map2)
    observation = env2.reset()
    env2.render()
    
    i = 0
    reward_sum = 0
    while i<100:
        # RL chooses action based on observation
        #current_step = i+1
        #action = RL.choose_action(str(observation),current_step)
        
        # Checking if the state exists in the table
        RL.check_state_exist(str(observation))
        # Choosing the best action
        state_action = RL.q_table.loc[str(observation), :]
        state_action = state_action.reindex(np.random.permutation(state_action.index))
        action = state_action.idxmax()


        # RL takes an action and get the next observation and reward
        observation_, reward, done, info = env2.step(action)
	    
        # access new reward from reward_table
        hang = observation_ // 5
        lie = observation_ % 5
        reward_new = reward_table[hang][lie]
        reward_sum = reward_sum +reward_new

        # RL learns from this transition and calculating the cost
        cost += RL.learn(str(observation), action, reward_new, str(observation_))

        # Swapping the observations - current and next
        observation = observation_

        # Calculating number of Steps in the current Episode
        i += 1
        
        # Refreshing environment
        env2.render()

        # Break while loop when it is the end of current Episode
        # When agent reached the goal or obstacle
        if done:
            break
            
    print("ave_reward = ",reward_sum/current_step)
    print("cost = ",cost)




# Commands to be implemented after running this file
if __name__ == "__main__":
    # Calling for the environment
   # env = Environment()
    # Calling for the main algorithm
    RL = QLearningTable(actions=list(range(4)))
    # Running the main loop with Episodes by calling the function update()
    update()
