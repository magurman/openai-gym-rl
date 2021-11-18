import gym
import numpy as np
env = gym.make('MountainCar-v0')
from collections import defaultdict
import math

Q = defaultdict( lambda: defaultdict(lambda:0) )

alpha = 0.1
gamma = 0.9
num_of_episodes = 5000
epsilon_start = 0.5
epsilon_end = 0.01
annealing_time =  num_of_episodes/15
buckets=(3, 3, 6, 6)

def choose_action(state, episode):
    action=0
    epsilon = max( epsilon_start - ((epsilon_start - epsilon_end)/annealing_time)*episode, epsilon_end)
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:

        if len(Q[state]) == 0:
            return 0
        action = max(Q[state], key=Q[state].get)
        
        
    return action

def discretize(observation):

    # print(observation)
    position = round(observation[0], 1)
    velocity = round(observation[1], 2)
    return (position, velocity)

for i_episode in range(num_of_episodes):

    observation = discretize(env.reset())
    # print('observation:', observation)
    action = choose_action(observation, i_episode )
    # print('action: ', action)
    done = False
    action_count = 0
    print(i_episode)
    while(not done):
        
        action_count += 1
        if i_episode > 3000 :
            env.render()
        observation_prime, reward, done, info = env.step(action)
        observation_prime = discretize(observation_prime)
        # print(observation_prime)
        if observation_prime[0] >= 0.5:
            Q[observation][action] = 0
            # print('Good game')
            print('action count:', action_count)
            # print('Episode:', i_episode)
            break 
            
        action_prime = choose_action(observation_prime, i_episode)
        Q[observation][action] = Q[observation][action] + alpha*(reward + gamma*(Q[observation_prime][action_prime] - Q[observation][action])) 
        observation = observation_prime
        action = action_prime
        
    # # print("Episode finished after {} timesteps".format(action_count+1))
    # if action_count > 199: 
    #     # print(action_count)
    #     print(i_episode)
        
env.close()