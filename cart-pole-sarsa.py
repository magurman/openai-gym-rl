import gym
import numpy as np
env = gym.make('CartPole-v0')
from collections import defaultdict
import math

# q = {(1,2,3,4): {0: 50, 1: 40}, (2,3,4,5): {0: 50, 1: 40}}
# Q = {}

Q = defaultdict( lambda: defaultdict(lambda:0) )
# Q[tuple([-0.02744534 , 0.0211086 ,  0.03182621 , 0.01719197])][0] = Q[tuple([-0.02744534 , 0.0211086 ,  0.03182621 , 0.01719197])][0]
# print(Q[tuple([-0.02744534 , 0.0211086 ,  0.03182621 , 0.01719197])][0])
# print('Action Space: ', env.action_space)
# print('Observation Space: ', env.observation_space)

alpha = 0.1
gamma = 0.9
num_of_episodes = 5000
epsilon_start = 1
epsilon_end = 0.01
annealing_time =  num_of_episodes/8
buckets=(3, 3, 6, 6)

def discretize(observation):

    discretized = []
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]
    for i in range(len(observation)):
        scaling = ((observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]))
        new_obs = int(round((buckets[i] - 1) * scaling))
        new_obs = min(buckets[i] - 1, max(0, new_obs))
        discretized.append(new_obs)
    return tuple(discretized)

def choose_action(state, episode):
    action=0
    epsilon = max( epsilon_start - ((epsilon_start - epsilon_end)/annealing_time)*episode, epsilon_end)
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        # action = np.argmax(Q[state, :])
        # print('length:', len(Q[state]))
        if len(Q[state]) == 0:
            return 0
        action = max(Q[state], key=Q[state].get)
        
        # for k in Q[tuple(state)]:
            # print('k:', k)
        
    return action

for i_episode in range(num_of_episodes):
    observation = discretize(env.reset())
    # print('observation:', observation)
    action = choose_action(observation, i_episode )
    # print('action: ', action)
    done = False
    action_count = 0
    while(not done):
        action_count += 1
        if i_episode > 3000 :
            env.render()
        observation_prime, reward, done, info = env.step(action)
        observation_prime = discretize(observation_prime)
        action_prime = choose_action(observation_prime, i_episode)
        Q[observation][action] = Q[observation][action] + alpha*(reward + gamma*(Q[observation_prime][action_prime] - Q[observation][action])) 
        observation = observation_prime
        action = action_prime
        
    # print("Episode finished after {} timesteps".format(action_count+1))
    if action_count > 199: 
        # print(action_count)
        print(i_episode)
        
env.close()


