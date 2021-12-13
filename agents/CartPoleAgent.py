import gym 
import collections
import random
import numpy as np
import math
import time
import statistics
from agents.AbstractQLearningAgent import AbstractQLearningAgent

class CartPoleAgent(AbstractQLearningAgent):

    def __init__(self, algorithm, alphaStart = 0.05, alphaEnd = 0.05, epsilonStart = 1, epsilonEnd = 0.01, gamma = 0.95, annealingTime=5000/8, numEpisodes = 5000, env='CartPole-v0', discretizeSize=[3,3,6,6]):
        super(CartPoleAgent, self).__init__(algorithm, alphaStart, alphaEnd, epsilonStart, epsilonEnd, gamma, annealingTime, numEpisodes, env)

        self.discretizeSize = discretizeSize

        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

    def runEpisode(self, episodeNum):
        steps = 0
        done = False
        totalReward = 0
        state = self.discretize_state(self.env.reset())
        while not done:
            steps += 1
            action = self.getActionToTake(state, episodeNum)
            obs, reward, done, _ = self.env.step(action)
            totalReward += reward
            newState = self.discretize_state(obs)

            self.updateQTable(state, action, reward, newState, episodeNum)
            state = newState

        return steps, totalReward

    def getQTableKey(self, state, action):
        return (state[0], state[1], state[2], state[3], action)
   
    def discretize_state(self, obs):

        discretized = list()

        for i in range(len(obs)):

            # scaling factor is this state's state variable + abs(lower bound) for this state variable / the range for this state varialbe. 
            scaling = ( ( obs[i] + abs(self.lower_bounds[i]) ) / ( self.upper_bounds[i] - self.lower_bounds[i] ) )

            # new obs = an integer. the bucket size for this state -1 multiplied by the scaling factor 
            new_obs = int(round((self.discretizeSize[i] - 1) * scaling))

            # set new_obs to min between bucket size for this state -1 and the max of 0 and new obs 
            new_obs = min(self.discretizeSize[i] - 1, max(0, new_obs))

            discretized.append(new_obs)

        return tuple(discretized)
    
    def train(self):

        for ep in range(self.numEpisodes):
            
            win = False

            steps, reward = self.runEpisode(ep)
            self.totalReward += reward

            if steps > 199:
                win = True
                self.wins += 1

            episodeData = (ep, reward, win)
            self.data.append(episodeData)
        
        print("Num episodes: ", self.numEpisodes)
        print("Num wins: ", self.wins)