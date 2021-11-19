import gym 
import collections
import random
import numpy as np
import math
import time
import statistics
from agents.AbstractQLearningAgent import AbstractQLearningAgent

class MountainCarAgent(AbstractQLearningAgent):
    def __init__(self, algorithm, alphaStart = 0.1, alphaEnd = 0.1, epsilonStart = 0.15, epsilonEnd = 0.01, gamma = 0.95, numEpisodes = 5000, env='MountainCar-v0', discretizeSize=[1,2]):
        super(MountainCarAgent, self).__init__(algorithm, alphaStart, alphaEnd, epsilonStart, epsilonEnd, gamma, numEpisodes, env)

        self.discretizeSize = discretizeSize

        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]


    def getQTableKey(self, state, action):
        return (state[0], state[1], action)

    def discretize_state(self, obs):
        return list(map(lambda x, y: round(x, y), obs, self.discretizeSize))

    def runEpisode(self, episodeNum):
        steps = 0
        done = False
        state = self.discretize_state(self.env.reset())
        totalReward = 0
        while not done:
            steps += 1
            action = self.getActionToTake(state, episodeNum)
            obs, reward, done, _ = self.env.step(action)
            totalReward += reward
            newState = self.discretize_state(obs)

            if done and newState[0] >= 0.5:
                self.qTable[(state[0], state[1], action)] = 0
                break

            self.updateQTable(state, action, reward, newState, episodeNum)
            state = newState
        return steps, totalReward, newState

    def train(self):
    
        for ep in range(self.numEpisodes):

            win = False

            steps, reward, finalState = self.runEpisode(ep)
            self.totalReward += reward

            if steps < 199 and finalState[0] >= 0.5:
                win = True
                self.wins += 1

            episodeData = (ep, reward, win)
            self.data.append(episodeData)

        print("Num episodes: ", self.numEpisodes)
        print("Num wins: ", self.wins)

    def getName(self):
        return "MountainCarAgent"
# class MountainCarAgent:

#     def __init__(self, alpha = .1, epsilonStart = .15, gamma = 0.95, numEpisodes = 5000, discretizeSize = [1,2]):
#         self.env = gym.make('MountainCar-v0')
#         self.numEpisodes = numEpisodes

#         self.alphaStart = alpha # learning rate 
#         self.alphaEnd = 0.1
#         self.epsilonStart = epsilonStart # greedy start rate 
#         self.epsilonEnd = 0.01 # greedy end rate

#         self.gamma = gamma # discount factor
        
#         self.annealingTime = self.numEpisodes / 8 # num episodes for decay to occur 

#         self.discretizeSize = discretizeSize

#         self.qTable = collections.defaultdict(int)

#         self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
#         self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

#     def getQValue(self, state, action):
#         return self.qTable[(state[0], state[1], action)]

#     def updateQTable(self, state, action, reward, nextState, episodeNum):
#         key = (state[0], state[1], action)
#         currentAlpha = max(self.alphaStart - episodeNum * (self.alphaStart - self.alphaEnd) / self.annealingTime, self.alphaEnd) # guard against decay going below end value

#         self.qTable[key] += (currentAlpha * 
#                                         (reward 
#                                          + self.gamma * np.max(self.computeValueFromQValues(nextState)) 
#                                          - self.qTable[key]))

#     def getActionToTake(self, state, episodeNum):
#         r = random.random()

#         currentEpsilon = max(self.epsilonStart - episodeNum * (self.epsilonStart - self.epsilonEnd) / self.annealingTime, self.epsilonEnd) # guard against decay going below end value

#         if currentEpsilon > r:
#             return self.env.action_space.sample()
#         else:
#             return self.computeActionFromQValues(state)

#     def computeValueFromQValues(self, state):
#         maxQVal = -math.inf
#         maxAction = None

#         for action in range(self.env.action_space.n):
#               qVal = self.getQValue(state, action)
#               if qVal > maxQVal:
#                     maxQVal = qVal
#                     maxAction = action
        
#         return maxQVal

#     def computeActionFromQValues(self, state):
#         bestVal = -math.inf
#         actionsToChooseFrom = []

#         actions = self.env.action_space

#         for action in range(actions.n):
#             qVal = self.getQValue(state, action)
#             if qVal > bestVal:
#                 bestVal = qVal
#                 actionsToChooseFrom = [action]
#             elif self.getQValue(state, action) == bestVal:
#                 actionsToChooseFrom.append(action)
#             else:
#                 pass

#         return random.choice(actionsToChooseFrom)

    # def runEpisode(self, episodeNum):
    #     steps = 0
    #     done = False
    #     state = self.discretize_state(self.env.reset())
    #     totalReward = 0
    #     while not done:
    #         steps += 1
    #         action = self.getActionToTake(state, episodeNum)
    #         obs, reward, done, _ = self.env.step(action)
    #         totalReward += reward
    #         newState = self.discretize_state(obs)

    #         if done and newState[0] >= 0.5:
    #             self.qTable[(state[0], state[1], action)] = 0
    #             break

    #         self.updateQTable(state, action, reward, newState, episodeNum)
    #         state = newState
    #     return steps, totalReward, newState

    # def train(self):

    #     wins = 0
    #     rewards = []
    #     for ep in range(self.numEpisodes):

    #         steps, totalReward, finalState = self.runEpisode(ep)
    #         rewards.append(totalReward)

    #         if steps < 199 and finalState[0] >= 0.5:
    #             wins += 1

        
    #     print("Num episodes: ", self.numEpisodes)
    #     print("Num wins: ", wins)

#     '''
#     need to modify this as well as the initialization of upper and lower bounds. 
#     '''
#     def discretize_state(self, obs):
#         return list(map(lambda x, y: round(x, y), obs, self.discretizeSize))


#     def resetQTable(self):
#         self.qTable = collections.defaultdict(int)

# if __name__ == "__main__":
#     agent = MountainCarAgent()
#     agent.train()