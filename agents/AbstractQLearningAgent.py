from abc import ABC, abstractmethod
import gym 
import collections
import math
import random
import numpy as np
import csv 
from algorithms import Algorithm

class AbstractQLearningAgent(ABC):

    @abstractmethod
    def __init__(self, algorithm, alphaStart, alphaEnd, epsilonStart , epsilonEnd, gamma, annealingTime, numEpisodes, env):
        self.env = gym.make(env)

        self.numEpisodes = numEpisodes

        self.alphaStart = alphaStart # learning rate 
        self.alphaEnd = alphaEnd
        self.epsilonStart = epsilonStart # greedy start rate 
        self.epsilonEnd = epsilonEnd # greedy end rate

        self.gamma = gamma # discount factor
        
        self.annealingTime = annealingTime # num episodes for decay to occur 
        self.qTable = collections.defaultdict(int)

        self. wins = 0
        self.totalReward = 0

        self.data = []
        self.colNames = ["ep", "reward", "win"]

        self.algorithm = algorithm

        self.filename = self.getName()

    def getQValue(self, state, action):
        key = self.getQTableKey(state, action)
        return self.qTable[key]

    def updateQTable(self, state, action, reward, nextState, episodeNum):
        currentAlpha = max(self.alphaStart - episodeNum * (self.alphaStart - self.alphaEnd) / self.annealingTime, self.alphaEnd) # guard against decay going below end value
        key = self.getQTableKey(state, action)

        if self.algorithm == Algorithm.QLEARNING:
            self.qTable[key] += (currentAlpha * 
                                            (reward + self.gamma * self.computeValueFromQValues(nextState) - self.qTable[key]))
        elif self.algorithm == Algorithm.SARSA:
            nextAction = self.getActionToTake(nextState, episodeNum)
            nextStateKey = self.getQTableKey(nextState, nextAction)

            self.qTable[key] += (currentAlpha * (reward + self.gamma * (self.qTable[nextStateKey] - self.qTable[key])))
            
    def getActionToTake(self, state, episodeNum):
        r = random.random()

        currentEpsilon = max(self.epsilonStart - episodeNum * (self.epsilonStart - self.epsilonEnd) / self.annealingTime, self.epsilonEnd) # guard against decay going below end value

        if currentEpsilon > r:
            return self.env.action_space.sample()
        else:
            return self.computeActionFromQValues(state)

    def computeValueFromQValues(self, state):
        maxQVal = -math.inf
        maxAction = None

        for action in range(self.env.action_space.n):
              qVal = self.getQValue(state, action)
              if qVal > maxQVal:
                    maxQVal = qVal
                    maxAction = action
        
        return maxQVal

    def computeActionFromQValues(self, state):
        bestVal = -math.inf
        actionsToChooseFrom = []

        actions = self.env.action_space

        for action in range(actions.n):
            qVal = self.getQValue(state, action)
            if qVal > bestVal:
                bestVal = qVal
                actionsToChooseFrom = [action]
            elif self.getQValue(state, action) == bestVal:
                actionsToChooseFrom.append(action)
            else:
                pass

        return random.choice(actionsToChooseFrom)

    def reset(self):
        self.qTable = collections.defaultdict(int)
        self.wins = 0
        self.totalReward = 0
        self.data = []

    def setEpsilonStart(self, epsilonStart):
        self.epsilonStart = epsilonStart

    def setEpsilonEnd(self, epsilonEnd):
        self.epsilonEnd = epsilonEnd

    def setAlphaStart(self, alphaStart):
        self.alphaStart = alphaStart

    def setAlphaEnd(self, alphaEnd):
        self.alphaEnd = alphaEnd

    def setGamma(self, gamma):
        self.gamma = gamma 

    def setAnnealingTime(self, annealingTime):
        self.annealingTime = annealingTime

    def getNumEpisodes(self):
        return self.numEpisodes

    def setNumEpisodes(self, numEpisodes):
        self.numEpisodes = numEpisodes

    def setAnnealingTime(self, annealingTime):
        self.annealingTime = annealingTime

    def getData(self):
        return self.data

    def getName(self):
        return self.env.unwrapped.spec.id + "-" + self.algorithm.value

    def getEpsilonStart(self):
        return self.epsilonStart

    def getGamma(self):
        return self.gamma

    def getAnnealingTime(self):
        return self.annealingTime

    def writeDataToFile(self, filename, colNames, data):
        with open(filename, mode="w") as file:
            writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(colNames)
            for row in data:
                writer.writerow([row[0], row[1], row[2]])

    @abstractmethod
    def getQTableKey(self, state, action):
        pass

    @abstractmethod
    def discretize_state(self, obs):
        pass

    @abstractmethod
    def runEpisode(self, episodeNum):
        pass

    @abstractmethod
    def train(self):
        pass