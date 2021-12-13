from agents.CartPoleAgent import CartPoleAgent
from agents.MountainCarAgent import MountainCarAgent
from algorithms import Algorithm

from agents.DQNLunarLanderAgent import DQNLunarLanderAgent

import csv 

def compareDiffEpsilonDecay(algorithm):
    mountainCar = MountainCarAgent(algorithm=algorithm)
    cartPole = CartPoleAgent(algorithm=algorithm)

    agents = [mountainCar, cartPole]

    numEpisodes = 5000

    annealingTime = numEpisodes / 8

    while annealingTime <= (numEpisodes / 1):
        for agent in agents:
            # for now running training  each agent 3 times for each gamma and taking data from max wins training session
            n = 3
            dataToWrite = None
            maxWins = -1
            while n > 0:
                agent.reset()
                agent.setAnnealingTime(annealingTime)
                print("Starting training for: " + agent.getName() + " with annealing time= " + str(agent.getAnnealingTime()))
                agent.train()

                if agent.wins > maxWins:
                    maxWins = agent.wins
                    dataToWrite = agent.data
                n -= 1
            agent.writeDataToFile("diffEpsilonDecay/" + agent.filename + "-annealingTime-" + str(annealingTime) + ".csv",agent.colNames,dataToWrite)
        
        annealingTime = annealingTime * 1.5

def compareDiffGamma(algorithm):
    mountainCar = MountainCarAgent(algorithm=algorithm)
    cartPole = CartPoleAgent(algorithm=algorithm)

    agents = [mountainCar, cartPole]

    gamma = 0.99

    while gamma >= 0.8:
        for agent in agents:

            # for now running training  each agent 3 times for each gamma and taking data from max wins training session
            n = 3
            dataToWrite = None
            maxWins = -1
            while n > 0:
                agent.reset()
                agent.setGamma(gamma)
                print("Starting training for: " + agent.getName() + " with gamma= " + str(agent.getGamma()))
                agent.train()

                if agent.wins > maxWins:
                    maxWins = agent.wins
                    dataToWrite = agent.data
                n -= 1
            agent.writeDataToFile("diffGamma/" + agent.filename + "-gamma-" + str(gamma) + ".csv",agent.colNames,dataToWrite)
        
        gamma = round(gamma - 0.05, 2)    

def compareDiffEpsilon(algorithm):
    mountainCar = MountainCarAgent(algorithm=algorithm)
    cartPole = CartPoleAgent(algorithm=algorithm)

    agents = [mountainCar, cartPole]

    epsilonStart = 1.0

    while epsilonStart >= 0.7:
        for agent in agents:

            # for now training each agent 3 times for each epsilonStart and taking data from max wins training session
            n = 3
            dataToWrite = None
            maxWins = -1
            while n > 0:
                agent.reset()
                agent.setEpsilonStart(epsilonStart)
                print("Starting training for: " + agent.getName() + " with epsilonStart= " + str(agent.getEpsilonStart()))
                agent.train()

                if agent.wins > maxWins:
                    maxWins = agent.wins
                    dataToWrite = agent.data
                n -= 1
            agent.writeDataToFile("diffEpsilonStart/" + agent.filename + "-epsilonStart-" + str(epsilonStart) + ".csv",agent.colNames,dataToWrite)
        
        epsilonStart = round(epsilonStart - 0.1, 1)

def writeDataToFile(filename, colNames, data):
    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(colNames)
        for row in data:
            writer.writerow([row[0], row[1], row[2]])

def main():
    # compareDiffGamma(Algorithm.SARSA)
    # compareDiffEpsilon(Algorithm.SARSA)
    # compareDiffEpsilonDecay(Algorithm.QLEARNING)
    # compareDiffEpsilonDecay(Algorithm.SARSA)

main()