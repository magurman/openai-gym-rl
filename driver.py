from agents.CartPoleAgent import CartPoleAgent
from agents.MountainCarAgent import MountainCarAgent
from algorithms import Algorithm

from agents.DQNLunarLanderAgent import DQNLunarLanderAgent

import csv 

def compareSarsaAndQ():
    cartPoleQ = CartPoleAgent(algorithm=Algorithm.QLEARNING)
    cartPoleSarsa = CartPoleAgent(algorithm=Algorithm.SARSA)

    mountainCarQ = MountainCarAgent(algorithm=Algorithm.QLEARNING)
    mountainCarSarsa = MountainCarAgent(algorithm=Algorithm.SARSA)

    colNames = ["ep", "reward", "win"]

    # print("training cart pole Q: ", cartPoleQ.getName())
    # cartPoleQ.train()
    # filename = cartPoleQ.getName() + ".csv"
    # writeDataToFile(filename, colNames, cartPoleQ.getData())

    # print("training cart pole Sarsa: ", cartPoleSarsa.getName())
    # cartPoleSarsa.train()
    # filename = cartPoleSarsa.getName() + ".csv"
    # writeDataToFile(filename, colNames, cartPoleSarsa.getData())

    print("training mountain car Q: ", mountainCarQ.getName())
    mountainCarQ.train()
    filename = mountainCarQ.getName() + ".csv"
    writeDataToFile(filename, colNames, mountainCarQ.getData())

    print("training mountain car Sarsa: ", mountainCarSarsa.getName())
    mountainCarSarsa.train()
    filename = mountainCarSarsa.getName() + ".csv"
    writeDataToFile(filename, colNames, mountainCarSarsa.getData())
    

def writeDataToFile(filename, colNames, data):
    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(colNames)
        for row in data:
            writer.writerow([row[0], row[1], row[2]])

def main():

    # compareSarsaAndQ()
    agent = DQNLunarLanderAgent("DQN")

main()