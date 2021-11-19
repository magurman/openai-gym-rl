from agents.CartPoleAgent import CartPoleAgent
from agents.MountainCarAgent import MountainCarAgent
from algorithms import Algorithm

import csv 

def compareSarsaAndQ():
    cartPoleQ = CartPoleAgent(algorithm=Algorithm.QLEARNING)
    cartPoleSarsa = CartPoleAgent(algorithm=Algorithm.SARSA)

    mountainCarQ = MountainCarAgent(algorithm=Algorithm.QLEARNING)
    mountainCarSarsa = MountainCarAgent(algorithm=Algorithm.SARSA)

    print("training cart pole Q: ", cartPoleQ.getName())
    cartPoleQ.train()

    print("training cart pole Sarsa: ", cartPoleSarsa.getName())
    cartPoleSarsa.train()

    print("training mountain car Q: ", mountainCarQ.getName())
    mountainCarQ.train()

    print("training mountain car Sarsa: ", mountainCarSarsa.getName())
    mountainCarSarsa.train()



def writeDataToFile(filename, colNames, data):
    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(colNames)
        for row in data:
            writer.writerow([row[0], row[1], row[2]])

def main():

    compareSarsaAndQ()


main()