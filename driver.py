from agents.CartPoleAgent import CartPoleAgent
from agents.MountainCarAgent import MountainCarAgent

import csv 


fileExtension = ".csv"
filename = "training"


def main():

    agents = [CartPoleAgent(), MountainCarAgent()]

    for agent in agents:
        for i in range(2):
            agent.train()

            finalFileName = agent.getName() + "-" + filename + "-" + str(i) + fileExtension

            with open(finalFileName, mode="w") as file:
                writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["episode", "reward", "win"])
                for data in agent.getData():
                    writer.writerow([data[0], data[1], data[2]])

            agent.reset()


main()