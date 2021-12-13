import matplotlib.pyplot as plt
import pandas

import os 

pathToProject = "/Users/magurman/NEU/Foundations of AI/openai-gym-rl"

def plotAndSave(data, xlabel, ylabel, title, location, xticks=None, yticks=None):
    
    for d in data:
        plt.plot(d[0], label=d[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks:
        plt.xticks(xticks[0], xticks[1])

    if yticks:
        plt.yticks(yticks[0], yticks[1])

    plt.legend()
    plt.savefig(location)
    plt.show()

def plotGammaAvgReward(env):
    gammas = []
    avgRewardQ = []
    avgRewardSarsa = []

    for filename in os.listdir(pathToProject + "/diffGamma"):
        gamma = filename[filename.index(".")-1:filename.index(".")+3]
        df = pandas.read_csv("diffGamma/" + filename)
        avgReward = df['reward'].mean()

        if env in filename and "Q-Learning" in filename:
            # gamma = filename[filename.index(".")-1:filename.index(".")+3]
            df = pandas.read_csv("diffGamma/" + filename)
            avgRewardQ.append(avgReward)
        elif env in filename and "SARSA" in filename:
            df = pandas.read_csv("diffGamma/" + filename)
            avgRewardSarsa.append(avgReward)

        if gamma not in gammas:
            gammas.append(gamma)

    zipped = reorganizeLabels([avgRewardQ, avgRewardSarsa], gammas)

    gammas = list(list(zipped)[0])
    avgRewardQ = list(list(zipped)[1])
    avgRewardSarsa = list(list(zipped)[2])

    data = [(avgRewardQ, "q-learning"), (avgRewardSarsa, "sarsa")]
    xlabel = "gamma"
    ylabel = "avg reward"
    xticks = ([0,1,2,3], gammas)
    title = env + " - Average Episode Reward - Q-Learning vs. Sarsa for Differing Gamma Values"

    plotAndSave(data, xlabel, ylabel, title, "diffGammaPlots/" + env + "-numWins.png", xticks)

def plotGammaNumWins(env):

    gammas = []
    numWinsQ = []
    numWinsSarsa = []

    for filename in os.listdir(pathToProject + "/diffGamma"):
        gamma = filename[filename.index(".")-1:filename.index(".")+3]
        df = pandas.read_csv("diffGamma/" + filename)
        numWins = df['win'].values.sum()

        if env in filename and "Q-Learning" in filename:
            # gamma = filename[filename.index(".")-1:filename.index(".")+3]
            df = pandas.read_csv("diffGamma/" + filename)
            numWinsQ.append(numWins)
        elif env in filename and "SARSA" in filename:
            df = pandas.read_csv("diffGamma/" + filename)
            numWinsSarsa.append(numWins)

        if gamma not in gammas:
            gammas.append(gamma)

    zipped = reorganizeLabels([numWinsQ, numWinsSarsa], gammas)

    gammas = list(list(zipped)[0])
    numWinsSarsa = list(list(zipped)[1])
    numWinsQ = list(list(zipped)[2])

    data = [(numWinsQ, "q-learning"), (numWinsSarsa, "sarsa")]
    xlabel = "gamma"
    ylabel = "number of wins"
    xticks = ([0,1,2,3], gammas)
    title = env + "- Number of Wins - Q-Learning vs. Sarsa for Differing Gamma Values"

    plotAndSave(data, xlabel, ylabel, title, "diffGammaPlots/" + env + "-numWins.png", xticks)


def reorganizeLabels(data, labels):
    lists = [labels]
    for i in data:
        lists.append(i)

    listsZipped = zip(*lists)
    sortedList = sorted(listsZipped, key= lambda pair: pair[0])
    zipped = list(zip(*sortedList))

    return zipped

def main():
    cartpole = "CartPole"
    mountaincar = "MountainCar"
     
    plotGammaNumWins(cartpole)
    plotGammaNumWins(mountaincar)
    plotGammaAvgReward(cartpole)
    plotGammaAvgReward(mountaincar)

main()