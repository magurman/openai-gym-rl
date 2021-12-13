import matplotlib.pyplot as plt
import pandas

from algorithms import Algorithm

import os 

pathToProject = "/Users/magurman/NEU/Foundations of AI/openai-gym-rl"

def plotAndSave(data, xlabel, ylabel, title, location, show, xticks=None, yticks=None):
    
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
    if show:
        plt.show()

def plotAll(env, metric, analysis, show):

    metricLabels = []
    q = []
    sarsa = []

    dir = "diffGamma" if analysis == "gamma" else "diffEpsilonStart" if analysis == "epsilonStart" else "diffEpsilonDecay"

    for filename in os.listdir(pathToProject + "/" + dir):
        filenameMetric = filename[filename.index(".")-1:filename.index(".")+3]
        df = pandas.read_csv(dir + "/" + filename)

        val = df['win'].values.sum() if metric == "win" else df['reward'].mean()

        if env in filename:
            if Algorithm.QLEARNING.value in filename:
                q.append(val)
            elif env in filename and Algorithm.SARSA.value in filename:
                sarsa.append(val)

        if filenameMetric not in metricLabels:
            metricLabels.append(filenameMetric)

    zipped = reorganizeLabels([q, sarsa], metricLabels)

    metricLabels = list(list(zipped)[0])
    q = list(list(zipped)[2])
    sarsa = list(list(zipped)[1])

    data = [(q, "q-learning"), (sarsa, "sarsa")]
    xlabel = "gamma" if analysis == "gamma" else "epsilon start" if analysis == "epsilonStart" else "annealing time"
    ylabel = "number of wins" if metric == "win" else "average reward"
    xticks = ([i for i in range(len(metricLabels))], metricLabels)

    titleMetric = "Number of Wins" if metric == "win" else "Average Episode Reward"

    titleAnalysis = "Gamma Values" if analysis == "gamma" else "Epsilon Start Values" if analysis == "epsilonStart" else "Annealing Times"

    title = env + "- " + titleMetric + " - Q-Learning vs. Sarsa for Differing " + titleAnalysis

    locationMetric = "numWins" if metric == "win" else "avgReward"
    location = dir + "Plots/" + env + "-" + locationMetric + ".png"
    plotAndSave(data, xlabel, ylabel, title, location, show, xticks)


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

    envs = [cartpole, mountaincar]

    numWins = "win"
    avgReward = "avgReward"

    metrics = [numWins, avgReward]

    diffGamma = "gamma"
    diffEpsilonDecay = "epsilonDecay"
    diffEpsilonStart = "epsilonStart"

    analysis = [diffGamma, diffEpsilonDecay, diffEpsilonStart]

    for env in envs:
        for metric in metrics:
            for anal in analysis:
                # print("Plotting:")
                # print(env)
                # print(metric)
                # print(anal)
                plotAll(env, metric, anal, show=False)

main()