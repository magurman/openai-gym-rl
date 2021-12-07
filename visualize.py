import seaborn as sns
import pandas


cartPoleQ = pandas.read_csv("CartPole-v0-Q-Learning.csv")
cartPoleSarsa = pandas.read_csv("CartPole-v0-SARSA.csv")

mountainCarQ = pandas.read_csv("MountainCar-v0-Q-Learning.csv")
mountainCarSarsa = pandas.read_csv("MountainCar-v0-SARSA.csv")





numWins0 = cartpole0['win'].value_counts()
averageReward0 = cartpole0['reward'].mean()

numWins1 = cartpole1['win'].value_counts()
averageReward1 = cartpole1['reward'].mean()
print(numWins0)
print(averageReward0)

print(numWins1)
print(averageReward1)