import random
import gym 
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense

from agents.AbstractQLearningAgent import AbstractQLearningAgent

class DQNLunarLanderAgent(AbstractQLearningAgent):

    replay_start_size = 1000
    replay_memory_size = 1000000
    target_update_steps = 2000
    batch_size = 32
    use_double_dqn = True

    def __init__(self, algorithm, alphaStart = 0.001, alphaEnd = 0.001, epsilonStart = 0.15, epsilonEnd = 0.1, gamma = 0.99, numEpisodes = 5000, env='LunarLander-v2'):
        super(DQNLunarLanderAgent, self).__init__(algorithm, alphaStart, alphaEnd, epsilonStart, epsilonEnd, gamma, numEpisodes, env)

        self.model = self.getModel(self.env)

        self.optimizer = keras.optimizers.Adam(learning_rate=alphaStart)

        self.model.compile(
            optimizer=self.optimizer,
            loss='mse',  # huber_loss or mse
            metrics=['accuracy']
        )

    
        self.targetModel = self.getModel(self.env)

        self.targetModel.set_weights(self.model.get_weights())

        self.replay_memory = []
    
    def getActionToTake(self, state, episodeNum):
        r = random.random()

        currentEpsilon = max(self.epsilonStart - episodeNum * (self.epsilonStart - self.epsilonEnd) / self.annealingTime, self.epsilonEnd) # guard against decay going below end value

        if currentEpsilon > r:
            return self.env.action_space.sample()
        else:
            return self.getActionFromModel(state)

    def getActionFromModel(self, state):
        return np.argmax(self.model.predict_on_batch(np.array([state]))[0])

    def train(self):
        for ep in range(self.numEpisodes):
            if ep % 50 == 0:
                print("episode number: ", ep)
            win = False

            steps, reward, finalState = self.runEpisode(ep)
            self.totalReward += reward

            if reward >= 199:
                win = True
                self.wins += 1

            episodeData = (ep, reward, win) # edit 
            self.data.append(episodeData)

            if ep == 1000:
                break

        self.writeDataToFile(self.filename, self.colNames, self.data)

    def runEpisode(self, episodeNum):
        steps = 0
        done = False
        state = self.env.reset()
        totalReward = 0
        while not done:

            # if episodeNum > 600:
            #     self.env.render()

            steps += 1
            action = self.getActionToTake(state, episodeNum)
            obs, reward, done, _ = self.env.step(action)
            self.observe(state, action, reward, obs, done, steps, episodeNum)
            totalReward += reward
            newState = obs

            state = newState

        return steps, totalReward, newState

    def getModel(self, env):

        model = keras.Sequential()
        model.add(keras.Input(shape=env.observation_space.shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.summary()
        return model
    
    def update_target(self):
        # print("Updated target. current epsilon:", self.epsilon)
        self.model_target.set_weights(self.model.get_weights())


    def observe(self, state, action, reward, next_state, done, timesteps, episodeNum):
        # store transition (state, action, reward, next_state, done) in replay memory D
        self.replay_memory.append((state, action, reward, next_state, done))

        if len(self.replay_memory) > self.replay_start_size:
            self.replay()
            
            self.epsilon = max(self.epsilonStart - episodeNum * (self.epsilonStart - self.epsilonEnd) / self.annealingTime, self.epsilonEnd)
            # self.epsilon = max(self.epsilonEnd, self.epsilonStart - self.epsilon_decay)

        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        if timesteps % self.target_update_steps == 0:
            # print("total timesteps:", timesteps)
            self.update_target()
        if timesteps % 10000 == 0:
            print("\n epsilon", self.epsilon, "timesteps", timesteps)

    def replay(self):
        # Sample random minibatch of transitions from replay memory D

        batch = random.sample(self.replay_memory, self.batch_size)

        X = []
        Y = []

        states = np.array([i[0] for i in batch])
        next_states = np.array([i[3] for i in batch])

        y = self.model.predict_on_batch(states)
        target_next = self.targetModel.predict_on_batch(next_states)

        y_next = None
        if self.use_double_dqn:
            y_next = self.model.predict_on_batch(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                y[i][action] = reward
            elif self.use_double_dqn:
                y[i][action] = reward + self.gamma * target_next[i][np.argmax(y_next[i])]
            else:
                y[i][action] = reward + self.gamma * np.amax(target_next[i])

            X.append(state)
            Y.append(y[i])

        # perform a gradient descent step on (y - Q)^2 with respect to the network parameters theta

        self.model.train_on_batch(np.array(X), np.array(Y))

        # every C steps reset Q^ = Q
    
    def discretize_state(self, obs):
        pass

    def getQTableKey(self, state, action):
        pass