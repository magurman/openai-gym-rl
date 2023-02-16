# openai-gym-rl

## Summary

This project explores different reinforcement learning methods to solve various environments provided by OpenAI's gym framework. The initial focus was on identifying the proper methods to use in the CartPole and MountainCar environments, which resulted in Q-Learning and SARSA agents solving their environments with 60-90% success rates with 5,000 episodes.

The project then shifted to using more complex methods to solve environments with much larger state-action spaces, specifically implementing a DQN and Actor-Critic algorithm. However, the agents in our models failed to reach sufficient performance using these methods.

Despite this setback, the project has laid a solid foundation and the team plans to improve the latter two algorithms to outperform our implementations for Q-Learning and SARSA.

## Project Description
This project uses OpenAI's gym library to train simple agents in classical environments using Reinforcement Learning algorithms. The team plans to experiment with three distinct environments: Mountain Car, Cart Pole, and Lunar Lander. Each environment is described in detail below.

### Cart Pole Environment
In the Cart Pole environment, a pole is attached to a moving cart (the agent). For each time step, a reward of +1 is given to the agent if the pole remains upright (within 15 degrees from vertical). If the angle between the pole and cart exceeds 15 degrees at any point, this is considered a terminal state and the episode will end. Additionally, if the cart moves 2.4 units from the center, this is also considered a terminal state.

The action space in this environment is discrete and is defined as either a 0 or 1, which is to apply one unit of force to the cart in the right or left direction, respectively. The state space in this environment is continuous and is defined as an array of four floats: cart position, cart velocity, pole angle, and pole velocity at tip.

### Mountain Car Environment
In the Mountain Car environment, a car (the agent) is positioned between two mountains and must use momentum to reach the top of the mountain at the right. The car receives no reward until it reaches the top of the mountain (+100).

The action space in this environment is discrete and is defined as either a 0, 1, or 2, which is to apply one unit of force to the left, do nothing, and apply one unit of force to the right, respectively. The state space in this environment is continuous and is defined as an array of two floats: car position and car velocity.

### Lunar Lander Environment
In the Lunar Lander environment, a spaceship (the agent) must land on a landing pad, while avoiding obstacles such as craters and rocks. The agent receives a reward for successfully landing on the pad and penalties for crashing into obstacles.

The action space in this environment is discrete and is defined as either a 0, 1, or 2, which is to apply one unit of force to the left, do nothing, and apply one unit of force to the right, respectively. The state space in this environment is continuous and is defined as an array of eight floats: the x and y coordinates, horizontal and vertical velocity, angle, angular velocity, and two boolean values indicating whether the legs are touching the ground.

The team has decided to pursue Model-Free Reinforcement Learning (RL) rather than Model-Based RL. Within the domain of Model-Free RL, the team has worked on four algorithms: Q-Learning, SARSA, Deep Q Learning, and Actor Critic.

