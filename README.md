# drlnd-navigation

## The Environment

The goal of this project is to train an agent to navigate (and collect bananas!) in a large, square world. THe environment used in this project is the Banana environment project in Unity Machine Learning Agents (ML-Agents).

![Alt Text](banana_anim.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

In order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

[1] Environment description from Deep Reinforcement Learning Nanodegree, Udacity.

## The Agent

A Deep Q Network (DQN) agent combines Q-Learning with a deep neural network to learn a policy by interacting with the environment, receiving rewards and updating the neural network weights. The DQN algorithm is also susceptible to overestimation of action values.

The agent in this project is a **Double-DQN** agent which solves the issue of overestimation. The double DQN algorithm improves on the original DQN algorithm by decoupling the action selection from the target network evaluation during computing the TD error. More information on the Double-DQN algorithm can be found in the paper [here](https://arxiv.org/abs/1509.06461).

Both the DQN and Double-DQN algorithms learn from an offline experience replay. In the original DQN paper, experiences are uniformly sampled from the replay when training in mini batches. However, this leads to situations where important experiences may not be selected often and eventually be deleted from the buffer. This project uses a **Prioritized Experience Replay** (proportional variant) to mitigate this issue. More information on prioritized replays can be found in the paper [here](https://arxiv.org/abs/1511.05952).

## Required Files 

The Double-DQN implementation can be found in **agents.py**.
Prioritied experience replay implementation can be found in **utils.py**.
Deep Q-network model can be found in **model.py**.

## Getting Started

**Prerequisites:**
You must install Python 3.6 and Unity-ML for this project.

Open the **Navigation.ipynb** file in a Jupyter notebook and click on Run.
