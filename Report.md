# drlnd-navigation

## The Environment

The goal of this project is to train an agent to navigate (and collect bananas!) in a large, square world. THe environment used in this project is the Banana environment project in Unity Machine Learning Agents (ML-Agents).

![Alt Text](banana_anim.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

[1] Environment description from Deep Reinforcement Learning Nanodegree, Udacity.

## The Agent

A Deep Q-Network (DQN) agent combines Q-Learning with a deep neural network to learn a policy. The agent learns by interacting with the environment, receiving rewards and updating the neural network weights. 

The agent in this project uses the **Double-DQN** algorithm which is an improvement on the original DQN algorithm. The double DQN algorithm decouples the action selection from the target network evaluation during computation of the TD error. This reduces issues like overestimation during training. More information on the Double-DQN algorithm can be found in the paper [here](https://arxiv.org/abs/1509.06461).


## Required Files 

The Double-DQN implementation can be found in **agents.py**.
Prioritied experience replay implementation can be found in **utils.py**.
Deep Q-network model can be found in **model.py**.

## Getting Started

**Prerequisites:**
To run this project, you must have Python3 and Pytorch installed. Install Python3 through the Anaconda distribution https://www.anaconda.com/products/individual. Install Pytorch binaries from https://pytorch.org/get-started/locally/.

You must use one of the following Unity environments for this project. Download the environment specific to your platform and place it in the same folder as this project.

* Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
* Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
* Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
* Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip


