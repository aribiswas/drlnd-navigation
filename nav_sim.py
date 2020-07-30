#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 08:39:35 2020

@author: aritra
"""

import collections
import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment
from agents import DQNAgent


BUFFERSIZE = int(1e6)    # Experience buffer size
GAMMA = 0.99            # Discount factor
EPSILON = 0.95           # Epsilon parameter for selecting action
DECAY = 1e-5            # Epsilon decay rate
EPMIN = 0.1             # Minimum value of epsilon
MINIBATCHSIZE = 64     # Batch size for sampling from experience replay
LEARNRATE = 2e-4        # Learn rate of Q network
TAU = 1e-3              # Target network update factor

# training options
MAX_EPISODES = 5000     # Maximum number of training episodes
AVG_WINDOW = 100        # Window length for calculating score averages
MAX_STEPS_PER_EPISODE = 1000    # Maximum agent steps per episode

# create environment
env = UnityEnvironment(file_name='./Banana.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# create DQN agent
osize = len(env_info.vector_observations[0])
asize = brain.vector_action_space_size
seed = 0
agent = DQNAgent(osize,asize,seed,BUFFERSIZE,GAMMA,EPSILON,DECAY,EPMIN,MINIBATCHSIZE,LEARNRATE,TAU)

# load the weights from file
agent.Q.load_state_dict(torch.load('checkpoint.pth'))

# number of simulations
NUM_SIMS = 3

for i in range(NUM_SIMS):
    
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    
    for t in range(1,MAX_STEPS_PER_EPISODE):
        
        # sample action
        action = agent.get_action(state)
        
        # step the environment
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0]
        
        state = next_state
        if done:
            break 
            
env.close()