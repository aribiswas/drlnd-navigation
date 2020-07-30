#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:37:21 2020
Revised on Tue Jul 28

@author: abiswas
"""

import collections
import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment
from agents import DQNAgent


# DQN hyperparameters
BUFFERSIZE = int(1e6)    # Experience buffer size
GAMMA = 0.99             # Discount factor
EPSILON = 0.95           # Epsilon parameter for selecting action
DECAY = 1e-5             # Epsilon decay rate
EPMIN = 0.1              # Minimum value of epsilon
MINIBATCHSIZE = 64       # Batch size for sampling from experience replay
LEARNRATE = 2e-4         # Learn rate of Q network
TAU = 1e-3               # Target network update factor

# training options
MAX_EPISODES = 5000      # Maximum number of training episodes
AVG_WINDOW = 100         # Window length for calculating score averages
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

# log scores
reward_log = []
avg_log = []
avg_window = collections.deque(maxlen=AVG_WINDOW)

# verbosity
VERBOSE = True

# Train the agent
for ep_count in range(1,MAX_EPISODES):

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    
    ep_reward = 0
    
    for t in range(1,MAX_STEPS_PER_EPISODE):
        # sample action from the current policy
        action = agent.get_action(state)
        
        # step the environment
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0] 
        done = env_info.local_done[0]
        
        # step the agent
        agent.step(state,action,reward,next_state,done)
        
        state = next_state
        ep_reward += reward
        
        # terminate if done
        if done:
            break
    
    # print training progress
    avg_window.append(ep_reward)
    avg_reward = np.mean(avg_window)
    avg_log.append(avg_reward)
    reward_log.append(ep_reward)
    if VERBOSE and (ep_count==1 or ep_count%100==0):
        print('EP: {} \tEPR: {:8.4f} \tAVR: {:8.4f} \tEpsilon: {:.4f} \tLoss: {:8.4f}'.format(ep_count,ep_reward,avg_reward,agent.epsilon,agent.loss_log[ep_count]))
    
    # check if env is solved
    if avg_log[ep_count-1] >= 13:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep_count, avg_log[ep_count-1]))
        torch.save(agent.Q.state_dict(), 'checkpoint.pth')
        break

# Close environment
env.close()

# plot score history
plt.ion()
fig, axarr = plt.subplots(2,1, figsize=(6,6), dpi=200)
ax1 = axarr[0]
ax1.set_title("Training Results", fontsize=16)
ax1.set_xlabel("Episodes", fontsize=16)
ax1.set_ylabel("Average Reward", fontsize=16)
ax1.set_xlim([0, ep_count+20])
ax1.set_ylim([0, 20])
ax1.plot(range(1,ep_count+1),avg_log)

# plot loss
ax2 = axarr[1]
ax2.set_xlabel("Steps", fontsize=16)
ax2.set_ylabel("Loss", fontsize=16)
ax2.set_xlim([0, agent.stepcount+20])
ax2.plot(range(agent.minibatchsize,agent.stepcount),agent.loss_log)

plt.show()
    
