#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:20:40 2020
Revised on Tue Jul 28

@author: abiswas
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:20:40 2020

@author: abiswas
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import QNetwork
from utils import ExperienceReplay

# DQNAgent models a reinforcement learning agent with Double DQN algorithm.
class DQNAgent:
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self,osize,asize,seed,buffersize=int(1e6),gamma=0.99,epsilon=0.05,epsilondecay=1e6,epsilonmin=0.1,minibatchsize=128,lr=0.01,tau=0.01):
        """
        Initialize DQN agent parameters.
        """
        
        # initialize agent parameters
        self.osize = osize
        self.asize = asize
        self.gamma = gamma
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.epsilondecay = epsilondecay
        self.epsilonmin = epsilonmin
        self.minibatchsize = minibatchsize
        self.lr = lr
        self.tau = tau
        self.stepcount = 0
        self.loss_log = []
        
        # set the random seed
        self.seed = torch.manual_seed(seed)
        
        # create local and target Q networks
        self.Q = QNetwork(osize,asize).to(self.device)
        self.targetQ = QNetwork(osize,asize).to(self.device)
        
        # initialize optimizer
        self.optimizer = optim.Adam(self.Q.parameters(),lr=self.lr)
        
        # initialize experience replay
        self.replay = ExperienceReplay(asize,buffersize,minibatchsize,seed)
        
        
    def step(self,state,action,reward,next_state,done):
        """
        Step the agent, and learn if necessary.
        """
        
        # add experience to replay
        self.replay.add(state,action,reward,next_state,done)
        
        # learn from experiences
        if self.replay.__len__() > self.minibatchsize:
            # create mini batch for learning
            experiences = self.replay.sample(self.device)
            # train the agent
            self.learn(experiences)
        
        # increase step count
        self.stepcount += 1
        
        # decay epsilon
        decayed_epsilon = self.epsilon * (1-self.epsilondecay)
        self.epsilon = max(self.epsilonmin, decayed_epsilon)
        
        
    def get_action(self,state):
        """
        Get an epsilon greedy action.
        """
        
        # convert network input to torch variable
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # obtain network output
        self.Q.eval()
        with torch.no_grad():   # do not calculate network gradients which will speed things up
            y = self.Q(x)
        self.Q.train()
            
        # select action
        if random.random() > self.epsilon:
            # epsilon greedy action
            action = np.argmax(y.cpu().data.numpy())  # action is actually action index
        else:
            # random action selection
            action = np.random.choice(np.arange(self.asize))
        
        return action
        
    
    def learn(self,experiences):
        """
        Learn using Double DQN algorithm.
        """
        
        # unpack experience
        states, actions, rewards, next_states, dones = experiences
        
        # get the argmax of Q(next_state)
        a_max = torch.argmax(self.Q(next_states), dim=1).cpu().data.numpy().reshape((self.minibatchsize,1))
        
        # obtain the target Q network output
        target_out = self.targetQ(next_states).detach().data.numpy()
        target_q = np.array([tout[aidx] for tout,aidx in zip(target_out,a_max)])
        
        # calculate target and local Qs
        target = rewards + self.gamma * target_q * (1-dones)
        local = self.Q(states).gather(1,actions)
        
        # calculate loss
        loss = F.mse_loss(local,target)
        self.loss_log.append(loss.cpu().data.numpy())
        
        # perform gradient descent step
        self.optimizer.zero_grad()    # reset the gradients to zero
        loss.backward()
        self.optimizer.step()
        
        if self.stepcount%10==0:
            # soft update target network
            for target_params, params in zip(self.targetQ.parameters(), self.Q.parameters()):
                target_params.data.copy_(self.tau*params + (1-self.tau)*target_params.data)
    