#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:20:40 2020

@author: abiswas
"""

import math
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import QNetwork
from utils import ExperienceReplay
from utils import PrioritizedExperienceReplay

# DQNAgent models a reinforcement learning agent with:
#   1. Double DQN algorithm
#   2. Prioritized experience replay
#
class DQNAgent:
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self,osize,asize,seed,buffersize=1e6,gamma=0.99,epsilon=0.05,epsilondecay=1e-6,epsilonmin=0.1,minibatchsize=128,lr=0.01,epochs=3,updatefreq=3,tau=0.01,alpha=0.6,beta=0.4,usedoubledqn=True):
        
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
        self.epochs = epochs
        self.updatefreq = updatefreq
        self.tau = tau
        self.alpha = alpha
        self.beta0 = beta
        self.beta = beta
        self.doubledqn = usedoubledqn
        self.stepcount = 0
        
        # set the random seed
        self.seed = torch.manual_seed(seed)
        
        # create local and target Q networks
        self.Q = QNetwork(osize,asize).to(self.device)
        self.targetQ = QNetwork(osize,asize).to(self.device) #copy.deepcopy(self.Q).to(self.device)
        
        # initialize optimizer
        self.optimizer = optim.Adam(self.Q.parameters(),lr=self.lr)
        
        # initialize experience replay
        self.replay = PrioritizedExperienceReplay(buffersize,alpha,beta)
        
        
    def step(self,state,action,reward,next_state,done,tderr,doUpdate=False):
        # add experience to replay
        exp = (torch.from_numpy(state).float().to(self.device),
               torch.from_numpy(np.array(action)).float().to(self.device),
               torch.from_numpy(np.array(reward)).float().to(self.device),
               torch.from_numpy(next_state).float().to(self.device),
               torch.from_numpy(np.array(int(done))).float().to(self.device))
        self.replay.append(exp)
        
        # learn from experiences
        if len(self.replay.memory) > self.minibatchsize:
            for i in range(self.epochs):
                # create mini batch for learning
                experiences, batch_idxs = self.replay.sample(self.minibatchsize)
                # train the agent
                self.learn(experiences,batch_idxs,doUpdate)
        
        # increase step count
        self.stepcount += 1
        
        # decay epsilon
        self.epsilon = max(self.epsilon0 * math.exp(-self.epsilondecay * self.stepcount), self.epsilonmin)
        
        # increase beta
        self.beta = self.beta0 + (1-self.beta0) * math.exp(1e-3 * self.stepcount)
        
        
    def get_action(self,state):
        # convert network input to torch variable
        x = torch.from_numpy(state).float().to(self.device)
        
        # set network layers to eval mode
        self.Q.eval()
        
        # obtain network output
        with torch.no_grad():   # do not calculate network gradients which will speed things up
            y = self.Q(x)
            
        # set it back to train mode
        self.Q.train()
        
        # select action
        if random.random() > self.epsilon:
            # epsilon greedy action
            action = np.argmax(y.cpu().data.numpy())  # action is actually action index
        else:
            # random action selection
            action = np.random.choice(np.arange(self.asize))
        
        return np.array(action)
        
    
    def learn(self,experiences,batch_idxs,doUpdate):
        states, actions, rewards, next_states, dones = experiences
        
        # calculate td targets
        target = torch.zeros(self.minibatchsize)    # output of target Q network
        local = torch.zeros(self.minibatchsize)     # output of local Q network
        for idx,state in enumerate(next_states):
            # calculate target Q(s,a)
            target_qvalue = self.targetQ(next_states[idx]).detach() 
            local_qvalue = self.Q(next_states[idx]).detach()
            # find action with max Q value
            if self.doubledqn == True:
                aidx = torch.argmax(self.Q(next_states[idx]))
            else:
                aidx = torch.argmax(target_qvalue)
            # update target and local arrays
            target[idx] = rewards[idx] + self.gamma * target_qvalue[aidx] * (1-dones[idx])
            local[idx] = torch.max(self.Q(state))
        
        # update priorities in experience replay
        td_error = target - local
        priorities = abs(td_error.detach().numpy()) + 1e-4
        self.replay.update_priorities(priorities,batch_idxs)
        
        # calculate loss
        N = len(self.replay.memory)
        IS_weights = np.array([(1/N * 1/p) ** self.beta for p in priorities])
        loss = self.loss(local,target,IS_weights)
        
        # perform gradient descent step
        self.optimizer.zero_grad()    # reset the gradients to zero
        loss.backward()
        self.optimizer.step()
        
        # update target network
        if doUpdate:
            for target_params, params in zip(self.targetQ.parameters(), self.Q.parameters()):
                target_params.data.copy_(self.tau*params + (1-self.tau)*target_params.data)
                
    def loss(self,local,target,weights):
        # loss function with importance sampling weights
        loss = torch.mean(torch.FloatTensor(weights) * (local - target) ** 2)
        return loss
        
        
        
        
        
        
        
        
        
        
    