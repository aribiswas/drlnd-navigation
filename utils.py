#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:27:30 2020

@author: aritra
"""

import torch
import random
import collections
import numpy as np

# ExperienceReplay stores experiences in a circular buffer.
# Each experience is a tuple of (state,action,reward,next_state,done)
class ExperienceReplay:
    def __init__(self,capacity=1e6):
        # initialize the buffer
        self.memory = collections.deque(maxlen=capacity)
        
    def reset(self):
        self.memory.clear()
        
    def append(self,experience):
        self.memory.append(experience)
    
    def sample(self,batchsize):
        # randomly sample experiences from uniform distribution
        experiences = random.sample(self.memory,batchsize)
        
        # stack state, action, reward, next_state and done values
        states = torch.stack([experiences[i][0] for i in range(len(experiences))])
        actions = torch.stack([experiences[i][1] for i in range(len(experiences))])
        rewards = torch.stack([experiences[i][2] for i in range(len(experiences))])
        next_states = torch.stack([experiences[i][3] for i in range(len(experiences))])
        dones = torch.stack([experiences[i][4] for i in range(len(experiences))])
        
        return (states,actions,rewards,next_states,dones)
    

# PrioritizedExperienceReplay stores experiences in a circular buffer.
# Each experience is a tuple of (state,action,reward,next_state,done)
# Experience priorities are stored in a second data structure
class PrioritizedExperienceReplay:
    def __init__(self,buffersize,alpha,beta):
        # initialize the buffer
        self.memory = collections.deque(maxlen=buffersize)
        self.priorities = collections.deque(maxlen=buffersize)
        self.alpha = alpha
        
    def reset(self):
        self.memory.clear()
        self.priorities.clear()
        
    def append(self,experience):
        self.memory.append(experience)
        max_prio = max(self.priorities) if self.priorities else 1
        self.priorities.append(max_prio)  # append with max priority
    
    def sample(self,batchsize):
        # total number of experiences in the replay memory
        num_exp_in_buffer = len(self.memory)
        
        # discrete probabilities of experiences
        # P(j) = (p_j)^a / sum(p^a)
        prio_alpha = [p ** self.alpha for p in self.priorities]
        sum_prio = sum(prio_alpha)
        probs = [prio / sum_prio for prio in prio_alpha]
        
        # select experiences
        choices = np.random.choice(num_exp_in_buffer, batchsize, p=probs)
        
        # stack state, action, reward, next_state and done values
        states = torch.stack([self.memory[item][0] for item in choices])
        actions = torch.stack([self.memory[item][1] for item in choices])
        rewards = torch.stack([self.memory[item][2] for item in choices])
        next_states = torch.stack([self.memory[item][3] for item in choices])
        dones = torch.stack([self.memory[item][4] for item in choices])
        
        experiences = (states,actions,rewards,next_states,dones)
        
        return experiences, choices
    
    def update_priorities(self,priorities,indices):
        for p,idx in zip(priorities,indices):
            self.priorities[idx] = p
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        