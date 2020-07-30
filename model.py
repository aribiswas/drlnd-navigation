#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-network representation.

@author: abiswas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self,osize,asize,seed=0):
        
        super(QNetwork,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # define the deep neural network structure
        self.fc1 = nn.Linear(osize,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,asize)
        self.bn1 = nn.BatchNorm1d(64)
        
    
    def forward(self,x):
        
        # make forward pass through the network
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x