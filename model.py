#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-network representation.

@author: abiswas
"""

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    
    def __init__(self,osize,asize):
        
        super(QNetwork,self).__init__()
        
        # define the deep neural network structure
        self.model = nn.Sequential(nn.Linear(osize,64),
                                   #nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Linear(64,32),
                                   #nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.Linear(32,asize))
        
    
    def forward(self,x):
        # flatten the input
        #x = x.view(-1,x.size(0))
        # make forward pass through the network
        x = self.model(x)
        return x