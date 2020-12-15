import os
from collections import defaultdict
from typing import List, Callable
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.cnn_1 = nn.Conv2d(
            in_channels=1, out_channels=16, 
            kernel_size=5, stride=1, padding=0
        )
        self.cnn_2 = nn.Conv2d(
            in_channels=16, out_channels=32, 
            kernel_size=5, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.out = nn.Linear(64, 10)
        
    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)
        
        return out
