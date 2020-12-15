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


class Reshape(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape
        
    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]
        X = X.reshape(batch_size, *self.shape)
        return X


class MultiHeadFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, feature_size: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        
    def forward(self, X: torch.Tensor):
        batch_size, head_size, channel_size, height, width = X.shape

        # Move batch and head dimensions to the end
        X = torch.movedim(X, 0, -1)
        X = torch.movedim(X, 0, -1)

        # Merge batch and head
        X = X.reshape(channel_size, height, width, batch_size * head_size)

        # Return batch dimension back
        X = torch.movedim(X, -1, 0)

        output = self.feature_extractor(X)

        # Expand batch and head
        output = torch.movedim(output, 0, -1)
        output = output.reshape(self.feature_size, batch_size, head_size)

        output = torch.movedim(output, -1, 0)
        output = torch.movedim(output, -1, 0)

        output = output.reshape(batch_size, head_size * self.feature_size)

        return output
