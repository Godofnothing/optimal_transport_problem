import os
import sys
from collections import defaultdict
from typing import List, Callable
from numbers import Number
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer

DIR_PATH = Path(__file__).parent
sys.path.append(str(DIR_PATH))
import sinkhorn_layers
import utils_layers


class DeepPermNet_v2(nn.Module):
    def __init__(
        self
        , feature_extractor: nn.Module
        , feature_size: int
        , head_size: int
        , disable_feature_extractor_training: bool
        , permutation_extractor_version: str
        , **permutation_extractor_extra_kwargs
    ):
        
        super().__init__()
        
        self.feature_extractor = self._setup_feature_extractor(
            feature_extractor
            , feature_size
            , disable_feature_extractor_training
        )
        
        self.permutation_extractor = getattr(
            self
            , f'_setup_permutation_extractor_{permutation_extractor_version.lower()}'
        )(head_size, feature_size, **permutation_extractor_extra_kwargs)
            
    def forward(self, X: torch.Tensor):
        """
            X - Torch tensor of shape (batch_size, head_size, color_channels, height, width)
        """

        features = self.feature_extractor(X)

        permutations = self.permutation_extractor(features)
        
        return permutations
    
    def _setup_feature_extractor(
        self
        , feature_extractor: nn.Module
        , feature_size: int
        , disable_feature_extractor_training: bool
    ):
        
        if disable_feature_extractor_training:
            for param in feature_extractor.parameters():
                param.requires_grad = False
                
        return utils_layers.MultiHeadFeatureExtractor(feature_extractor, feature_size)
    
    def _setup_permutation_extractor_v1(
        self
        , head_size: int
        , feature_size: int
        , bottleneck_features_num: int
        , use_stable_normalizer: bool
        , **norm_kwargs
    ):

        return nn.Sequential(
            nn.Linear(feature_size * head_size, bottleneck_features_num)
            , nn.ReLU()
            , nn.Linear(bottleneck_features_num, head_size ** 2)
            , nn.ReLU()
            , utils_layers.Reshape((head_size,) * 2)
            , sinkhorn_layers.SinkhornNormalizer(**norm_kwargs) if use_stable_normalizer \
                else sinkhorn_layers.SinkhornOriginalNormalizer(**norm_kwargs)
            , sinkhorn_layers.SinkhornOptimizer_v1()
        )
    
    def _setup_permutation_extractor_v2(
        self
        , head_size: int
        , feature_size: int
        , entropy_reg: float
        , bottleneck_features_num: int
        , use_stable_normalizer: bool
        , **norm_kwargs
    ):

        return nn.Sequential(
            nn.Linear(feature_size * head_size, bottleneck_features_num)
            , nn.ReLU()
            , nn.Linear(bottleneck_features_num, head_size ** 2)
            , nn.ReLU()
            , utils_layers.Reshape((head_size,) * 2)
            , sinkhorn_layers.SinkhornNormalizer(**norm_kwargs) if use_stable_normalizer \
                else sinkhorn_layers.SinkhornOriginalNormalizer(**norm_kwargs)
            , sinkhorn_layers.SinkhornOptimizer_v2(head_size, entropy_reg)
        )
    
    def _setup_permutation_extractor_v3(
        self
        , head_size: int
        , feature_size: int
        , bottleneck_features_num: int
        , entropy_reg: float
    ):

        return nn.Sequential(
            nn.Linear(feature_size * head_size, bottleneck_features_num)
            , nn.ReLU()
            , nn.Linear(bottleneck_features_num, head_size ** 2)
            , nn.ReLU()
            , utils_layers.Reshape((head_size,) * 2)
            , sinkhorn_layers.SinkhornOptimizer_v3(head_size, entropy_reg)
        )

    def _setup_permutation_extractor_v4(
        self
        , head_size: int
        , feature_size: int
        , bottleneck_features_num: int
        , entropy_reg: float
        , max_iter: int
        , tol: float
        , log: bool
        , verbose: bool
        , log_interval: int
    ):

        return nn.Sequential(
            nn.Linear(feature_size * head_size, bottleneck_features_num)
            , nn.ReLU()
            , nn.Linear(bottleneck_features_num, head_size ** 2)
            , nn.ReLU()
            , utils_layers.Reshape((head_size,) * 2)
            , sinkhorn_layers.SinkhornOptimizer_v4(
                head_size, entropy_reg
                , max_iter, tol, log, verbose
                , log_interval
            )
        )

    def _setup_permutation_extractor_v5(
        self
        , head_size: int
        , feature_size: int
        , bottleneck_features_num: int
        , entropy_reg: float
        , max_iter: int
        , tau: float
        , tol: float
        , log: bool
        , verbose: bool
        , log_interval: int
    ):

        return nn.Sequential(
            nn.Linear(feature_size * head_size, bottleneck_features_num)
            , nn.ReLU()
            , nn.Linear(bottleneck_features_num, head_size ** 2)
            , nn.ReLU()
            , utils_layers.Reshape((head_size,) * 2)
            , sinkhorn_layers.SinkhornOptimizer_v5(
                head_size, entropy_reg
                , max_iter, tau, tol, log, verbose
                , log_interval
            )
        )
