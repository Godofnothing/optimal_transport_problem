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
sys.path.append(str(Path(DIR_PATH.parent, 'src')))
import sinkhorn


class SinkhornOriginalNormalizer(nn.Module):
    def __init__(self, eps: float=1e-3, L: int=100):
        super().__init__()
        self.eps = eps
        self.L = L

    def forward(self, x):
        x += self.eps
        for _ in range(self.L):
            # row normalization
            x = x / torch.sum(x, dim=-2, keepdims=True)
            # column normalization
            x = x / torch.sum(x, dim=-1, keepdims=True)

        return x


class SinkhornNormalizer(nn.Module):
    """
        Numerically stable version of casting matrix to double stochastic
        https://www.groundai.com/project/learning-permutations-with-sinkhorn-policy-gradient/1#S4.SS2
    """

    def __init__(self, eps: float=1e-3, L: int=100, tau: float=1):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.L = L
            
    def forward(self, x):
        x /= self.tau

        for _ in range(self.L):
            # row normalization
            x = x - torch.logsumexp(x, dim=-2, keepdims=True)
            # column normalization
            x = x - torch.logsumexp(x, dim=-1, keepdims=True)
            
        # add a small offset ’eps’ to avoid numerical errors due to exp()
        return torch.exp(x) + self.eps


class SinkhornOptimizer_v1(nn.Module):
    def forward(self, X: torch.Tensor):
        return X


class SinkhornOptimizer_v2(nn.Module):
    def __init__(self, head_size: int, entropy_reg: float):
        super().__init__()
        e = np.ones((head_size, 1))
        Q = cp.Parameter((head_size, head_size))
        P_hat = cp.Variable((head_size, head_size))
        
        objective = cp.Minimize(
            cp.norm(P_hat - Q, p='fro') - entropy_reg * cp.sum(cp.entr(P_hat))
        )
        constraints = [
            P_hat @ e == e
            , e.T @ P_hat == e.T
            , P_hat >= 0
            , P_hat <= 1
        ]
        problem = cp.Problem(objective, constraints)
        
        self.model = CvxpyLayer(problem, parameters=[Q], variables=[P_hat])
        
    def forward(self, X: torch.Tensor):
        output, = self.model(X)

        return output

    
class SinkhornOptimizer_v3(nn.Module):
    def __init__(self, head_size: int, entropy_reg: float):
        super().__init__()
        # Generate data.
        e = np.ones((head_size, 1))
        M = cp.Parameter((head_size, head_size))
        P_hat = cp.Variable((head_size, head_size))

        objective = cp.Minimize(
            cp.norm(cp.multiply(P_hat, M), p='fro') - cp.sum(cp.entr(P_hat))
        )
        constraints = [
            P_hat @ e == e
            , e.T @ P_hat == e.T
            , P_hat >= 0
        ]
        problem = cp.Problem(objective, constraints)

        self.model = CvxpyLayer(problem, parameters=[M], variables=[P_hat])
        
    def forward(self, X: torch.Tensor):
        output, = self.model(X)
        return output


class SinkhornOptimizer_v4(nn.Module):
    def __init__(
        self
        , head_size: int
        , entropy_reg: float
        , max_iter: int=1e2
        , tol: float=1e-6
        , log: bool=False
        , verbose: bool=False
        , log_interval: int=10
    ):

        super().__init__()
        self.head_size = head_size
        self.entropy_reg = entropy_reg
        self.max_iter = max_iter
        self.tol = tol
        self.log = log
        self.verbose = verbose
        self.log_interval = log_interval
        
    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]

        a = torch.ones(self.head_size, dtype=X.dtype, device=X.device)
        b = a.clone()

        for i in range(batch_size):
            X[i] = sinkhorn.sinkhorn_knopp(
                X[i], self.entropy_reg, a, b, max_iter=self.max_iter
                , tol=self.tol, log=self.log, verbose=self.verbose
                , log_interval=self.log_interval
            )

        return X


class SinkhornOptimizer_v5(nn.Module):
    def __init__(
        self
        , head_size: int
        , entropy_reg: float
        , max_iter: int=1e3
        , tau: float=1e3
        , tol: float=1e-9
        , log: bool=False
        , verbose: bool=False
        , log_interval: int=10
    ):

        super().__init__()
        self.head_size = head_size
        self.entropy_reg = entropy_reg
        self.max_iter = max_iter
        self.tau = tau
        self.tol = tol
        self.log = log
        self.verbose = verbose
        self.log_interval = log_interval
        
    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]

        a = torch.ones(self.head_size, dtype=X.dtype, device=X.device)
        b = a.clone()

        for i in range(batch_size):
            X[i] = sinkhorn.sinkhorn_stabilized(
                X[i], self.entropy_reg, a, b, max_iter=self.max_iter, tau=self.tau
                , tol=self.tol, log=self.log, verbose=self.verbose
                , log_interval=self.log_interval
            )

        return X
