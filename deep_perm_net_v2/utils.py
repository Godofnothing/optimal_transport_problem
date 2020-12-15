import os
import sys
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


def get_mean_accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    y = y.type(torch.int)

    n, m = y_hat.shape[-2:]
    assert n == m

    e = np.ones((n, 1))
    Q = cp.Parameter((n, n))
    P_hat = cp.Variable((n, n), boolean=True)

    objective = cp.Minimize(
        cp.norm(P_hat - Q, p='fro')
    )
    constraints = [
        P_hat @ e == e
        , e.T @ P_hat == e.T
    ]
    problem = cp.Problem(objective, constraints)

    # Iterate through batch
    accuracy_batch = torch.empty(y_hat.shape[0])
    for i, y_hat_i in enumerate(y_hat):
        Q.value = y_hat_i.cpu().detach().numpy()
        problem.solve(solver='ECOS_BB')

        acc = 0
        # print(problem.status)
        if problem.status == 'optimal':
            y_hat_i_bin = torch.as_tensor(P_hat.value > 1e-8, dtype=torch.int)
            acc = (y_hat_i_bin * y[i]).sum() / n

        accuracy_batch[i] = acc

    return accuracy_batch.mean()
