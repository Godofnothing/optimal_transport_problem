import numpy as np

import jax
import jax.numpy as jnp

import scipy.sparse as spsp

def get_cost(G, C):
    '''
    G: ndarray
        transportation plan
    C: ndarray
        cost matrix
    '''
    if isinstance(G, np.ndarray):
        return np.sum(G * C)
    elif isinstance(G, jax.interpreters.xla.DeviceArray):
        return jnp.sum(G * C)
    elif  isinstance(G, spsp.csr_matrix):
        return np.sum(G.toarray() * C)