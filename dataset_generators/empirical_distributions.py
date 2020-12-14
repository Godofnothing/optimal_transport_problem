import numpy as np

import jax
import jax.numpy as jnp

def generate_1d_normal_numpy(n_samples, mu = 0.0, sigma = 1.0):
    '''
    generate points on the line from normal distribution
    
    n_samples : int
        number of points to be generated
    mu : float
        mean of normal distribution
    sigma : float
        standard deviation of normal distribution
    '''
    X = np.random.normal(loc = mu, scale = sigma, size = n_samples)
    X.sort()
    return X[:, None]

def generate_1d_normal_jax(n_samples, mu = 0.0, sigma = 1.0, seed = 7):
    '''
    generate points on the line from normal distribution
    
    n_samples : int
        number of points to be generated
    mu : float
        mean of normal distribution
    sigma : float
        standard deviation of normal distribution
    seed: int
        random seed for random number generator
    '''
    key = jax.random.PRNGKey(seed)
    X = mu + sigma * jax.random.normal(key = key, shape = (n_samples,)) 
    X.sort()
    return X[:, None]

def generate_on_square_grid(num_points, n_x, a_x):
    '''
    generate points on the square multidimensional grid
    
    n_x : array-like
        number of lattice steps
    a_x : array-like
        lattice spacings
    num_points : int
        number of points
    '''
    if isinstance(a_x, np.ndarray):
        return np.hstack(
            [a * np.random.randint(low = 0, high = n, size = (num_points, 1)) for n, a in zip(n_x, a_x)]
        )
    elif isinstance(a_x, jax.interpreters.xla.DeviceArray):
        key = jax.random.PRNGKey(7)
        
        return np.hstack(
            [a * jnp.random.randint(low = 0, high = n, shape = (num_points, 1)) for n, a in zip(n_x, a_x)]
        )
    else:
         raise NotImplementedError(f"The type {type(a_x[-1])} is not supported!")
        
def make_blobs(num_points, proportions, means, covariances):
    '''
    sample mixture of gaussians
    
    num_points: int
        number of points to be generated
    proportions:
        the proportion of each mixture in the total distribution.
        The total sum must be equal to 1.
    means: list(ndarray)
        means of the gaussian distributions
    covariances: list(ndarray)
        covariances of the gaussian distributions
    '''
    assert sum(proportions) == 1
    
    if isinstance(means[-1], np.ndarray):
        return np.vstack(
            [np.random.multivariate_normal(mean, cov, size = int(p * num_points)) for p, mean, cov in zip(proportions, means, covariances)]
        )
    elif isinstance(means[-1], jax.interpreters.xla.DeviceArray):
        key = jax.random.PRNGKey(7)
        
        return jnp.vstack(
            [jnp.random.multivariate_normal(key, mean, cov, shape = int(p * num_points)) for p, mean, cov in zip(proportions, means, covariances)]
        )
    else:
         raise NotImplementedError(f"The type {type(means[-1])} is not supported!")