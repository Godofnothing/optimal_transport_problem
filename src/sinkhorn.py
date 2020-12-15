import sys
from pathlib import Path
from typing import Union

#import numpy 
import numpy as np

#import scipy
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

#import jax
import jax
import jax.numpy as jnp
from jax import jit

#import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

DIR_PATH = Path(__file__).parent
sys.path.append(str(Path(DIR_PATH.parent, 'utils')))
from sparsify import *


def sinkhorn():
    pass

def get_K(C, reg):
    if isinstance(C, np.ndarray):
        # Faster exponent
        K = np.divide(C, -reg)
        return np.exp(K)
    elif isinstance(C, jax.interpreters.xla.DeviceArray):
        K = jnp.divide(C, -reg)
        return jnp.exp(K)
    elif isinstance(C, torch.Tensor):
        K = torch.divide(C, -reg)
        return torch.exp(K)
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")

def get_K_stab(C, alpha, beta, reg):
    if isinstance(C, np.ndarray):
        # Faster exponent
        K = np.divide(C - alpha[:, None] - beta[None, :], -reg)
        return np.exp(K)
    elif isinstance(C, jax.interpreters.xla.DeviceArray):
        K = jnp.divide(C - alpha[:, None] - beta[None, :], -reg)
        return jnp.exp(K)
    elif isinstance(C, torch.Tensor):
        K = torch.divide(C - alpha[:, None] - beta[None, :], -reg)
        return torch.exp(K)
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")
        
def truncate_kernel(K, sparsification_strategy, thr = 1e-10, min_nnz = 5):
    if sparsification_strategy == "threshold":
        return sparsify(K, thr)
    elif sparsification_strategy == 'keep_n_largest':
        return from_largest_row_col(K, min_nnz)
    else:
        raise NotImplementedError("unknown spasification strategy")

def sinkhorn_knopp(C, reg, a = None, b = None, 
                   max_iter = 1e3, tol = 1e-9, 
                   log = False, verbose = False, log_interval = 10, 
                   make_sparse = False, sparsification_strategy = "threshold", 
                   min_nnz = 5, thr = 1e-10):
    '''
    The vanilla Sinkhorn-Knopp iterations
    
    Parameters
    ----------
    C : ndarray (dim_a, dim_b)
        the cost matrix
    reg: float
        the regularization term
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples weights in the target domain
    max_iter:
        maximal number of iterations   
    tol: float
        the addmissible value of deviation ||a_tilde - a||_2 + ||b_tilde - b||_2, 
        where a_tilde, b_tilde are estimates, obtained from the projection of the ot_matrix
    log: bool 
        whether to track or not error
    verbose: bool
        print err history
    log_interval: bool
        record error each log_interval steps
    make_sparse: bool
        if True create sparse matrix following one of the strategies
    sparsification_strategy: string
        "threshold" - eliminate element below a certain threshold
        "keep_n_largest" - keep only n_largest elements in row and column
    min_nnz (only if make_sparse = True and sparsification_strategy = "keep_n_largest"): int
        keep n_largest elements in row and column 
    thr (only if make_sparse = True and sparsification_strategy = "threshold"): float
        the threshold below which the element is eliminated
    '''
    
    if isinstance(C, jax.interpreters.xla.DeviceArray):
        K = get_K(C, reg)
        return sinkhorn_knopp_jax(K, a, b, max_iter, tol, log, verbose, log_interval)
    elif isinstance(C, np.ndarray) or isinstance(C, spsp.csr_matrix):
        K = get_K(C, reg)
        if make_sparse:
            K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)
                
        return sinkhorn_knopp_numpy(K, a, b, max_iter, tol, log, verbose, log_interval)
    elif isinstance(C, torch.Tensor) or isinstance(C, torch.sparse.FloatTensor):
        K = get_K(C, reg)
        if make_sparse:
            K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)

        return sinkhorn_knopp_torch(K, a, b, max_iter, tol, log, verbose, log_interval)
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")
        
def sinkhorn_stabilized(C, reg, a = None, b = None, 
                        max_iter = 1e3, tau=1e3, tol = 1e-9, 
                        warmstart=None, log = False, verbose = False, 
                        log_interval = 10, 
                        make_sparse = False, sparsification_strategy = "threshold", 
                        min_nnz = 5, thr = 1e-10):
    '''
    The stabilized Sinkhorn-Knopp iterations
    
    Parameters
    ----------
    For complete description look at sinkhorn_stabilized_numpy
    '''
    if isinstance(C, jax.interpreters.xla.DeviceArray):
        return sinkhorn_stabilized_jax(C, reg, a, b, max_iter, tau, tol, warmstart, log, verbose, log_interval)
    elif isinstance(C, np.ndarray) or isinstance(C, spsp.csr_matrix):
        return sinkhorn_stabilized_numpy(C, reg, a, b, max_iter, tau, tol, warmstart, log, verbose, log_interval, 
                                         make_sparse, sparsification_strategy, min_nnz, thr)
    elif isinstance(C, torch.Tensor) or isinstance(C, torch.sparse.FloatTensor):
        return sinkhorn_stabilized_torch(C, reg, a, b, max_iter, tau, tol, warmstart, log, verbose, log_interval, 
                                         make_sparse, sparsification_strategy, min_nnz, thr)
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")


def singularity_check_numpy(u, v):
    #check, whether there was a division by zero
    return np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isinf(u)) or np.any(np.isinf(v))
        
#@jit
def singularity_check_jax(u, v):
    #check, whether there was a division by zero
    return jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)) or jnp.any(jnp.isinf(u)) or jnp.any(jnp.isinf(v))

def singularity_check_torch(u: torch.Tensor, v: torch.Tensor):
    #check, whether there was a division by zero
    return torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))

def sinkhorn_knopp_numpy(K, a = None, b = None, max_iter = 1e3, tol = 1e-9, log = False, verbose = False, log_interval = 10):
    r'''
    Parameters
    ----------
    K : ndarray, shape (dim_a, dim_b)
        iteration matrix
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples weights in the target domain
    reg : float
        regularization term > 0
    '''
  
    # if the weights are not specified, assign them uniformly
    if a is None:
         a = np.ones((K.shape[0],), dtype=np.float64) / K.shape[0]
    if b is None:
         b = np.ones((K.shape[1],), dtype=np.float64) / K.shape[1]       

    # Init data
    dim_a = len(a)
    dim_b = len(b)
    
    # Set ininital values of u and v
    u = np.ones(dim_a) / dim_a
    v = np.ones(dim_b) / dim_b
    
    err = 1
    cpt = 0
    
    if log:
        log = {'err' : []}
        
    #in order not to calculate each time
    K_T = K.T
    
    while(err > tol and cpt < max_iter):
        uprev = u
        vprev = v
        
        u = a / (K @ v)
        v = b / (K_T @ u)
        
        if (singularity_check_numpy(u, v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % log_interval == 0:
            # current approximations
            a_tilde = u * (K @ v)
            b_tilde = v * (K_T @ u)
            # violation of marginals
            err = np.linalg.norm(a_tilde - a) + np.linalg.norm(b_tilde - b)
            
            if log:
                log['err'].append(err)
            if verbose:
                print(f"residual norm on iteration {cpt}: {err:.2e}\n")
        cpt += 1
                 
    #return OT matrix
    if type(K) == np.ndarray:
        ot_matrix = u[:, None] * K * v[None, :]
    elif type(K) == spsp.csr_matrix:
        ot_matrix = spsp.diags(u) @ K @ spsp.diags(v)
    if log:
        return ot_matrix, log
    else:
        return ot_matrix

#version for the dense matrices
def sinkhorn_knopp_jax(K, a = None, b = None, max_iter = 1e3, tol = 1e-9, log = False, verbose = False, log_interval = 10):
  
    # if the weights are not specified, assign them uniformly
    if a is None:
         a = jnp.ones((K.shape[0],), dtype=jnp.float32) / K.shape[0]
    if b is None:
         b = jnp.ones((K.shape[1],), dtype=jnp.float32) / K.shape[1]   

    # Init data
    dim_a = len(a)
    dim_b = len(b)
    
    # Set ininital values of u and v
    u = jnp.ones(dim_a) / dim_a
    v = jnp.ones(dim_b) / dim_b
    
    err = 1
    cpt = 0
    
    if log:
        log = {'err' : []}
        
    #in order not to calculate each time
    K_T = K.T
    
    while(err > tol and cpt < max_iter):
        # backup variables for the case of singularity
        uprev = u
        vprev = v
        
        u = a / (K @ v)
        v = b / (K_T @ u)
        
        if (singularity_check_jax(u, v)):
             # we have reached the machine precision
             # come back to previous solution and quit loop
             print('Warning: numerical errors at iteration', cpt)
             u = uprev
             v = vprev
             break
        if cpt % log_interval == 0:
            # current approximations
            a_tilde = u * (K @ v)
            b_tilde = v * (K_T @ u)
            # violation of marginals
            err = np.linalg.norm(a_tilde - a) + np.linalg.norm(b_tilde - b)
            
            if log:
                log['err'].append(err)
        cpt += 1
                 
    #return OT matrix
    ot_matrix = u[:, None] * K * v[None, :]
    if log:
        return ot_matrix, log
    else:
        return ot_matrix

def sinkhorn_knopp_torch(
    K: Union[torch.Tensor, torch.sparse.FloatTensor], a: torch.Tensor = None, b: torch.Tensor = None
    , max_iter: float = 1e3, tol: float = 1e-9, log: bool = False
    , verbose: bool = False, log_interval: int = 10
):

    # if the weights are not specified, assign them uniformly
    if a is None:
         a = torch.ones((K.shape[0],), dtype=K.dtype, device=K.device) / K.shape[0]
    if b is None:
         b = torch.ones((K.shape[1],), dtype=K.dtype, device=K.device) / K.shape[1]       

    # Init data
    dim_a = len(a)
    dim_b = len(b)
    
    # Set ininital values of u and v
    u = torch.ones(dim_a, dtype=K.dtype, device=K.device) / dim_a
    v = torch.ones(dim_b, dtype=K.dtype, device=K.device) / dim_b

    # print(K.device, a.device, b.device, u.device, v.device)
    
    err = 1
    cpt = 0
    
    if log:
        log = {'err' : []}
        
    #in order not to calculate each time
    K_T = K.t()
    
    while(err > tol and cpt < max_iter):
        uprev = u.clone()
        vprev = v.clone()
        
        u = a / (K @ v)
        v = b / (K_T @ u)
        
        if (singularity_check_torch(u, v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev.clone()
            v = vprev.clone()
            break
        if cpt % log_interval == 0:
            # current approximations
            a_tilde = u * (K @ v)
            b_tilde = v * (K_T @ u)
            # violation of marginals
            err = torch.linalg.norm(a_tilde - a) + torch.linalg.norm(b_tilde - b)
            
            if log:
                log['err'].append(err)
            if verbose:
                print(f"residual norm on iteration {cpt}: {err:.2e}\n")
        cpt += 1
                 
    #return OT matrix
    if not K.is_sparse:
        ot_matrix = u[:, None] * K * v[None, :]
    else:
        u2 = torch.diag(u)
        v2 = torch.diag(v)
        ot_matrix = u2 @ (K @ v2)
    if log:
        return ot_matrix, log
    else:
        return ot_matrix
    
def sinkhorn_stabilized_numpy(C, reg, a, b, max_iter=1000, tau=1e3, tol=1e-9,
                              warmstart=None, log=False, verbose=False, log_interval=20, 
                              make_sparse = False, sparsification_strategy = "threshold", 
                              min_nnz = 5, thr = 1e-10):
    r"""
    Solve the entropic regularization OT problem with log stabilization
    ----------
    C : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term > 0
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    max_iter : int, optional
        Max number of iterations
    tau : float
        thershold for max value in u or v for log scaling
    tol : float, optional
        Stop threshold on error (>0)
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    verbose: bool
        print err history
    log_interval: bool
        record error each log_interval steps
    make_sparse: bool
        if True create sparse matrix following one of the strategies
    sparsification_strategy: string
        "threshold" - eliminate element below a certain threshold
        "keep_n_largest" - keep only n_largest elements in row and column
    min_nnz (only if make_sparse = True and sparsification_strategy = "keep_n_largest"): int
        keep n_largest elements in row and column 
    thr (only if make_sparse = True and sparsification_strategy = "threshold"): float
        the threshold below which the element is eliminated
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """
    if a is None:
         a = np.ones((С.shape[0],), dtype=np.float64) / С.shape[0]
    if b is None:
         b = np.ones((С.shape[1],), dtype=np.float64) / С.shape[1]       

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else:
        alpha, beta = warmstart

    u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return u[:, None] * get_K_stab(C, alpha, beta, reg) * v[None, :]

    K = get_K_stab(C, alpha, beta, reg)
    if make_sparse:
                K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)
            
    cpt = 0
    err = 1
    while True:
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (K.T @ u + 1e-16)
        u = a / (K @ v   + 1e-16)

        #print(np.abs(u).max(), np.abs(v).max(), np.abs(u).min(), np.abs(v).min())

        # absorption iteration
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
            u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
            K = get_K_stab(C, alpha, beta, reg)
            if make_sparse:
                K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)

        if cpt % log_interval == 0:
            # we can speed up the process by checking for the error only log_interval iterations
            gamma = get_Gamma(alpha, beta, u, v)
            err = np.linalg.norm((np.sum(gamma, axis=0) - b)) + np.linalg.norm((np.sum(gamma, axis=1) - a))
            if log:
                log['err'].append(err)

        if err <= tol or cpt >= max_iter:
            break

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1
        
    if log:
        log['alpha'] = alpha + reg * np.log(u)
        log['beta']  = beta  + reg * np.log(v)
        return get_Gamma(alpha, beta, u, v), log
    else:
        return get_Gamma(alpha, beta, u, v)
    
def sinkhorn_stabilized_jax(C, reg, a, b, max_iter=1000, tau=1e3, tol=1e-9,
                            warmstart=None, log=False, verbose=False, log_interval=20):
    r"""
    Solve the entropic regularization OT problem with log stabilization
    ----------
    C : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term > 0
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    max_iter : int, optional
        Max number of iterations
    tau : float
        thershold for max value in u or v for log scaling
    tol : float, optional
        Stop threshol on error (>0)
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    log_interval: bool
        record error each log_interval steps
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """
    if a is None:
         a = jnp.ones((С.shape[0],), dtype = jnp.float32) / С.shape[0]
    if b is None:
         b = jnp.ones((С.shape[1],), dtype = jnp.float32) / С.shape[1]       

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = jnp.zeros(dim_a), jnp.zeros(dim_b)
    else:
        alpha, beta = warmstart

    u, v = jnp.ones(dim_a) / dim_a, jnp.ones(dim_b) / dim_b

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return u[:, None] * get_K_stab(C, alpha, beta, reg) * v[None, :]

    K = get_K_stab(C, alpha, beta, reg)
    
    cpt = 0
    err = 1
    while True:
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / ((K.T @ u) + 1e-16)
        u = a / ((K @ v)   + 1e-16)

        # remove numerical problems and store them in K
        if jnp.abs(u).max() > tau or jnp.abs(v).max() > tau:
            alpha, beta = alpha + reg * jnp.log(u), beta + reg * jnp.log(v)
            u, v = jnp.ones(dim_a) / dim_a, jnp.ones(dim_b) / dim_b
            K = get_K_stab(C, alpha, beta, reg)

        if cpt % log_interval == 0:
            # we can speed up the process by checking for the error only log_interval iterations
            gamma = get_Gamma(alpha, beta, u, v)
            err = jnp.linalg.norm((jnp.sum(gamma, axis=0) - b)) + jnp.linalg.norm((jnp.sum(gamma, axis=1) - a))
            if log:
                log['err'].append(err)

        if err <= tol or cpt >= max_iter:
            break

        if jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1
        
    if log:
        log['alpha'] = alpha + reg * jnp.log(u)
        log['beta']  = beta  + reg * jnp.log(v)
        return get_Gamma(alpha, beta, u, v), log
    else:
        return get_Gamma(alpha, beta, u, v)


def sinkhorn_stabilized_torch(C, reg, a, b, max_iter=1000, tau=1e3, tol=1e-9,
                              warmstart=None, log=False, verbose=False, log_interval=20, 
                              make_sparse = False, sparsification_strategy = "threshold", 
                              min_nnz = 5, thr = 1e-10):

    if a is None:
         a = torch.ones((C.shape[0],), dtype=C.dtype, device=C.device) / C.shape[0]
    if b is None:
         b = torch.ones((C.shape[1],), dtype=C.dtype, device=C.device) / C.shape[1]       

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha = torch.zeros(dim_a, dtype=C.dtype, device=C.device)
        beta = torch.zeros(dim_b, dtype=C.dtype, device=C.device)
    else:
        alpha, beta = warmstart

    u = torch.ones(dim_a, dtype=C.dtype, device=C.device) / dim_a
    v = torch.ones(dim_b, dtype=C.dtype, device=C.device) / dim_b

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return u[:, None] * get_K_stab(C, alpha, beta, reg) * v[None, :]

    K = get_K_stab(C, alpha, beta, reg)
    if make_sparse:
        K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)
            
    cpt = 0
    err = 1
    while True:
        uprev = u.clone()
        vprev = v.clone()

        # sinkhorn update
        v = b / (K.t() @ u + 1e-16)
        u = a / (K @ v   + 1e-16)

        #print(np.abs(u).max(), np.abs(v).max(), np.abs(u).min(), np.abs(v).min())

        # absorption iteration
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
            u = torch.ones(dim_a, dtype=C.dtype, device=C.device) / dim_a
            v = torch.ones(dim_b, dtype=C.dtype, device=C.device) / dim_b
            K = get_K_stab(C, alpha, beta, reg)
            if make_sparse:
                K = truncate_kernel(K, sparsification_strategy, thr, min_nnz)

        if cpt % log_interval == 0:
            # we can speed up the process by checking for the error only log_interval iterations
            gamma = get_Gamma(alpha, beta, u, v)
            err = torch.linalg.norm((torch.sum(gamma, dim=0) - b)) + torch.linalg.norm((torch.sum(gamma, dim=1) - a))
            if log:
                log['err'].append(err)

        if err <= tol or cpt >= max_iter:
            break

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev.clone()
            v = vprev.clone()
            break

        cpt = cpt + 1
        
    if log:
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta']  = beta  + reg * torch.log(v)
        return get_Gamma(alpha, beta, u, v), log
    else:
        return get_Gamma(alpha, beta, u, v)
    

def sinkhorn_epsilon_scaling(C, a, b, eps_fn, max_outer_iter=100, min_outer_iter = 10, max_inner_iter=100, 
                             tau = 1e3, tol=1e-9, warmstart=None, log=False, verbose=False, log_interval=10,
                             make_sparse = False, sparsification_strategy = "threshold", 
                             min_nnz = 5, thr = 1e-10):
    r"""
    Solve the entropic regularization optimal transport problem with log
    stabilization and epsilon scaling.
    
    Parameters
    ----------
    C : ndarray, shape (dim_a, dim_b)
        loss matrix
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    eps_fn : function of callable object
        the epsilon scaling policy
    max_outer_iter : int, optional
        Max number of iterations
    min_outer_iter: int, optional
        Min number of iterations
    max_inner_iter : int, optional
        Max number of iterations in the inner slog stabilized sinkhorn
    tau : float
        thershold for max value in u or v for log scaling
    tol : float, optional
        Stop threshol on error (>0)
    warmstart : tuple of vectors
        if given then sarting values for alpha an beta log scalings
    log : bool, optional
        record log if True
    verbose : bool, optional
        Print information along iterations
    log_interval: bool
        record error each log_interval steps
    make_sparse: bool
        if True create sparse matrix following one of the strategies
    sparsification_strategy: string
        "threshold" - eliminate element below a certain threshold
        "keep_n_largest" - keep only n_largest elements in row and column
    min_nnz (only if make_sparse = True and sparsification_strategy = "keep_n_largest"): int
        keep n_largest elements in row and column 
    thr (only if make_sparse = True and sparsification_strategy = "threshold"): float
        the threshold below which the element is eliminated
    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """

    if a is None:
         a = np.ones((K.shape[0],), dtype=np.float64) / K.shape[0]
    if b is None:
         b = np.ones((K.shape[1],), dtype=np.float64) / K.shape[1] 

    # init data
    dim_a = len(a)
    dim_b = len(b)

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else:
        alpha, beta = warmstart

    cpt = 0
    err = 1
    while True:
        reg_i = eps_fn(cpt)

        G, log_i = sinkhorn_stabilized(C, reg_i, a, b,
                                       max_iter = max_inner_iter, tau=tau, tol=1e-9,
                                       warmstart=(alpha, beta), 
                                       log=True, verbose=False, log_interval = 20,
                                       make_sparse = make_sparse, 
                                       sparsification_strategy = sparsification_strategy, 
                                       min_nnz = min_nnz, thr = thr)
        
        alpha = log_i['alpha']
        beta = log_i['beta']
        err = log_i['err'][-1]

        if cpt % (log_interval) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if log:
                log['err'].append(err)

            if verbose:
                print(f'iteration {cpt:5d}\terr{err:8e}')
                    
        if cpt > max_outer_iter:
            break

        if err <= tol and cpt > min_outer_iter:
            break

        cpt = cpt + 1

    if log:
        return G, log
    else:
        return G