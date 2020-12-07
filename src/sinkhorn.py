#import numpy 
import numpy as np

#import scipy
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

#import jax
import jax
import jax.numpy as jnp
from jax import jit

def sinkhorn():
    pass

def get_K(C, reg):
    if isinstance(C, np.ndarray):
        # Faster exponent
        K = np.divide(C, -reg)
        K = np.exp(K)
        return K
    elif isinstance(C, jax.interpreters.xla.DeviceArray):
        K = jnp.divide(C, -reg)
        K = jnp.exp(K)
        return K
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")

def sparsify(K, eps, min_nnz_row = 5):
    '''
    This function eliminates the elements which are smaller that the maximal element in the matrix,
    but keeps some fixed number of elements, specified by the user. 
    
    Parameters
    ----------
    K : ndarray
        \exp(-C / \gamma) - the input matrix
    eps:
        the threshold below which the element is eliminated, unless preserved by the another condition
    min_nnz_row:
        the minimal number of elements to be preserved
    
    '''
    return spsp.csr_matrix(
         np.where(np.bitwise_or(
             K > eps * K.max(), K >= -np.partition(-K, min_nnz_row - 1, axis = 1)[:, min_nnz_row - 1]), K, 0
         )
    )

def sinkhorn_knopp(C, reg, a = None, b = None, 
                   max_iter = 1e3, eps = 1e-9, 
                   log = False, verbose = False, log_interval = 10, 
                   make_sparse = False, elim_eps = 1e-2, min_nnz_row = 5):
    '''
    This function eliminates the elements which are smaller that the maximal element in the matrix,
    but keeps some fixed number of elements, specified by the user. 
    
    Parameters
    ----------
    C : ndarray
        the const matrix
    reg:
        the regularization term
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples weights in the target domain
    max_iter:
        maximal number of iterations
        
    make_sparse:
        if True remove small elements from the matrix
    elim_eps (only if make_sparse = True):
        the threshold below which the element is eliminated, unless preserved by the another condition
    min_nnz_row (only if make_sparse = True):
        the minimal number of elements to be preserved
    
    '''
    
    if isinstance(C, jax.interpreters.xla.DeviceArray):
        K = get_K(C, reg)
        return sinkhorn_knopp_jax(K, a, b, max_iter, eps, log, verbose, log_interval)
    elif isinstance(C, np.ndarray) or isinstance(C, spsp.csr_matrix):
        K = get_K(C, reg)
        if make_sparse:
            K = sparsify(K, elim_eps, min_nnz_row)
        return sinkhorn_knopp_numpy(K, a, b, max_iter, eps, log, verbose, log_interval)
    else:
        raise NotImplementedError(f"The type {type(C)} is not supported!")

def singularity_check_numpy(u, v):
    #check, whether there was a division by zero
    return np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isinf(u)) or np.any(np.isinf(v))
        
#@jit
def singularity_check_jax(u, v):
    #check, whether there was a division by zero
    return jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)) or jnp.any(jnp.isinf(u)) or jnp.any(jnp.isinf(v))

def sinkhorn_knopp_numpy(K, a = None, b = None, max_iter = 1e3, eps = 1e-9, log = False, verbose = False, log_interval = 10):
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
    
    r = np.empty_like(b)
    
    Kp = (1 / a) * K
    err = 1
    cpt = 0
    
    if log:
        log = {'err' : []}
    
    while(err > eps and cpt < max_iter):
        uprev = u
        vprev = v
        
        v = 1. / (K.T @ u)
        u = 1. / (Kp @ v)
        
        if (singularity_check_numpy(u, v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % log_interval == 0:
            #residual on the iteration
            r = (u @ K) * v 
            # violation of marginal
            err = np.linalg.norm(r - b)  
            
            if log:
                log['err'].append(err)
            if verbose:
                print(f"residual norm on iteration {cpt}: {err:.2e}\n")
        cpt += 1
                 
    #return OT matrix
    if type(K) == np.ndarray:
        ot_matrix = u[:, None] * K * v[None, :]
    if type(K) == spsp.csr_matrix:
        raise NotImplementedError("so far the version for scipy sparse matrices in not working :(")
    if log:
        return ot_matrix, log
    else:
        return ot_matrix

#version for the dense matrices
def sinkhorn_knopp_jax(K, a = None, b = None, max_iter = 1e3, eps = 1e-9, log = False, verbose = False, log_interval = 10):
  
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
    
    r = jnp.empty_like(b)
    Kp = (1 / a).reshape(-1, 1) * K
    err = 1
    cpt = 0
    
    if log:
        log = {'err' : []}
    
    while(err > eps and cpt < max_iter):
        # backup variables for the case of singularity
        uprev = u
        vprev = v
        
        v = 1. / (K.T @ u)
        u = 1. / (Kp @ v)
        
        if (singularity_check_jax(u, v)):
             # we have reached the machine precision
             # come back to previous solution and quit loop
             print('Warning: numerical errors at iteration', cpt)
             u = uprev
             v = vprev
             break
        if cpt % log_interval == 0:
            #residual on the iteration
            r = (u @ K) * v 
            # violation of marginal
            err = jnp.linalg.norm(r - b)  
            
            if log:
                log['err'].append(err)
        cpt += 1
                 
    #return OT matrix
    ot_matrix = u[:, None] * K * v[None, :]
    if log:
        return ot_matrix, log
    else:
        return ot_matrix
    
def sinkhorn_stabilized_numpy(C, reg, a, b, max_iter=1000, tau=1e3, eps=1e-9,
                              warmstart=None, verbose=False, log_interval=20, log=False):
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
    tau : float
        thershold for max value in u or v for log scaling
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    max_iter : int, optional
        Max number of iterations
    eps : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
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

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(-(C - alpha.reshape((dim_a, 1))
                        - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return np.exp(-(C - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + np.log(u.reshape((dim_a, 1))) + np.log(v.reshape((1, dim_b))))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / ((K.T @ u) + 1e-16)
        u = a / ((K @ v)   + 1e-16)

        # remove numerical problems and store them in K
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
            u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
            K = get_K(alpha, beta)

        if cpt % log_interval == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = get_Gamma(alpha, beta, u, v)
            err = np.linalg.norm((np.sum(transp, axis=0) - b))
            if log:
                log['err'].append(err)

        if err <= eps:
            loop = False

        if cpt >= max_iter:
            loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    if log:
        logu = alpha / reg + np.log(u)
        logv = beta / reg + np.log(v)
        log['logu'] = logu
        log['logv'] = logv
        log['alpha'] = alpha + reg * np.log(u)
        log['beta'] = beta + reg * np.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        
        return get_Gamma(alpha, beta, u, v), log
    else:
        return get_Gamma(alpha, beta, u, v)