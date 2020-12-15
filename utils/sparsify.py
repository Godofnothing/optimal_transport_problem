import scipy.sparse as spsp
import numpy as np
import torch

def sparsify(C, threshold):
    '''
    Setting to zero elements below a given threshold
    and returning the resulting csr_matrix
    '''

    if isinstance(C, np.ndarray):
        sp_C = spsp.coo_matrix((C[C > threshold], np.where(C > threshold)), shape=C.shape).tocsr()
    elif isinstance(C, torch.Tensor):
        sp_C = torch.sparse.FloatTensor(
            torch.nonzero(C > threshold).t()
            , C[C > threshold]
            , C.shape
        )

    return sp_C

def from_largest_row_col(K, n):
    '''
    Keep only n-largest elements in the row or column
    and returning the resulting csr_matrix
    '''

    if isinstance(K, np.ndarray):
        cond = np.bitwise_or(
            K >= -np.partition(-K, n - 1, axis = 1)[:, n - 1][:, None], 
            K >= -np.partition(-K, n - 1, axis = 0)[n - 1, :][None, :]
        )
    else:
        raise NotImplementedError()
    
    return spsp.coo_matrix((K[cond], np.where(cond)), shape = K.shape).tocsr()
