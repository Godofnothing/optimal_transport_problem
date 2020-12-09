import scipy.sparse as spsp
import numpy as np

def sparsify(C, threshold):
    '''
    Setting to zero elements below a given threshold
    and returning the resulting csr_matrix
    '''
    return spsp.coo_matrix((C[C > threshold], np.where(C > threshold)), shape = C.shape).tocsr()

def from_largest_row_col(K, n):
    '''
    Keep only n-largest elements in the row or column
    and returning the resulting csr_matrix
    '''
    cond = np.bitwise_or(
        K >= -np.partition(-K, n - 1, axis = 1)[:, n - 1][:, None], 
        K >= -np.partition(-K, n - 1, axis = 0)[n - 1, :][None, :]
    )
    
    return spsp.coo_matrix((K[cond], np.where(cond)), shape = K.shape).tocsr()