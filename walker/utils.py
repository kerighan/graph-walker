from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def cumsum_probability_matrix(indptr, data, indices, n_nodes):
    for i in prange(n_nodes):
        start = indptr[i]
        end = indptr[i + 1]
        data[start:end] = np.cumsum(data[start:end])
    return indptr, data, indices


@jit(nopython=True, nogil=True, fastmath=True)
def isin(element, array):
    for item in array:
        if element == item:
            return True
    return False
