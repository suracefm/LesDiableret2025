import numpy as np
import scipy.sparse as spa
from functools import reduce
from typing import List, Tuple, Dict, Sequence
from numpy.typing import NDArray


def operator(sites: Sequence[int], matrices: Sequence[NDArray], L: int, dtype=None) -> spa.csr_matrix:
    # Construct a sparse matrix corresponding to the many-body operator on the Hilbert space of L qubits:
    # A_{i}\otimes B_{j} \otimes C_{k}...
    # where matrices = [A, B, C, ...], sites = [i, j, k] and L is the system size
    if dtype is None:
        dtype = float
    for matrix in matrices:
        if not np.isrealobj(matrix):
            dtype = complex        
            
    dim = 2**L    
    row_idx, col_idx, data_val = [], [], []
    row_dict, data_dict = matrixtodicts(matrices)
    
    for cstate in range(dim):
        j = get_bits(cstate, sites)
        for i, data in zip(row_dict[j], data_dict[j]):
            rstate = flip_bits(cstate, sites, j^i)
            row_idx.append(rstate)
            col_idx.append(cstate)
            data_val.append(data)
    
    return spa.coo_matrix((data_val, (row_idx, col_idx)),\
                          shape=(dim, dim), dtype=dtype)


def get_bits(num: int, sites: Sequence[int]) -> int:
    j = 0
    for site in sites:
        j = (j<<1) + ((num >> site) & 1)
    return j


def flip_bits(state: int, sites: Sequence[int], which: int) -> int:
    temp = 0
    for site in reversed(sites):
        if (which&1):
            temp+=(1<<site)
        which = which>>1
    return state^temp


def matrixtodicts(matrices: Sequence[np.ndarray]) -> Tuple[Dict, Dict]:
    if len(matrices)>1:
        matrix = reduce(np.kron, matrices)
    else:
        matrix = matrices[0]
    size = matrix.shape[0]
    matrix = spa.coo_matrix(matrix)

    data_dict = {i:[] for i in range(size)}
    row_dict = {i:[] for i in range(size)}
    for row, col, data in zip (matrix.row, matrix.col, matrix.data):
        data_dict[col].append(data)
        row_dict[col].append(row)
    return row_dict, data_dict

