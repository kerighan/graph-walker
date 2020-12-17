from .utils import cumsum_probability_matrix
from sklearn.preprocessing import normalize
from scipy.sparse import diags
import networkx as nx
import numpy as np
import math
import time


def _degree_inv_diagonal(node, G, m, sub_sampling):
    z = G.degree(node, weight="weight") + 1
    factor = 1 / (z**sub_sampling)
    return factor


def _preprocess_graph(G, n_nodes, sub_sampling=0.1):
    A = nx.adjacency_matrix(G).astype(np.float32)
    if sub_sampling != 0:
        m = len(G.edges)
        D_inv = diags([
            _degree_inv_diagonal(node, G, m, sub_sampling)
            for node in G.nodes
        ])
        A = A.dot(D_inv)
    
    normalize(A, norm="l1", axis=1, copy=False)
    res = cumsum_probability_matrix(A.indptr, A.data, A.indices, n_nodes)
    return res
