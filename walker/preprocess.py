from .utils import cumsum_probability_matrix
from scipy.sparse import diags
import networkx as nx
import numpy as np
import time


def _preprocess_graph(G, n_nodes):
    A = nx.adjacency_matrix(G).astype(np.float32)
    D_inv = diags([
        1. / max(G.degree(node, weight="weight"), .1)
        for node in G.nodes
    ])
    P = D_inv.dot(A)

    res = cumsum_probability_matrix(P.indptr, P.data, P.indices, n_nodes)
    return res
