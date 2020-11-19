from .utils import cumsum_probability_matrix
from scipy.sparse import diags
import networkx as nx
import numpy as np


def _preprocess_graph(G, num_nodes):
    A = nx.adjacency_matrix(G).astype(np.float32)
    D_inv = diags([1. / G.degree(node) for node in G.nodes])
    P = D_inv.dot(A)
    return cumsum_probability_matrix(P.indptr, P.data, P.indices, num_nodes)
