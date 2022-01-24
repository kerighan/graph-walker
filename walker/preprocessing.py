import networkx as nx
import numpy as np
from scipy.sparse import diags
from sklearn.preprocessing import normalize


def _weight_node(node, G, m, sub_sampling):
    z = G.degree(node, weight="weight") + 1
    weight = 1 / (z**sub_sampling)
    return weight


def get_normalized_adjacency(G, sub_sampling=0.1):
    A = nx.adjacency_matrix(G).astype(np.float32)
    if sub_sampling != 0:
        m = len(G.edges)
        D_inv = diags([
            _weight_node(node, G, m, sub_sampling)
            for node in G.nodes
        ])
        A = A.dot(D_inv)

    normalize(A, norm="l1", axis=1, copy=False)
    return A
