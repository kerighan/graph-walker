from .walks import (
    _random_walks,
    _node2vec_random_walks,
    _corrupt_walks,
    _weighted_corrupt_walks)
from .preprocess import _preprocess_graph
import networkx as nx
import numpy as np
import time


def node2vec_random_walks(
    G, p=.25, q=.25,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.1,
    verbose=True):
    start_time = time.time()

    n_nodes = len(G.nodes)
    indptr, data, indices = _preprocess_graph(
        G, n_nodes, sub_sampling=sub_sampling)
    walks = _node2vec_random_walks(
        indptr, data, indices, n_nodes,
        n_walks, walk_len, p, q)
    
    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def random_walks(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.1,
    verbose=True
):
    start_time = time.time()

    n_nodes = len(G.nodes)
    indptr, data, indices = _preprocess_graph(
        G, n_nodes, sub_sampling=sub_sampling)
    walks = _random_walks(
        indptr, data, indices, n_nodes,
        n_walks, walk_len)
    
    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def corrupt(G, walks, r=.1, use_degree=True, verbose=True):
    start_time = time.time()
    n_nodes = len(G.nodes)

    A = nx.adjacency_matrix(G)
    adj_indptr = A.indptr
    adj_indices = A.indices

    if use_degree:
        degree = np.array(A.sum(axis=1), dtype=np.float32).reshape((n_nodes,))
        degree = degree.cumsum()
        degree /= degree.max()
    
        walks, similarity = _weighted_corrupt_walks(
            walks, adj_indptr, adj_indices, n_nodes, degree, p=r)
    else:
        walks, similarity = _corrupt_walks(
            walks, adj_indptr, adj_indices, n_nodes, p=r)

    if verbose:
        duration = time.time() - start_time
        print(f"Corrupt random walks - T={duration:.2f}s")
    return walks, similarity
