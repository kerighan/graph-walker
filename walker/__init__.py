import time

import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from sklearn.preprocessing import normalize
from _walker import node2vec_random_walks as _node2vec_random_walks
from _walker import random_walks as _random_walks
from _walker import random_walks_with_restart as _random_walks_with_restart
from _walker import random_walks_with_weights as _random_walks_with_weights
from _walker import weighted_corrupt as _corrupt

from .preprocessing import (
    get_normalized_adjacency,
    get_normalized_adjacency_and_original,
)


def random_walks(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.0,
    p=1,
    q=1,
    alpha=0,
    start_nodes=None,
    verbose=True,
):
    start_time = time.time()

    A = get_normalized_adjacency(G, sub_sampling=sub_sampling)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)

    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    if p == 1 and q == 1:
        if alpha == 0:
            walks = _random_walks(indptr, indices, data, start_nodes, n_walks, walk_len)
        else:
            walks = _random_walks_with_restart(
                indptr, indices, data, start_nodes, n_walks, walk_len, alpha
            )
    else:
        walks = _node2vec_random_walks(
            indptr, indices, data, start_nodes, n_walks, walk_len, p, q
        )

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def random_walks_with_weights(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.0,
    p=1,
    q=1,
    alpha=0,
    start_nodes=None,
    verbose=True,
):
    start_time = time.time()

    A, original_data = get_normalized_adjacency_and_original(
        G, sub_sampling=sub_sampling
    )
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)

    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    walks = _random_walks_with_weights(
        indptr, indices, data, original_data, start_nodes, n_walks, walk_len
    )

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def random_walks_maximum_entropy(
    G, n_walks=10, walk_len=10, sub_sampling=0.0, start_nodes=None, verbose=True
):
    start_time = time.time()

    # Step 1: Get the normalized adjacency matrix
    A = get_normalized_adjacency(G, sub_sampling=sub_sampling)

    # Step 2: Compute the largest eigenvalue and eigenvector of the adjacency matrix
    largest_eigenvalue, eigenvector = eigs(A, k=1, which="LM")
    largest_eigenvalue = largest_eigenvalue.real[0]
    eigenvector = eigenvector.real.flatten()

    # Step 3: Modify transition probabilities according to MERW using sparse operations
    # Scale each row by the corresponding eigenvector entry
    row_scaling = diags(1.0 / (largest_eigenvalue * eigenvector))
    col_scaling = diags(eigenvector)
    P = row_scaling @ A @ col_scaling  # This applies MERW scaling

    # Normalize each row to ensure it's a valid probability distribution
    P = normalize(P, norm="l1", axis=1)  # Ensures each row sums to 1

    # Prepare data for random walk functions
    indptr = P.indptr.astype(np.uint32)
    indices = P.indices.astype(np.uint32)
    data = P.data.astype(np.float32)

    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    # Run the random walks with weighted probabilities
    walks = _random_walks_with_weights(
        indptr, indices, data, data, start_nodes, n_walks, walk_len
    )

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks (Maximal Entropy) - T={duration:.2f}s")
    return walks


def corrupt(G, walks, r=0.01, ns_exponent=0.75, negative_size=100000, verbose=True):
    # corrupt random walks
    start_time = time.time()

    n_nodes = len(G.nodes)
    A = nx.adjacency_matrix(G)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)

    # compute weights for each node
    weights = np.array([G.degree(node, "weight") for node in G.nodes], dtype=np.float32)
    weights **= ns_exponent
    weights /= weights.sum()

    # draw negative table
    neg = np.random.choice(range(n_nodes), size=negative_size, p=weights, replace=True)

    # corrupt random walks
    similarity = _corrupt(walks, neg, n_nodes, r)

    if verbose:
        elapsed = time.time() - start_time
        print(f"Corrupt random walks - T={elapsed:.02}s")

    return similarity


def corrupted_random_walks(
    G,
    n_walks=10,
    walk_len=10,
    sub_sampling=0.0,
    p=1,
    q=1,
    r=0.1,
    ns_exponent=0.75,
    negative_size=100000,
    start_nodes=None,
    verbose=True,
):
    start_time = time.time()

    n_nodes = len(G.nodes)
    A = get_normalized_adjacency(G, sub_sampling=sub_sampling)
    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)

    if start_nodes is None:
        start_nodes = np.arange(len(G.nodes)).astype(np.uint32)
    else:
        start_nodes = np.array(start_nodes, dtype=np.uint32)

    if p == 1 and q == 1:
        walks = _random_walks(indptr, indices, data, start_nodes, n_walks, walk_len)
    else:
        walks = _node2vec_random_walks(
            indptr, indices, data, start_nodes, n_walks, walk_len, p, q
        )

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    # corrupt random walks
    start_time = time.time()

    # compute weights for each node
    weights = np.array([G.degree(node, "weight") for node in G.nodes], dtype=np.float32)
    weights **= ns_exponent
    weights /= weights.sum()

    # draw negative table
    neg = np.random.choice(range(n_nodes), size=negative_size, p=weights, replace=True)

    # corrupt random walks
    similarity = _corrupt(walks, neg, n_nodes, r)

    if verbose:
        elapsed = time.time() - start_time
        print(f"Corrupt random walks - T={elapsed:.02}s")

    return walks, similarity
