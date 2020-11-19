from .utils import _random_walks, _corrupt_walks
from .preprocess import _preprocess_graph
import networkx as nx
import time


def random_walks(G, num_walks=10, walk_len=10, verbose=True):
    start_time = time.time()

    num_nodes = len(G.nodes)
    indptr, data, indices = _preprocess_graph(G, num_nodes)
    walks = _random_walks(
        indptr, data, indices, num_nodes,
        num_walks, walk_len)
    
    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")
    return walks


def corrupt_walks(G, num_walks=10, walk_len=10, p=.1, verbose=True):
    start_time = time.time()

    A = nx.adjacency_matrix(G)
    adj_indptr = A.indptr
    adj_indices = A.indices

    num_nodes = len(G.nodes)
    indptr, data, indices = _preprocess_graph(G, num_nodes)

    walks = _random_walks(
        indptr, data, indices, num_nodes,
        num_walks=num_walks, walk_len=walk_len)
    walks, similarity = _corrupt_walks(
        walks, adj_indptr, adj_indices, num_nodes, p=p)

    if verbose:
        duration = time.time() - start_time
        print(f"Corrupted random walks - T={duration:.2f}s")
    return walks, similarity
