"""
Basic test suite, all test examples taken from README.md
"""

import networkx as nx
import walker
import pytest
import numpy as np

@pytest.fixture(scope="module")
def random_partition_graph()->nx.Graph:
    return nx.random_partition_graph([1000] * 15, .01, .001)

def assert_walk_shape(walks:np.ndarray, n_walks:int, walk_len:int, n_nodes: int):
    assert walks.shape == (n_walks * n_nodes, walk_len)

def test_random_walks(random_partition_graph: nx.Graph):
    n_walks = 50
    walk_len = 25
    
    walks = walker.random_walks(random_partition_graph, n_walks=n_walks, walk_len=walk_len)
    assert_walk_shape(walks, n_walks, walk_len, len(random_partition_graph.nodes))
    
    
def test_generate_random_walks_with_restart_probability(random_partition_graph: nx.Graph):
    n_walks = 50
    walk_len = 25
    alpha = .1
    
    walks = walker.random_walks(random_partition_graph, n_walks=n_walks, walk_len=walk_len, alpha=alpha)
    assert_walk_shape(walks, n_walks, walk_len, len(random_partition_graph.nodes))


def test_from_starting_nodes(random_partition_graph: nx.Graph):
    n_walks = 50
    walk_len = 25
    start_nodes = [0, 1, 2]
    
    walks = walker.random_walks(random_partition_graph, n_walks=n_walks, walk_len=walk_len, start_nodes=start_nodes)
    assert_walk_shape(walks, n_walks, walk_len, len(start_nodes))


def test_p_and_q(random_partition_graph: nx.Graph):
    n_walks = 50
    walk_len = 25
    p = .25
    q = .25
    
    walks = walker.random_walks(random_partition_graph, n_walks=n_walks, walk_len=walk_len, p=p, q=q)
    assert_walk_shape(walks, n_walks, walk_len, len(random_partition_graph.nodes))


def test_corrupt_walks(random_partition_graph: nx.Graph):
    n_walks = 50
    walk_len = 25
    p = .25
    q = .25
    r = .1
    
    walks = walker.random_walks(random_partition_graph, n_walks=n_walks, walk_len=walk_len, p=p, q=q)
    assert_walk_shape(walks, n_walks, walk_len, len(random_partition_graph.nodes))
    corrupted_walks = walker.corrupt(random_partition_graph, walks, r=r)
    assert_walk_shape(corrupted_walks, n_walks, walk_len-1, len(random_partition_graph.nodes))
