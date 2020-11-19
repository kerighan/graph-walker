import networkx as nx
from numba import jit, prange
import numpy as np
import time


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _random_walks(
    indptr,
    data,
    indices,
    num_nodes,
    num_walks,
    walk_len
):
    walks = np.zeros((num_nodes * num_walks, walk_len), dtype=np.uint16)
    for i in prange(num_nodes * num_walks):
        step = i % num_nodes
        walks[i, 0] = step
        for k in range(1, walk_len):
            # get data indices
            start = indptr[step]
            end = indptr[step + 1]

            # draw random float
            draw = np.random.rand()

            # get next index
            index = start + np.searchsorted(data[start:end], draw)
            step = indices[index]

            # update walk at current timestep
            walks[i, k] = step
    return walks


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _corrupt_walks(walks, adj_indptr, adj_indices, num_nodes, p=.1):
    num_corruptions = int(walks.size * p)
    walk_len = walks.shape[1]
    num_walks = walks.shape[0]
    similarity = np.ones(walks.shape, dtype=np.uint8)
    for i in prange(num_corruptions):
        # draw random position on matrix
        x = np.random.randint(0, num_walks)
        y = np.random.randint(1, walk_len)

        # change step by a random node
        random_node = np.random.randint(0, num_nodes)
        walks[x, y] = random_node

        # change similarity value based on adjacency matrix
        start = adj_indptr[random_node]
        end = adj_indptr[random_node + 1]
        neighbors = adj_indices[start:end]

        # modify connectivity around corruption
        previous_node = walks[x, y - 1]
        if y > walk_len - 2:
            previous_value = 0
            for neighbor in neighbors:
                if neighbor == previous_node:
                    value = 1
                    break
            similarity[x, y - 1] = previous_value
        else:
            next_node = walks[x, y + 1]
            previous_value = 0
            next_value = 0
            found = 0
            for neighbor in neighbors:
                if neighbor == previous_node:
                    previous_value = 1
                    found += 1
                elif neighbor == next_node:
                    next_node = 1
                    found += 1
                if found == 2:
                    break
            similarity[x, y - 1] = previous_value
            similarity[x, y] = next_value

    return walks, similarity


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def cumsum_probability_matrix(indptr, data, indices, num_nodes):
    for i in prange(num_nodes):
        start = indptr[i]
        end = indptr[i + 1]
        data[start:end] = np.cumsum(data[start:end])
    return indptr, data, indices
