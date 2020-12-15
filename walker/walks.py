from numba import jit, prange
from .utils import isin
import numpy as np
import time


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _random_walks(
    indptr,
    data,
    indices,
    n_nodes,
    n_walks,
    walk_len
):
    walks = np.zeros((n_nodes * n_walks, walk_len), dtype=np.uint32)
    for i in prange(n_nodes * n_walks):
        step = i % n_nodes
        walks[i, 0] = step
        for k in range(1, walk_len):
            # get data indices
            start = indptr[step]
            end = indptr[step + 1]

            # if no neighbors, we fill in current node
            if start == end:
                walks[i, k] = step
                continue

            # draw random float
            draw = np.random.rand()

            # get next index
            index = start + np.searchsorted(data[start:end], draw)
            step = indices[index]

            # update walk at current timestep
            walks[i, k] = step
    return walks


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _node2vec_random_walks(
    indptr,
    data,
    indices,
    n_nodes,
    n_walks,
    walk_len,
    p=.5, q=.5
):
    walks = np.zeros((n_nodes * n_walks, walk_len), dtype=np.uint32)
    for i in prange(n_nodes * n_walks):
        step = i % n_nodes
        walks[i, 0] = step
        for k in range(1, walk_len):
            # get data indices
            start = indptr[step]
            end = indptr[step + 1]

            # if no neighbors, we fill in current node
            if start == end:
                walks[i, k] = step
                continue

            # draw random float
            draw = np.random.rand()

            if k >= 2:
                # previous node
                prev = walks[i, k - 2]
                prev_start = indptr[prev]
                prev_end = indptr[prev + 1]
                prev_neighbors = indices[prev_start:prev_end]

                # step is the current node
                step_neighbors = indices[start:end]
                step_data = data[start:end]

                cumsum = 0.
                weights = np.zeros((end - start,), dtype=np.float32)
                for n in range(end - start):
                    neighbor = step_neighbors[n]
                    w = step_data[n]
                    if neighbor == prev:
                        w /= p
                    elif isin(neighbor, prev_neighbors):
                        pass
                    else:
                        w /= q
                    weights[n] = w
                    cumsum += w
                weights /= cumsum
                weights = weights.cumsum()

                # get next index
                index = np.searchsorted(weights, draw)
                step = step_neighbors[index]

                # update walk at current timestep
                walks[i, k] = step
            else:
                # get next index
                index = start + np.searchsorted(data[start:end], draw)
                step = indices[index]

                # update walk at current timestep
                walks[i, k] = step
    return walks


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _corrupt_walks(walks, adj_indptr, adj_indices, n_nodes, p=.1):
    n_corruptions = int(walks.size * p)
    walk_len = walks.shape[1]
    n_walks = walks.shape[0]
    similarity = np.ones((n_walks, walk_len - 1), dtype=np.uint8)
    for i in prange(n_corruptions):
        # draw random position on matrix
        x = np.random.randint(0, n_walks)
        y = np.random.randint(1, walk_len)

        # change step by a random node
        random_node = np.random.randint(0, n_nodes)
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
def _weighted_corrupt_walks(walks, adj_indptr, adj_indices, n_nodes, weights, p=.1):
    n_corruptions = int(walks.size * p)
    walk_len = walks.shape[1]
    n_walks = walks.shape[0]
    similarity = np.ones((n_walks, walk_len - 1), dtype=np.uint8)
    for i in prange(n_corruptions):
        # draw random position on matrix
        x = np.random.randint(0, n_walks)
        y = np.random.randint(1, walk_len)

        # draw random float
        draw = np.random.rand()
        # change step by a random node
        random_node = np.searchsorted(weights, draw)
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
