# How to install

```
pip install pybind11
pip install graph-walker
```

# How to use

```python
import networkx as nx
import walker

# create a random graph
G = nx.random_partition_graph([1000] * 15, .01, .001)

# generate random walks
X = walker.random_walks(G, n_walks=50, walk_len=25)

# generate random walks with restart probability alpha
X = walker.random_walks(G, n_walks=50, walk_len=25, alpha=.1)

# you can generate random walks from specified starting nodes
X = walker.random_walks(G, n_walks=50, walk_len=25, start_nodes=[0, 1, 2])

# generate random walks according to Node2Vec methodology by specifying p and q
X = walker.random_walks(G, n_walks=50, walk_len=25, p=.25, q=.25)

# corrupt random walks by randomly changing nodes in random walks
# `y` matrix has a size (N, walk_len - 1) with:
# y[i, j] = 1 if nodes X[i, j] and X[i, j + 1] share an edge, 0 otherwise
y = walker.corrupt(G, X, r=.1)
```
