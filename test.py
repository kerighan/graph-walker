import networkx as nx
import walker


G = nx.random_partition_graph([1000] * 15, .01, .001)
num_nodes = len(G.nodes)
num_edges = len(G.edges)

X = walker.random_walks(G, num_walks=100, walk_len=50)
X, y = walker.corrupt_walks(G, num_walks=100, walk_len=50, p=.1)
print(X.shape)
