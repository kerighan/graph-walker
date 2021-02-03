import networkx as nx
import numpy as np
import walker


G = nx.karate_club_graph()
walks, similarity = walker.corrupted_random_walks(
    G, n_walks=100, walk_len=50,
    p=.25, q=.25, sub_sampling=0.5)
