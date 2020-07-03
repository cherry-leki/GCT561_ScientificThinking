import matplotlib.pyplot as plt
import networkx as nx

G = nx.gnm_random_graph(4, 3)

print(G.degree)
plt.hist([G.degree(n) for n in G.nodes()])
plt.show()