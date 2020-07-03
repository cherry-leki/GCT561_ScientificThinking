import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Q3
m_3b = np.matrix([[0, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 0, 1],
                  [0, 0, 1, 0]])

w, v = LA.eig(m_3b)

# 3-b
print('3-(b) eigenvalues')
print(w)
print()

# 3-c
print('3-(c) eigenvectors')
print(v[:, 0])
print()

# 3-d
G = nx.from_numpy_matrix(m_3b, create_using=nx.Graph)
G_centrality = nx.eigenvector_centrality_numpy(G.copy())
print("3-(d) eigenvector_centrality")
print(['%s %f'%(node,G_centrality[node]) for node in G_centrality])
print()
print("#################################")



# Q4
m_4a = np.matrix([[0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
DG = nx.from_numpy_matrix(m_4a, create_using=nx.DiGraph)
print('4-(d) katz_centrality')
for i in [0, 0.5, 0.85, 1, 2]:
    katz_centrality = nx.katz_centrality_numpy(DG, alpha=i, beta=1, normalized=False)
    print('alpha: ' + str(i))
    print(['%s %f' % (node, katz_centrality[node]) for node in katz_centrality])