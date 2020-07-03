import networkx as nx
import numpy as np
from numpy import linalg as LA

def CALC_katz_by_lin_eq(G, alpha):
    """
    linear eq: (I - alpha*A) = b
    를 풀어서, katz centrality를 계산하는 방법.
    """
    A = nx.adj_matrix(G).T
    n = A.shape[0]
    b = np.ones((n, 1)) * float(1)
    # linear equation solve
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * A), b)
    centrality = {n: centrality[i][0] for i, n in enumerate(G)}
    return centrality


# Q4
m_4a = np.matrix([[0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])

_, v = LA.eig(m_4a)

# 4-(b)
print("4-(b),(c) eigenvector_centrality")
print(v[:, 0])
print()

# another method
DG = nx.from_numpy_matrix(m_4a, create_using=nx.DiGraph)
DG_centrality = nx.eigenvector_centrality_numpy(DG.copy())
print(['%s %f'%(node,DG_centrality[node]) for node in DG_centrality])
print()


# 4-(c)
# nx.draw(DG)
# plt.show()
print('4-(d) katz_centrality')
for i in [0, 0.5, 0.85, 1, 2]:
    katz_centrality = nx.katz_centrality_numpy(DG, alpha=i, beta=1, normalized=False)
    print('alpha: ' + str(i))
    print(['%s %f' % (node, katz_centrality[node]) for node in katz_centrality])

# another method
# for i in [0, 0.5, 0.85, 1, 2]:
#     centrality = CALC_katz_by_lin_eq(DG.copy(), alpha=i)
#     print("alpha: " + str(i))
#     print(centrality)