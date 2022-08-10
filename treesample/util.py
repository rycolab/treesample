import numpy as np


def laplacian(A, r):
    """
    Root-weighted Laplacian of Koo et al. (2007)
    A is the adjacency matrix and r is the root weight
    """
    L = -A + np.diag(np.sum(A, 0))
    L[0] = r
    return L


def is_single_root(tree):
    return np.count_nonzero(tree == 0) == 1


def tree_weight(tree, graph):
    return np.prod(graph[tree[1:], np.arange(1, len(tree))])
