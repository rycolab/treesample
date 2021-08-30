import numpy as np


def all_unrooted_spanning_trees(A):
    n = len(A)

    def enum_dst(cost, included, rest, excluded):
        if len(included) == n:
            return [(rest, cost)]
        dsts = []
        new_excluded = list(excluded)
        for i in included:
            for j in range(n):
                cost_ij = A[i, j]
                if j not in included and (i, j) not in excluded and cost_ij:
                    new_excluded += [(i, j)]
                    dsts += enum_dst(cost * cost_ij, included + [j],
                                     rest + [(i, j, cost_ij)], new_excluded)
        return dsts
    A = np.copy(A)
    np.fill_diagonal(A, 0)
    A[:, 0] = 0.
    dsts = []
    unrooted_dsts = []
    for i in range(n):
        unrooted_dsts += enum_dst(1, [i], [], [])
    for tree, cost in unrooted_dsts:
        t = - np.ones(n)
        for i, j, _ in tree:
            t[j] = i
        dsts.append((t, cost))
    return dsts


def enumerate_directed_spanning_trees(A, root, root_weight):
    n = len(A)

    def enum_dst(cost, included, rest, excluded):
        if len(included) == n:
            return [(rest, cost)]
        dsts = []
        new_excluded = list(excluded)
        for i in included:
            for j in range(n):
                cost_ij = A[i, j]
                if j not in included and (i, j) not in excluded and cost_ij:
                    new_excluded += [(i, j)]
                    dsts += enum_dst(cost * cost_ij, included + [j],
                                     rest + [(i, j, cost_ij)], new_excluded)
        return dsts
    return enum_dst(root_weight, [root], [], [])


def all_rooted_spanning_trees(W):
    W = np.copy(W)
    r = W[0, 1:]
    A = W[1:, 1:]
    n = len(A)
    np.fill_diagonal(A, 0)
    dsts = []
    for root, weight in enumerate(r):
        if weight:
            rooted_dsts = enumerate_directed_spanning_trees(A, root, weight)
            for r_tree, cost in rooted_dsts:
                tree = - np.ones(n + 1)
                tree[root + 1] = 0
                for i, j, _ in r_tree:
                    tree[j + 1] = i + 1
                dsts += [(tree, cost)]
    return dsts