import numpy as np

from unique_randomizer import unique_randomizer as ur

from treesample.util import laplacian


class SWORStruct:
    def __init__(self, W):
        self.W = np.copy(W)
        self.r = self.W[0, 1:]
        self.A = self.W[1:, 1:]
        np.fill_diagonal(self.A, 0)
        self.n = len(self.A)
        self._L = laplacian(self.A, self.r)
        self._B = np.linalg.inv(self._L).transpose()
        self.L = None
        self.B = None

    def reset(self):
        self.L = np.copy(self._L)
        self.B = np.copy(self._B)

    def replace_col(self, i, j):
        # Create column to replace
        uj = np.zeros(self.n)
        if i == j:
            uj[0] = self.r[j]
        else:
            if j != 0:
                uj[j] = self.A[i, j]
            if i != 0:
                uj[i] = -self.A[i, j]
        # Update L and B
        u = uj - self.L[:, j]
        self.L[:, j] = uj
        bj = self.B[:, j]
        ub = u.T @ bj
        s = 1 + ub
        self.B -= np.outer(bj, u.T @ self.B) / s


def _get_marginals(struct, j):
    marginals = np.zeros(struct.n)
    for i in range(struct.n):
        if i == j:
            marginals[i] = struct.B[0, i] * struct.r[i]
        else:
            if j != 0:
                marginals[i] += struct.B[j, j] * struct.A[i, j]
            if i != 0:
                marginals[i] -= struct.B[i, j] * struct.A[i, j]
    # Correct very small numbers to 0
    marginals[marginals < 1e-7] = 0
    # Ensure marginals sum to 1
    marginals /= np.sum(marginals)
    return marginals


def sample(struct, randomizer):
    struct.reset()
    tree = - np.ones(struct.n + 1).astype(int)
    nodes = np.arange(struct.n)
    for j in range(struct.n):
        marginals = _get_marginals(struct, j)  # O(n)
        i = nodes[randomizer.sample_distribution(marginals)]
        if i == j:
            tree[j + 1] = 0
        else:
            tree[j + 1] = i + 1
        struct.replace_col(i, j)  # O(n^2)
    return tree


def tree_swor(W, K=0) -> None:
    struct = SWORStruct(W)
    randomizer = ur.UniqueRandomizer()
    k = 0
    while not randomizer.exhausted() and (not K or k < K):
        yield sample(struct, randomizer)
        k += 1
