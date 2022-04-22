import numpy as np


def laplacian(A, r):
    """
    Root-weighted Laplacian of Koo et al. (2007)
    A is the adjacency matrix and r is the root weight
    """
    L = -A + np.diag(np.sum(A, 0))
    L[0] = r
    return L


class TreeSample:
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

    def _sample_edge(self, j):
        marginals = self._get_marginals(j) # O(n)
        node = np.random.choice(np.arange(self.n), p=marginals)
        return node, marginals[node]

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

    def _get_marginals(self, j):
        marginals = np.zeros(self.n)
        for i in range(self.n):
            if i == j:
                marginals[i] = self.B[0, i] * self.r[i]
            else:
                if j != 0:
                    marginals[i] += self.B[j, j] * self.A[i, j]
                if i != 0:
                    marginals[i] -= self.B[i, j] * self.A[i, j]
        # Correct very small numbers to 0
        marginals[np.isclose(marginals, 0)] = 0
        return marginals

    def weight(self, tree):
        return np.prod(self.W[(tree[1:], np.arange(1, self.n+1))])

    def _sample(self):
        self.L = np.copy(self._L)
        self.B = np.copy(self._B)
        tree = - np.ones(self.n + 1).astype(int)
        p = 1.0
        for j in range(self.n):
            i, p_i = self._sample_edge(j)  # O(n)
            p *= p_i
            if i == j:
                tree[j+1] = 0
            else:
                tree[j+1] = i+1
            self.replace_col(i, j)  # O(n^2)
        return tree, p

    def sample(self):
        while True:
            yield self._sample()


class TreeSwor:
    def __init__(self, W):
        self.W = np.copy(W)
        self.r = W[0, 1:]
        self.A = W[1:, 1:]
        np.fill_diagonal(self.A, 0)
        self.n = len(self.A)
        self._L = None
        self._B = None
        self._Z = None
        self._trees = None
        self._Z_ts = None
        self.L = None
        self.B = None
        self.Z = None
        self.trees = None
        self.Z_ts = None

    def _reset(self):
        self._L = laplacian(self.A, self.r)
        self._B = np.linalg.inv(self._L).transpose()
        self._Z = np.linalg.det(self._L)
        self._trees = []
        self._Z_ts = 0

    def _init_trees(self):
        self.trees = []
        for tree, weight in self._trees:
            self.trees.append((tree, weight))

    def _update_trees(self, i, j):
        trees = []
        if i == j:
            i = 0
        else:
            i += 1
        j += 1
        for tree, weight in self.trees:
            if tree[j] == i:
                trees.append((tree, weight))
            else:
                self.Z_ts -= weight
        self.trees = trees

    def _sample_edge(self, j):
        marginals = self._get_marginals(j) # O(nk)
        node = np.random.choice(np.arange(self.n), p=marginals)
        return node, marginals[node]

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
        # Update L, B and Z
        u = uj - self.L[:, j]
        self.L[:, j] = uj
        bj = self.B[:, j]
        ub = u.T @ bj
        s = 1 + ub
        self.B -= np.outer(bj, u.T @ self.B) / s
        self.Z *= s

    def _get_marginals(self, j):
        marginals = np.zeros(self.n)
        jj = j + 1
        for i in range(self.n):
            if i == j:
                ii = 0
                marginals[i] = self.B[0, i] * self.r[i]
            else:
                ii = i + 1
                if j != 0:
                    marginals[i] += self.B[j, j] * self.A[i, j]
                if i != 0:
                    marginals[i] -= self.B[i, j] * self.A[i, j]
            marginals[i] *= self.Z
            for tree, weight in self.trees:
                if tree[jj] == ii:
                    marginals[i] -= weight
        # Correct very small numbers to 0
        marginals[np.isclose(marginals, 0)] = 0
        # Normalize
        marginals /= self.Z - self.Z_ts
        return marginals

    def weight(self, tree):
        return np.prod(self.W[(tree[1:], np.arange(1, self.n+1))])

    def prob(self, tree):
        return self.weight(tree) / self.Z

    def _sample(self):
        """
        Sample without replacement
        """
        self._init_trees()
        self.L = np.copy(self._L)
        self.B = np.copy(self._B)
        self.Z = self._Z
        self.Z_ts = self._Z_ts
        tree = - np.ones(self.n + 1).astype(int)
        p = 1.0
        for j in range(self.n):
            i, p_i = self._sample_edge(j)  # O(K N)
            p *= p_i
            if i == j:
                tree[j+1] = 0
            else:
                tree[j+1] = i+1
            self.replace_col(i, j)  # O(N^2)
            self._update_trees(i, j)  # O(K)
        weight = self.weight(tree)
        self._trees.append((tree, weight))
        self._Z_ts += weight
        return tree, p

    def sample(self):
        self._reset()
        while not np.allclose(self._Z, self._Z_ts):
            yield self._sample()
