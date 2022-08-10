import numpy as np
from tqdm import tqdm

from treesample.util import laplacian, tree_weight


class ColbournSample:
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
        marginals[marginals < 1e-7] = 0
        # Ensure marginals sum to 1
        marginals /= np.sum(marginals)
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
            yield self._sample()[0]

def test_colbourn(N, K=10000):
    W = np.random.uniform(0, 1, size=(N, N))
    np.fill_diagonal(W, 0)
    colbourn = ColbournSample(W)
    dist = {}
    sampler = colbourn.sample()
    for _ in range(K):
        tree = tuple(next(sampler))
        if tree not in dist:
            dist[tree] = 0
        dist[tree] += 1 / K
    A = np.copy(W)
    L = laplacian(A[1:, 1:], A[0, 1:])
    Z = np.linalg.det(L)
    approx = 0
    total = len(dist)
    for tree in dist:
        prob = dist[tree]
        real_prob = tree_weight(np.array(tree, dtype=int), W) / Z
        approx += np.isclose(prob, real_prob, rtol=0.1)
    return approx, total


def tests():
    N = 4
    approx = total = 0
    for seed in tqdm(range(10)):
        np.random.seed(seed)
        a, t = test_colbourn(N)
        approx += a
        total += t
    print(f"Colbourn sampled {approx} out of {total} trees ({approx / total * 100:.2f}%) with correct probabilities")


if __name__ == '__main__':
    tests()
