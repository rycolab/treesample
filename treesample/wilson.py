import numpy as np
from tqdm import tqdm

from treesample.util import is_single_root, laplacian
from treesample.util import tree_weight


class WilsonSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.n = len(W)
        self.W[:, 0] = 0
        np.fill_diagonal(self.W, 0)
        for i in range(1, self.n):
            self.W[:, i] /= np.sum(self.W[:, i])

    def _sample(self):
        inTree = np.zeros(self.n).astype(bool)
        next = - np.ones(self.n).astype(int)
        inTree[0] = True
        for i in range(1, self.n):
            u = i
            while not inTree[u]:
                next[u] = np.random.choice(np.arange(self.n), p=self.W[:, u])
                u = next[u]
            u = i
            while not inTree[u]:
                inTree[u] = True
                u = next[u]
        return next

    def sample(self):
        while True:
            yield self._sample()


class WilsonRCRejectSample(WilsonSample):
    def sample(self):
        for tree in super(WilsonRCRejectSample, self).sample():
            if is_single_root(tree):
                yield tree


class WilsonRCMarginalSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.A = self.W[1:, 1:]

        np.fill_diagonal(self.A, 0)
        B = np.linalg.inv(laplacian(self.A, self.W[0, 1:])).transpose()
        self.r = self.W[0, 1:] * B[0]
        self.r[np.abs(self.r) < 1e-7] = 0
        self.r /= np.sum(self.r)
        if np.any(self.r < 0):
            import ipdb; ipdb.set_trace()

        self.n = len(self.A)
        for i in range(self.n):
            self.A[:, i] /= np.sum(self.A[:, i])

    def _sample(self):
        inTree = np.zeros(self.n + 1).astype(bool)
        next = - np.ones(self.n + 1).astype(int)
        inTree[0] = True
        # Pick root edge
        j = np.random.choice(np.arange(self.n), p=self.r)
        next[j+1] = 0
        inTree[j+1] = True
        for i in range(self.n):
            u = i + 1
            while not inTree[u]:
                next[u] = np.random.choice(np.arange(self.n), p=self.A[:, u-1]) + 1
                u = next[u]
            u = i + 1
            while not inTree[u]:
                inTree[u] = True
                u = next[u]
        return next

    def sample(self):
        while True:
            yield self._sample()


class WilsonRCBiasedSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.r = self.W[0, 1:]
        self.A = self.W[1:, 1:]
        np.fill_diagonal(self.A, 0)
        self.n = len(self.A)
        self.r /= np.sum(self.r)
        for i in range(self.n):
            self.A[:, i] /= np.sum(self.A[:, i])

    def _sample(self):
        inTree = np.zeros(self.n + 1).astype(bool)
        next = - np.ones(self.n + 1).astype(int)
        inTree[0] = True
        # Pick root edge
        j = np.random.choice(np.arange(self.n), p=self.r)
        next[j+1] = 0
        inTree[j+1] = True
        for i in range(self.n):
            u = i + 1
            while not inTree[u]:
                next[u] = np.random.choice(np.arange(self.n), p=self.A[:, u-1]) + 1
                u = next[u]
            u = i + 1
            while not inTree[u]:
                inTree[u] = True
                u = next[u]
        return next

    def sample(self):
        while True:
            yield self._sample()


def test_wilson(N, K=10000):
    W = np.random.uniform(0, 1, size=(N, N))
    np.fill_diagonal(W, 0)
    wilson = WilsonSample(W)
    dist = {}
    sampler = wilson.sample()
    for _ in range(K):
        tree = tuple(next(sampler))
        if tree not in dist:
            dist[tree] = 0
        dist[tree] += 1 / K
    L = - np.copy(W)
    np.fill_diagonal(L, np.sum(W, axis=0))
    Z = np.linalg.det(L[1:, 1:])
    approx = 0
    total = len(dist)
    for tree in dist:
        prob = dist[tree]
        real_prob = tree_weight(np.array(tree, dtype=int), W) / Z
        approx += np.isclose(prob, real_prob, rtol=0.1)
    return approx, total


def test_wilson_rc_slow(N, K=10000):
    W = np.random.uniform(0, 1, size=(N, N))
    np.fill_diagonal(W, 0)
    wilson = WilsonRCMarginalSample(W)
    dist = {}
    sampler = wilson.sample()
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


def test_wilson_rc(N, K=10000):
    W = np.random.uniform(0, 1, size=(N, N))
    np.fill_diagonal(W, 0)
    wilson = WilsonRCRejectSample(W)
    dist = {}
    sampler = wilson.sample()
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
    approx_rc_slow = total_rc_slow = 0
    approx_rc = total_rc = 0
    for seed in tqdm(range(100)):
        np.random.seed(seed)
        a, t = test_wilson(N)
        approx += a
        total += t
        a, t = test_wilson_rc_slow(N)
        approx_rc_slow += a
        total_rc_slow += t
        a, t = test_wilson_rc(N)
        approx_rc += a
        total_rc += t
    print(f"Wilson sampled {approx} out of {total} trees ({approx / total * 100:.2f}%) with correct probabilities")
    print(f"WilsonSlowRc sampled {approx_rc_slow} out of {total_rc_slow} trees ({approx_rc_slow / total_rc_slow * 100:.2f}%) with correct probabilities")
    print(f"WilsonRC sampled {approx_rc} out of {total_rc} trees ({approx_rc / total_rc * 100:.2f}%) with correct probabilities")


if __name__ == '__main__':
    tests()