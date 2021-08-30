import numpy as np


class Sample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.n = len(W)
        self.W[:, 0] = 0
        np.fill_diagonal(self.W, 0)
        for i in range(1, self.n):
            self.W[:, i] /= np.sum(self.W[:, i])

    def _desc(self, tree, i):
        nodes = set()
        stack = [i]
        while len(stack):
            x = stack.pop(0)
            nodes.add(x)
            for j, u in enumerate(tree):
                if u == x:
                    stack.append(j)
        return nodes

    def sample(self):
        W = np.copy(self.W)
        tree = - np.ones(self.n).astype(int)
        for j in range(1, self.n):
            nodes = self._desc(tree, j)
            for u in nodes:
                W[u, j] = 0
            W[:, j] /= np.sum(W[:, j])
            i = np.random.choice(np.arange(self.n), p=W[:, j])
            tree[j] = i
        return tree


class WilsonSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.n = len(W)
        self.W[:, 0] = 0
        np.fill_diagonal(self.W, 0)
        for i in range(1, self.n):
            self.W[:, i] /= np.sum(self.W[:, i])

    def sample(self):
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


class WilsonRCSample:
    def __init__(self, W):
        self.W = np.copy(W)
        self.r = self.W[0, 1:]
        self.A = self.W[1:, 1:]
        np.fill_diagonal(self.A, 0)
        self.n = len(self.A)
        self.r /= np.sum(self.r)
        for i in range(self.n):
            self.A[:, i] /= np.sum(self.A[:, i])

    def sample(self):
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
