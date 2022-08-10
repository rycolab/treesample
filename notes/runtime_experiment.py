import numpy as np
import pylab as pl
from tqdm import tqdm

from arsenal.timer import timers
from treesample.colbourn import ColbournSample
from treesample.wilson import WilsonRCRejectSample, WilsonRCMarginalSample
from treesample.swor import TreeSwor
from treesample.swor_ur import tree_swor


def experiment():
    T = timers()
    for N in tqdm(range(5, 101, 5)):
        for _ in range(100):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, 0)
            colbourn = ColbournSample(graph)
            wilson = WilsonRCRejectSample(graph)
            wilson_slow = WilsonRCMarginalSample(graph)
            with T["colbourn"](n=N):
                for i, sample in enumerate(colbourn.sample()):
                    if i == 19:
                        break
            with T["wilson_rc"](n=N):
                for i, sample in enumerate(wilson.sample()):
                    if i == 19:
                        break
            with T["wilson_slow"](n=N):
                for i, sample in enumerate(wilson_slow.sample()):
                    if i == 19:
                        break
    T.compare()
    T.plot_feature("n")
    pl.savefig("notes/runtime.png")


def experiment_swor(N):
    T = timers()
    for K in tqdm(range(1, 3001, 100)):
        for _ in range(3):
            graph = np.random.uniform(0, 1, size=(N, N))
            np.fill_diagonal(graph, 0)
            swor = TreeSwor(graph).sample()
            with T["trie"](k=K):
                tree_swor(np.copy(graph), K)
            with T["swor"](k=K):
                for _ in range(K):
                    next(swor)
    T.compare()
    T.plot_feature("k")
    pl.savefig(f"notes/runtime_swor_n={N}.png")


if __name__ == '__main__':
    experiment()
    experiment_swor(10)
