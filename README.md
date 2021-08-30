# Tree Sampling Algorithms
This library contains implementations for two sampling algorithms discussed in
["Efficient Sampling of Dependency Structure"]().
The original algorithms are due to Wilson (1996) and Colbourn et al. (1996).
While not originally present in the algorithms, this library contains extensions that allows sampling trees
with a root constraint.
While Wilson's algorithm has a faster runtime (O(H) where H is the mean hitting time) than
Colbourn's algorithm (O(N^3) where N is the number of nodes in the tree), Cobourn's algorithm
is amenable to sampling without replacement. See the above paper for explanations of all algorithms.

## Citation

This code is for the papers _Please Mind the Root: Decoding Arborescences for Dependency Parsing_ and
_On Finding the K-best Non-projective Dependency Trees_ featured in EMNLP 2020 and ACL 2021 respectively.
Please cite as:

```bibtex
@inproceedings{zmigrod-etal-2021-efficient,
    title = "Efficient Sampling of Dependency Structure",
    author = "Zmigrod, Ran  and
      Vieira, Tim  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

## Requirements and Installation

* Python version >= 3.6

Installation:
```bash
git clone https://github.com/rycolab/treesample
cd treesample
pip install -e .
```

## Related Work
This code repository focuses on sampling MSTs.
A useful library to use during training and learning of edges weights
can be found [here](https://github.com/rycolab/tree_expectations).

A useful library for exact decoding can be found [here](https://github.com/rycolab/spanningtrees).

Other libraries for performing MST computations are [networkx](https://networkx.github.io/documentation/stable/index.html)
and [stanza](https://stanfordnlp.github.io/stanza/).
