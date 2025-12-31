# FCEst-benchmarking

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Functional connectivity (FC) estimates are sensitive to choice of estimation method (e.g., sliding windows, MGARCH, Wishart processes, hidden markov models).
This project aims to design a range of benchmarks to determine what estimation method to use.
At the moment, this only includes fMRI benchmarks, but in the future we may add EEG and MEG benchmarks.

These benchmarks have been particularly developed to test the estimation methods included in the [FCEst](https://github.com/OnnoKampman/FCEst) Python package.
Results have been published in an article in Imaging Neuroscience (see `CITATION.cff`).

Many extensions of this project are possible, both in terms of adding more estimation methods and more benchmarks.

## Getting Started

The easiest way to set up your local Python environment is to use Anaconda.

```zsh
$ conda env create -f environment.yml
$ conda activate fcest-env
```

Importantly, this installs the `FCEst` package.

## Contributing

This is an open-source project and contributions are more than welcome.
Please raise an issue here on Github or send me a message.

## References

A curated list of relevant publications related to FC estimation benchmarking can be found on [Semantic Scholar](https://www.semanticscholar.org/shared/library/folder/101603).
