# Simulation benchmark data

Our synthetic data sets need to be imported and processed with `R` for the `MGARCH` models.
Therefore, we generate the data with the respective `generate_dataset.py` scripts and save them as `.csv` files.

You can generate a bivariate data set of `N` time steps and `T` trials by running

```shell
python datasets/fmri/sim/generate_dataset.py <data_set_name> <N> <T>
```

For example:

```shell
python datasets/fmri/sim/generate_dataset.py d2 400 200
```

## Data dimensionality

At the moment we support 'd2', 'd3d', 'd3s', 'd4s', 'd6s', 'd9s', and 'd15' simulations data sets.
