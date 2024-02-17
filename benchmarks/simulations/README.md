# Simulations benchmark

## Train WP models on cluster

```shell
$ python benchmarks/simulations/train_models/generate_slurm_jobs_train_WP.py <data_set_name> <experiment_name> <model_name>
$ python benchmarks/simulations/train/models/generate_slurm_jobs_train_WP.py d2 N0200_T0001 SVWP
```

## Compute quantitative results

Locally and on the cluster:

```shell
$ python benchmarks/simulations/compute_all_quantitative_results.py d2 N0200_T0001
$ python benchmarks/simulations/compute_all_quantitative_results.py d2 N0200_T0001 /mnt/Data/neuro-dynamic-covariance/experiments experiments
```

```shell
python benchmarks/simulations/generate_slurm_jobs_compute_all_quantitative_results.py d2 N0200_T0001
python benchmarks/simulations/compute_average_quantitative_results.py d2 N0200_T0001
```

## Generate plots

```shell
python benchmarks/simulations/plot_TVFC_estimates.py d2 N0200_T0001
```
