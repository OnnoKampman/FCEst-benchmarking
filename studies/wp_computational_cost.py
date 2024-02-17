import logging
import os
import socket
import time

from fcest.helpers.inference import run_adam_vwp, run_adam_svwp
from fcest.models.wishart_process import VariationalWishartProcess, SparseVariationalWishartProcess
import gpflow
from gpflow.ci_utils import ci_niter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import set_size
from helpers.synthetic_covariance_structures import get_ground_truth_covariance_structure
from helpers.simulations import simulate_time_series


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}


def _plot_compute_times_heatmap(df, savepath: str = None) -> None:
    """
    Save compute times heatmap.

    :param df:
    :param savepath:
    :return:
    """
    plt.rcParams.update(tex_fonts)

    # Initialize figure instance.
    fig, ax = plt.subplots(1, 1, figsize=set_size(fraction=0.47))

    print(df)
    sns.heatmap(
        df,
        # vmin=0,
        # vmax=400,
        cbar_kws={'label': "compute time [s]"},
        norm=LogNorm(),
        ax=ax
    )
    ax.invert_yaxis()
    plt.yticks(rotation=0)
    ax.set_xlabel('N')
    ax.set_ylabel('D')

    if savepath is not None:
        fig.savefig(
            os.path.join(savepath),
            format='pdf',
            bbox_inches='tight'
        )
        print(f"Figure saved in '{savepath:s}'.")


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    cov_structure_type = 'null'
    cfg = get_config_dict(
        data_set_name='sim',
        experiment_data='',
        hostname=hostname
    )

    model_name = 'VWP'
    max_dim = 20
    max_n_time_steps = 1400
    step_size_n_time_steps = 100

    dim_range = range(2, max_dim + 1)
    n_time_steps_range = range(step_size_n_time_steps, max_n_time_steps+step_size_n_time_steps, step_size_n_time_steps)

    n_iterations = 4
    log_interval = 2

    figures_savedir = os.path.join(
        # '..',
        'figures', 'studies', 'wp_computational_cost'
    )
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    # Train model and save compute times for each configuration.
    compute_times_df = pd.DataFrame(
        np.arange(len(dim_range) * len(n_time_steps_range)).reshape(len(dim_range), len(n_time_steps_range)),
        index=dim_range, columns=n_time_steps_range
    )
    for D in dim_range:
        for N in n_time_steps_range:
            cov_structure = get_ground_truth_covariance_structure(
                covs_type=cov_structure_type,
                data_set_name=f'd{D:d}s',  # train on sparse covariance structure
                n_samples=N,
                signal_to_noise_ratio=None
            )
            y = simulate_time_series(cov_structure)
            x = np.linspace(0, 1, N).reshape(-1, 1)
            n_time_series = y.shape[1]
            nu = n_time_series
            k = gpflow.kernels.Matern52()
            match model_name:
                case 'VWP':
                    m = VariationalWishartProcess(
                        x, y,
                        nu=n_time_series,
                        kernel=k
                    )
                case 'SVWP':
                    m = SparseVariationalWishartProcess(
                        D=n_time_series,
                        Z=x[:cfg['n-inducing-points']],
                        nu=n_time_series,
                        kernel=k
                    )
                case _:
                    logging.error(f"Model name '{model_name:s}' not recognized.")
                    continue
            maxiter = ci_niter(n_iterations)
            train_start_time = time.time()
            match model_name:
                case 'VWP':
                    _ = run_adam_vwp(
                        model=m,
                        iterations=maxiter,
                        log_interval=log_interval
                    )
                case 'SVWP':
                    _ = run_adam_svwp(
                        model=m,
                        data=(x, y),
                        iterations=maxiter,
                        log_interval=log_interval
                    )
            run_time = time.time() - train_start_time
            print("My program took", run_time, " seconds to run")
            compute_times_df.loc[D, N] = run_time

    # TODO: The first training routine somehow takes a lot longer.
    compute_times_df.loc[dim_range[0], n_time_steps_range[0]] = compute_times_df.loc[dim_range[0], n_time_steps_range[1]]

    _plot_compute_times_heatmap(
        df=compute_times_df,
        savepath=os.path.join(figures_savedir, f"{model_name:s}.pdf"),
    )
