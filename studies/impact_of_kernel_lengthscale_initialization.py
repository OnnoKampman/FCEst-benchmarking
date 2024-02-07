import logging
import os
import socket

from fcest.helpers.inference import run_adam_vwp
from fcest.models.wishart_process import VariationalWishartProcess, SparseVariationalWishartProcess
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary, set_trainable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.inference import run_adam_svwp, save_elbo_plot
from helpers.plotters import plot_wishart_process_covariances_pairwise
from helpers.synthetic_covariance_structures import get_covariance_structure, get_ylim
from helpers.synthetic_covariance_structures import get_d2_covariance_structure
from helpers.simulations import simulate_time_series


def check_model_predictions(model, savepath: str = None, title: str = '') -> None:
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(12, 6))
    plt.plot(x, [step[0, 1] for step in cov_structure], linewidth=4, label='ground truth')
    plot_wishart_process_covariances_pairwise(
        x, model
    )
    plt.title(title)
    plt.ylim(get_ylim(cov_structure_type))
    plt.ylabel('TVFC estimates')
    if savepath is not None:
        plt.savefig(os.path.join(savepath))


if __name__ == "__main__":

    # Bivariate only now.
    # We either freeze the initial kernel lengthscales or we let it be trainable.

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    model_name = 'SVWP'
    cfg = get_config_dict(
        data_set_name='d2',
        experiment_data='',
        hostname=hostname
    )

    N = 400
    n_iterations = 15_000
    log_interval = 200
    initial_kernel_lengthscales = [
        0.01,
        0.05,
        0.1,
        0.3,
        1.0,
        10.0
    ]
    figures_savedir = os.path.join('..', 'figures', 'studies', 'impact_of_kernel_lengthscale', f'N{N:04d}')
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)
        os.makedirs(os.path.join(figures_savedir, 'TVFC_estimates'))

    for cov_structure_type in cfg['all-covs-types']:
        cov_structure = get_d2_covariance_structure(
            get_covariance_structure(cov_type=cov_structure_type, n_samples=N)
        )
        y = simulate_time_series(cov_structure)
        x = np.linspace(0, 1, N).reshape(-1, 1)
        n_time_series = y.shape[1]
        nu = n_time_series
        for initial_kernel_lengthscale in initial_kernel_lengthscales:
            for train_kernel_lengthscale in [True, False]:
                figure_name = f"{model_name:s}_{cov_structure_type:s}_init_{initial_kernel_lengthscale:0.2f}_trainable_{str(train_kernel_lengthscale):s}.png"
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
                            kernel=k,
                            train_additive_noise=True
                        )
                    case _:
                        logging.error(f"Model name '{model_name:s}' not recognized.")
                        continue
                set_trainable(m.kernel.lengthscales, train_kernel_lengthscale)
                print_summary(m)
                maxiter = ci_niter(n_iterations)
                match model_name:
                    case 'VWP':
                        logf = run_adam_vwp(m, maxiter, log_interval)
                    case 'SVWP':
                        logf = run_adam_svwp(m, (x, y), maxiter, log_interval)
                save_elbo_plot(
                    maxiter, log_interval, logf,
                    savedir=os.path.join(figures_savedir, 'elbo'),
                    figure_name=figure_name
                )
                print_summary(m)
                final_kernel_lengthscale = float(m.kernel.lengthscales.numpy())
                check_model_predictions(
                    m, savepath=os.path.join(figures_savedir, 'TVFC_estimates', figure_name),
                    title=f"Kernel lengthscales: Init {initial_kernel_lengthscale:0.3f} -> Final {final_kernel_lengthscale:0.3f}"
                )
