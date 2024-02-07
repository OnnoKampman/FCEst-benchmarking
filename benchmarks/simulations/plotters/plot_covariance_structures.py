import logging
import os
import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.synthetic_covariance_structures import get_covariance_time_series


def _plot_synthetic_covariance_structures(
        config_dict: dict, n_time_steps: int = 400, figures_save_basedir: str = None
) -> None:
    """
    Saves overview of all synthetic covariance structures used in the simulations benchmark.

    :param config_dict:
    :param n_time_steps:
    :param figures_save_basedir:
    """
    sns.set(style="white", font_scale=1.4)
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))
    matplotlib.rc('ytick', labelsize=14)

    ylim = [-1.1, 1.1]
    xx = np.linspace(0, 1, n_time_steps).reshape(-1, 1).astype(np.float64)

    fig, axes = plt.subplots(
        nrows=2, ncols=4,
        sharex=True, sharey=True,
        figsize=config_dict['figure-covariance-structures-figsize']
    )
    for i_covs_type, covs_type in enumerate(config_dict['plot-covs-types']):
        ground_truth_covariance_structure = get_covariance_time_series(
            covs_type,
            n_samples=n_time_steps,
            signal_to_noise_ratio=None
        )  # (N, )

        i_row, i_col = np.unravel_index(i_covs_type, (2, 4))
        axes[i_row, i_col].plot(
            xx, ground_truth_covariance_structure,
            linewidth=3.8
        )
        if not (i_covs_type == 0 or i_covs_type == 4):
            axes[i_row, i_col].axis('off')
        else:
            axes[i_row, i_col].get_xaxis().set_visible(False)
            axes[i_row, i_col].spines['right'].set_visible(False)
            axes[i_row, i_col].spines['top'].set_visible(False)
            axes[i_row, i_col].spines['bottom'].set_visible(False)

        covs_type = _convert_title(covs_type)
        axes[i_row, i_col].set_title(covs_type)
        axes[i_row, i_col].set_ylim(ylim)
    axes[np.unravel_index(7, (2, 4))].axis('off')  # turn off empty subplot
    fig.text(
        0.04, 0.5,
        'covariance / correlation',  # 'synthetic TVFC\n(covariance / correlation)'
        va='center', rotation='vertical', fontsize=20
    )
    plt.subplots_adjust(hspace=0.3, wspace=0.0)
    if figures_save_basedir is not None:
        figure_name = 'covariance_structures.pdf'
        plt.savefig(
            os.path.join(figures_save_basedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_save_basedir:s}'.")
        plt.close()


def _convert_title(covariance_structure_type: str) -> str:
    if covariance_structure_type == 'periodic_1':
        covariance_structure_type = 'periodic (slow)'
    if covariance_structure_type == 'periodic_3':
        covariance_structure_type = 'periodic (fast)'
    if covariance_structure_type == 'state_transition':
        covariance_structure_type = 'state transition'
    if covariance_structure_type == 'checkerboard':
        covariance_structure_type = 'boxcar'
    return covariance_structure_type


if __name__ == "__main__":

    cfg = get_config_dict(
        data_set_name='sim',  # not used here
        experiment_data='',  # not used here
        hostname=socket.gethostname()
    )

    _plot_synthetic_covariance_structures(
        config_dict=cfg,
        figures_save_basedir=os.path.join(
            cfg['project-basedir'], 'opk20_hivemind_paper_1', 'figures', 'simulations'
        )
    )
