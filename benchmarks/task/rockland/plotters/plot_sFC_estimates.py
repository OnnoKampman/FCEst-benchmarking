import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.rockland import get_rockland_subjects, load_rockland_data


def _plot_static_functional_connectivity_estimate(
        config_dict: dict, correlation_matrix: np.array, brain_regions_of_interest: list[str],
        subject_filename: str = None, mean_estimate: bool = False, figures_savedir: str = None
) -> None:
    """
    Plots average TVFC estimates over all subjects for a single TVFC estimation method.

    :param config_dict:
    """
    # sns.set_palette("colorblind")
    sns.set(style="whitegrid", font_scale=1.0)
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    if mean_estimate:
        figure_name_correlation_matrix = "mean_sFC_correlation_matrix.pdf"
    else:
        figure_name_correlation_matrix = f"{subject_filename:s}_sFC_correlation_matrix.pdf"

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )
    sns.heatmap(
        correlation_matrix,
        cmap='jet',  # or 'viridis'
        xticklabels=brain_regions_of_interest,
        yticklabels=brain_regions_of_interest,
        square=True,
        vmin=0.0,
        vmax=1.0,
        ax=ax
    )
    plt.yticks(rotation=0)
    plt.tight_layout()

    if figures_savedir is not None:

        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)

        plt.savefig(
            os.path.join(
                figures_savedir, figure_name_correlation_matrix
            ),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name_correlation_matrix:s}' in '{figures_savedir:s}'.")
        plt.close()


if __name__ == "__main__":

    data_split = 'all'
    metric = 'correlation'
    pp_pipeline = 'custom_fsl_pipeline'

    repetition_time = sys.argv[1]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    brain_regions_of_interest = cfg['roi-list']
    edges_to_plot_indices = cfg['roi-edges-list']
    n_time_series = len(brain_regions_of_interest)
    figures_base_savedir = os.path.join(
        cfg['figures-basedir'], pp_pipeline, 'sFC_estimates', cfg['roi-list-name'], data_split
    )
    if not os.path.exists(figures_base_savedir):
        os.makedirs(figures_base_savedir)

    mean_sfc_estimate = np.zeros((n_time_series, n_time_series))
    for i_subject, subject_filename in enumerate(all_subjects_list):
        
        print(f'\n> Subject {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}\n')
        data_file = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries',
            cfg['roi-list-name'], subject_filename
        )

        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        y_train = y

        # Compute static correlation structure (functional connectivity).
        corr_matrix = np.corrcoef(y_train.T)  # (D, D)

        mean_sfc_estimate += corr_matrix

    mean_sfc_estimate /= len(all_subjects_list)

    _plot_static_functional_connectivity_estimate(
        config_dict=cfg,
        correlation_matrix=mean_sfc_estimate,
        brain_regions_of_interest=brain_regions_of_interest,
        mean_estimate=True,
        figures_savedir=figures_base_savedir
    )
