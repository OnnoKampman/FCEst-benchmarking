import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import reorder_ica_components
from helpers.hcp import load_human_connectome_project_data, get_human_connectome_project_subjects


def _plot_static_functional_connectivity_estimate(
        config_dict: dict, correlation_matrix: np.array,
        subject: str = None, mean_estimate: bool = False,
        figures_savedir: str = None
) -> None:
    """
    Plot static functional connectivity, i.e. just a correlation matrix.
    """
    sns.set(style="whitegrid", font_scale=1.0)
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    if mean_estimate:
        figure_name_correlation_matrix = "mean_sFC_correlation_matrix.pdf"
    else:
        figure_name_correlation_matrix = f"{subject:s}_sFC_correlation_matrix.pdf"

    n_time_series = correlation_matrix.shape[0]

    # Re-order ICA components.
    re_ordered_correlation_matrix, new_rsn_names = reorder_ica_components(
        config_dict=config_dict,
        original_matrix=correlation_matrix,
        n_time_series=n_time_series,
    )

    plt.figure()
    sns.heatmap(
        correlation_matrix,
        cmap='jet',  # or 'viridis'
        # xticklabels=new_rsn_names,
        # yticklabels=new_rsn_names,
        xticklabels=False,
        yticklabels=False,
        square=True,
        vmin=-0.8,
        vmax=0.8,
    )
    plt.xticks(fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    # plt.xlabel("ICA component", fontsize=12)
    # plt.ylabel("ICA component", fontsize=12)
    # plt.tight_layout()

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

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_time_series = int(data_dimensionality[1:])
    scan_ids = cfg['scan-ids']
    subjects = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'], as_ints=True
    )

    mean_sfc_estimate = np.zeros((n_time_series, n_time_series))
    for i_subject, subject in enumerate(subjects):

        print(f"\n> SUBJECT {i_subject+1:d}: {subject:d}\n")
        data_file = os.path.join(cfg['data-dir'], f'{subject:d}.txt')

        for scan_id in scan_ids:
            print(f'SCAN ID {scan_id:d}')

            x, y = load_human_connectome_project_data(
                data_file, scan_id=scan_id, verbose=False
            )  # (N, 1), (N, D)
            y_train = y

            # Compute static correlation structure (functional connectivity).
            corr_matrix = np.corrcoef(y_train.T)  # (D, D)

            mean_sfc_estimate += corr_matrix

            # _plot_static_functional_connectivity_estimate(
            #     config_dict=cfg,
            #     correlation_matrix=corr_matrix,
            #     subject=subject,
            #     figures_savedir = os.path.join(
            #         cfg['figures-basedir'], 'sFC_estimates', f'scan_{scan_id:d}', data_split
            #     )
            # )

    mean_sfc_estimate /= len(subjects)
    mean_sfc_estimate /= len(scan_ids)

    _plot_static_functional_connectivity_estimate(
        config_dict=cfg,
        correlation_matrix=corr_matrix,
        mean_estimate=True,
        figures_savedir = os.path.join(
            cfg['figures-basedir'], 'sFC_estimates', data_split
        )
    )
