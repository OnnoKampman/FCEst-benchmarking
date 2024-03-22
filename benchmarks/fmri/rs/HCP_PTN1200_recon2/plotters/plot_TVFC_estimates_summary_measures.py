import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import reorder_ica_components


def _plot_tvfc_summary_measures_mean_over_subjects_all_edges(
        config_dict: dict, summarized_tvfc_df: pd.DataFrame,
        scan_id: int, connectivity_metric: str, summary_measure: str, model_name: str,
        data_split: str = 'all'
) -> None:
    """
    Plots the mean over subjects of a certain TVFC summary measure.
    Similar to Figure 6 from Choe et al. (2017).

    :param config_dict:
    :param summarized_tvfc_df: array of shape (D, D).
    :param scan_id:
    :param connectivity_metric:
    :param summary_measure:
    :param model_name:
    :param data_split:
    :return:
    """
    sns.set(style="whitegrid", font_scale=1.2)  # scales colorbar labels as well
    plt.rcParams["font.family"] = 'serif'

    # Set colorbar minimum and maximum.
    match summary_measure:
        case 'mean':
            vmin = -1
            vmax = 1
        case 'variance':
            vmin = 0
            vmax = 0.09
        case 'std':
            vmin = 0
            vmax = 0.3
        case 'rate_of_change':
            vmin = 0
            vmax = 0.4
        case 'ar1':
            vmin = -1
            vmax = 1
        case _:
            raise NotImplementedError(f"Summary measure '{summary_measure:s}' not recognized.")

    # Re-order ICA components.
    n_time_series = summarized_tvfc_df.shape[0]
    if data_dimensionality == 'd15':
        summarized_tvfc_array, new_rsn_names = reorder_ica_components(
            config_dict=config_dict,
            original_matrix=summarized_tvfc_df.values,
            n_time_series=n_time_series
        )
    else:
        # TODO: add RSN names map for d50
        summarized_tvfc_array = summarized_tvfc_df.values
        new_rsn_names = np.arange(n_time_series)

    # Define mask for diagonal and upper triangular values.
    mask = np.zeros_like(summarized_tvfc_array)
    mask[np.triu_indices_from(mask, k=0)] = True

    sns.heatmap(
        summarized_tvfc_array,
        cmap='jet',
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        square=True,
        cbar_kws={
            'label': f"TVFC {summary_measure.replace('_', '-'):s}",
            'shrink': 0.6
        }
    )

    # ax.tick_params(left=False, bottom=False)
    # plt.xticks(rotation=45, ha="right", fontsize=figure_ticks_size)
    plt.yticks(rotation=0)

    # plt.tight_layout()
    figures_savedir = os.path.join(
        cfg['figures-basedir'], 'TVFC_estimates_summary_measures', f'scan_{scan_id:d}', data_split
    )
    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_filename = f"{connectivity_metric:s}_TVFC_{summary_measure:s}_{model_name:s}.pdf"
        plt.savefig(
            os.path.join(figures_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        plt.close()
        logging.info(f"Saved figure '{figure_filename:s}' in '{figures_savedir:s}'.")


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    metric = sys.argv[2]               # 'covariance', 'correlation'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        for model_name in cfg['plot-model-estimates-summary-measures-methods']:
            for scan_id in cfg['scan-ids']:
                print(f'\n> SUMMARY MEASURE: {tvfc_summary_measure:s}')
                print(f'> MODEL NAME: {model_name:s}')
                print(f'> SCAN ID {scan_id:d}')

                tvfc_estimates_git_savedir = os.path.join(
                    cfg['git-results-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                    data_split, experiment_dimensionality, metric
                )
                file_name = f"{model_name:s}_TVFC_{tvfc_summary_measure:s}_mean_over_subjects.csv"
                filepath = os.path.join(tvfc_estimates_git_savedir, file_name)
                if os.path.exists(filepath):
                    mean_over_subjects_tvfc_summaries_df = pd.read_csv(
                        filepath, index_col=0
                    )  # (D, D)
                    logging.info(f"Loaded TVFC summaries '{file_name:s}' from '{tvfc_estimates_git_savedir:s}'.")

                    _plot_tvfc_summary_measures_mean_over_subjects_all_edges(
                        config_dict=cfg,
                        summarized_tvfc_df=mean_over_subjects_tvfc_summaries_df,
                        scan_id=scan_id,
                        connectivity_metric=metric,
                        summary_measure=tvfc_summary_measure,
                        model_name=model_name
                    )
                else:
                    logging.warning(f"TVFC summaries '{filepath:s}' not found.")
