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


def plot_tvfc_summary_measures_mean_over_subjects_all_edges_joint(
    config_dict: dict,
    data_dimensionality: str,
    scan_id: int = 0,
    connectivity_metric: str = 'correlation',
    data_split: str = 'all',
    experiment_dimensionality: str = 'multivariate',
    figures_savedir: str = None,
) -> None:
    """
    Plots the mean over subjects of a certain TVFC summary measure.
    Similar to Figure 6 from Choe et al. (2017).

    Parameters
    ----------
    :param config_dict:
    :param data_dimensionality:
    :param scan_id:
    :param connectivity_metric:
    :param data_split:
    :param experiment_dimensionality:
    :param figures_savedir:
    """
    sns.set(style="white")  # scales colorbar labels as well
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(9.2, 9.2),
        sharex=True,
        sharey=True,
    )

    for i_tvfc_summary_measure, tvfc_summary_measure in enumerate(config_dict['plot-TVFC-summary-measures']):

        # tuple (left, bottom, width, height)
        cbar_ax = fig.add_axes(
            [.91, 0.652 - i_tvfc_summary_measure * 0.272, .03, .19]
        )

        vmin, vmax = _set_colorbar_min_max(tvfc_summary_measure)

        for i_model_name, model_name in enumerate(config_dict['models-brain-state-analysis']):

            tvfc_estimates_git_savedir = os.path.join(
                config_dict['git-results-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, connectivity_metric
            )
            file_name = f"{model_name:s}_TVFC_{tvfc_summary_measure:s}_mean_over_subjects.csv"
            filepath = os.path.join(tvfc_estimates_git_savedir, file_name)

            summarized_tvfc_df = pd.read_csv(
                filepath,
                index_col=0,
            )  # (D, D)

            # Re-order ICA components.
            n_time_series = summarized_tvfc_df.shape[0]
            if data_dimensionality == 'd15':
                summarized_tvfc_array, new_rsn_names = reorder_ica_components(
                    config_dict=config_dict,
                    original_matrix=summarized_tvfc_df.values,
                    n_time_series=n_time_series,
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
                ax=axes[i_tvfc_summary_measure, i_model_name],
                cmap='jet',
                mask=mask,
                vmin=vmin,
                vmax=vmax,
                xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
                yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
                square=True,
                cbar=i_model_name == 2,
                cbar_ax=cbar_ax,
                cbar_kws={
                    'label': f"TVFC {tvfc_summary_measure.replace('_', '-'):s}",
                    'shrink': 0.6,
                },
            )

            # ax.tick_params(left=False, bottom=False)
            # plt.xticks(rotation=45, ha="right", fontsize=figure_ticks_size)
            plt.yticks(rotation=0)

            # if i_tvfc_summary_measure == 2:
            #     axes[2, i_model_name].set_xlabel(f"{model_name:s}")

            # if i_model_name == 0:
            #     axes[i_tvfc_summary_measure, 0].set_ylabel(f"TVFC {tvfc_summary_measure.replace('_', '-'):s}")

    # plt.tight_layout()

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_filename = f"{connectivity_metric:s}_TVFC_summary_measures_joint.pdf"
        plt.savefig(
            os.path.join(figures_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        plt.close()
        logging.info(f"Saved figure '{figure_filename:s}' in '{figures_savedir:s}'.")


def plot_tvfc_summary_measures_mean_over_subjects_all_edges(
    config_dict: dict,
    summarized_tvfc_df: pd.DataFrame,
    summary_measure: str,
    model_name: str,
    data_dimensionality: str,
    connectivity_metric: str = 'correlation',
    figures_savedir: str = None,
) -> None:
    """
    Plots the mean over subjects of a certain TVFC summary measure.
    Similar to Figure 6 from Choe et al. (2017).

    Parameters
    ----------
    :param config_dict:
    :param summarized_tvfc_df:
        Array of shape (D, D).
    :param connectivity_metric:
    :param summary_measure:
    :param model_name:
    :param data_split:
    :param figures_savedir:
    """
    sns.set(style="white")  # scales colorbar labels as well
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    vmin, vmax = _set_colorbar_min_max(summary_measure)

    # Re-order ICA components.
    n_time_series = summarized_tvfc_df.shape[0]
    if data_dimensionality == 'd15':
        summarized_tvfc_array, new_rsn_names = reorder_ica_components(
            config_dict=config_dict,
            original_matrix=summarized_tvfc_df.values,
            n_time_series=n_time_series,
        )
    else:
        # TODO: add RSN names map for d50
        summarized_tvfc_array = summarized_tvfc_df.values
        new_rsn_names = np.arange(n_time_series)

    # Define mask for diagonal and upper triangular values.
    mask = np.zeros_like(summarized_tvfc_array)
    mask[np.triu_indices_from(mask, k=0)] = True

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )
    sns.heatmap(
        summarized_tvfc_array,
        ax=ax,
        cmap='jet',
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        square=True,
        cbar_kws={
            'label': f"TVFC {summary_measure.replace('_', '-'):s}",
            'shrink': 0.6,
        },
    )

    # ax.tick_params(left=False, bottom=False)
    # plt.xticks(rotation=45, ha="right", fontsize=figure_ticks_size)
    plt.yticks(rotation=0)

    # plt.tight_layout()

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


def _set_colorbar_min_max(summary_measure: str) -> tuple[float, float]:

    match summary_measure:
        case 'ar1':
            vmin = -1
            vmax = 1
        case 'mean':
            vmin = -1
            vmax = 1
        case 'rate_of_change':
            vmin = 0
            vmax = 0.4
        case 'std':
            vmin = 0
            vmax = 0.3
        case 'variance':
            vmin = 0
            vmax = 0.09
        case _:
            raise NotImplementedError(f"Summary measure '{summary_measure:s}' not recognized.")

    return vmin, vmax


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
                        filepath,
                        index_col=0,
                    )  # (D, D)
                    logging.info(f"Loaded TVFC summaries '{file_name:s}' from '{tvfc_estimates_git_savedir:s}'.")

                    plot_tvfc_summary_measures_mean_over_subjects_all_edges(
                        config_dict=cfg,
                        summarized_tvfc_df=mean_over_subjects_tvfc_summaries_df,
                        summary_measure=tvfc_summary_measure,
                        model_name=model_name,
                        connectivity_metric=metric,
                        figures_savedir=os.path.join(
                            cfg['figures-basedir'], 'TVFC_estimates_summary_measures',
                            f'scan_{scan_id:d}', data_split,
                        ),
                    )
                else:
                    logging.warning(f"TVFC summaries '{filepath:s}' not found.")
