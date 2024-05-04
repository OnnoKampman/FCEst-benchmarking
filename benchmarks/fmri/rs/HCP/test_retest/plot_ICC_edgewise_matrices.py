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


def plot_icc_scores_per_edge_joint(
    config_dict: dict,
    data_dimensionality: str,
    metric: str = 'correlation',
    figures_savedir: str = None,
) -> None:
    """
    Plot ICC test-retest scores per edge jointly.

    Parameters
    ----------
    :param config_dict:
    :param data_dimensionality:
    :param metric:
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
            [.91, 0.38, .03, .19]
        )

        vmin, vmax = 0, 0.65

        for i_model_name, model_name in enumerate(config_dict['models-brain-state-analysis']):

            icc_scores_filepath = os.path.join(
                config_dict['git-results-basedir'], 'test_retest', metric, f'{tvfc_summary_measure:s}_ICCs_{model_name:s}.csv'
            )

            icc_edgewise_df = pd.read_csv(
                icc_scores_filepath, 
                index_col=0,
            )

            n_time_series = icc_edgewise_df.shape[0]  # D

            if data_dimensionality == 'd15':
                icc_edgewise_array, new_rsn_names = reorder_ica_components(
                    config_dict=config_dict,
                    original_matrix=icc_edgewise_df.values,
                    n_time_series=n_time_series,
                    lower_triangular=True
                )
            else:
                icc_edgewise_array = icc_edgewise_df.values
                new_rsn_names = np.arange(n_time_series)

            # Define mask for upper triangular values.
            mask = np.zeros_like(icc_edgewise_array)
            mask[np.triu_indices_from(mask)] = True

            sns.heatmap(
                icc_edgewise_array,
                ax=axes[i_tvfc_summary_measure, i_model_name],
                cmap='jet',  # 'viridis'
                mask=mask,
                vmin=vmin,
                vmax=vmax,
                xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
                yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
                square=True,
                cbar=i_model_name == 2,
                cbar_ax=cbar_ax,
                cbar_kws={
                    'label': "ICC score",
                    'shrink': 0.6,
                },
            )

            # ax.tick_params(left=False, bottom=False)
            # plt.xticks(rotation=45, ha="right", fontsize=figure_ticks_size)
            plt.yticks(rotation=0)

            # plt.xlabel("ICA component", fontsize=figure_label_fontsize)
            # plt.ylabel("ICA component", fontsize=figure_label_fontsize)

    # plt.tight_layout()

    if figures_savedir is not None:
        figure_filename = f'{metric:s}_ICCs_joint.pdf'
        plt.savefig(
            os.path.join(figures_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_filename:s}' in '{figures_savedir:s}'.")
        plt.close()


def plot_icc_scores_per_edge(
    config_dict: dict,
    icc_edgewise_df: pd.DataFrame,
    data_dimensionality: str,
    tvfc_summary_measure: str,
    model_name: str,
    metric: str = 'correlation',
    figures_savedir: str = None,
) -> None:
    """
    Plot ICC test-retest scores per edge.

    TODO: add single label for multiple ICA components belonging to the same RSN

    Parameters
    ----------
    :param config_dict:
    :param icc_edgewise_df:
        DataFrame of shape (D, D).
    :param tvfc_summary_measure:
    :param model_name:
    :param metric:
    :param figures_savedir:
    """
    sns.set(style="white")  # scales colorbar labels as well
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    n_time_series = icc_edgewise_df.shape[0]  # D

    if data_dimensionality == 'd15':
        icc_edgewise_array, new_rsn_names = reorder_ica_components(
            config_dict=config_dict,
            original_matrix=icc_edgewise_df.values,
            n_time_series=n_time_series,
            lower_triangular=True
        )
    else:
        icc_edgewise_array = icc_edgewise_df.values
        new_rsn_names = np.arange(n_time_series)

    # Define mask for upper triangular values.
    mask = np.zeros_like(icc_edgewise_array)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(
        figsize=(4, 4)
    )
    sns.heatmap(
        icc_edgewise_array,
        ax=ax,
        cmap='jet',  # 'viridis'
        mask=mask,
        vmin=0,
        vmax=0.65,
        xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        square=True,
        cbar_kws={
            'label': f"TVFC {tvfc_summary_measure.replace('_', '-'):s}\nICC score",
            'shrink': 0.6,
        },
    )

    # ax.tick_params(left=False, bottom=False)
    # plt.xticks(rotation=45, ha="right", fontsize=figure_ticks_size)
    plt.yticks(rotation=0)

    # plt.xlabel("ICA component", fontsize=figure_label_fontsize)
    # plt.ylabel("ICA component", fontsize=figure_label_fontsize)
    # plt.tight_layout()

    if figures_savedir is not None:
        figure_filename = f'{metric:s}_{tvfc_summary_measure:s}_ICCs_{model_name:s}.pdf'
        plt.savefig(
            os.path.join(figures_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_filename:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_icc_scores_per_edge_binned(
    config_dict: dict,
    icc_edgewise_df: pd.DataFrame,
    tvfc_summary_measure: str,
    model_name: str,
    metric: str,
    figures_savedir: str,
) -> None:
    """
    Save figure with performance bins.
    """
    sns.set(style="whitegrid", font_scale=1.1)  # scales colorbar labels as well
    plt.rcParams["font.family"] = 'serif'

    n_time_series = icc_edgewise_df.shape[0]  # D

    if data_dimensionality == 'd15':
        icc_edgewise_array, new_rsn_names = reorder_ica_components(
            config_dict=config_dict,
            original_matrix=icc_edgewise_df.values,
            n_time_series=n_time_series,
            lower_triangular=True
        )
    else:
        icc_edgewise_array = icc_edgewise_df.values
        new_rsn_names = np.arange(n_time_series)

    plt.figure()

    # choose 4 colours to create 4 bins
    colors = sns.color_palette('viridis', 4)
    levels = [0, 0.4, 0.6, 0.75]
    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="max")

    fig, ax = plt.subplots()
    im = ax.imshow(
        icc_edgewise_array,
        cmap=cmap,
        norm=norm
    )
    # ax.set(
    #     xticks=range(scores_df_sw.shape[1]),
    #     yticks=range(scores_df_sw.shape[0]),
    #     xticklabels=flights.columns, yticklabels=flights.index
    # )

    plt.xticks([])
    plt.yticks([])
    # plt.xticks(fontsize=figure_ticks_size)
    # plt.yticks(rotation=0, fontsize=figure_ticks_size)

    fig.colorbar(im, ax=ax, spacing="proportional")
    # plt.tight_layout()

    figure_name = f'{metric:s}_{tvfc_summary_measure:s}_ICCs_{model_name:s}_binned.pdf'
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
    plt.close()


def _plot_icc_scores_per_edge_two_model_scatterplot(
    summary_measure_icc_df_1: pd.DataFrame,
    summary_measure_icc_df_2: pd.DataFrame,
    model_name_1: str,
    model_name_2: str,
    tvfc_summary_measure: str,
    metric: str,
    figures_savedir: str,
    figure_label_fontsize=20,
):
    sns.set(style="whitegrid", font_scale=1.1)  # scales colorbar labels as well
    plt.rcParams["font.family"] = 'serif'

    plt.figure()
    plt.scatter(
        summary_measure_icc_df_1.values.flatten(),
        summary_measure_icc_df_2.values.flatten()
    )
    plt.plot([0, 1], [0, 1], linestyle='dashed')

    plt.xlim([0, 0.69])
    plt.ylim([0, 0.69])

    plt.xlabel(model_name_1, fontsize=figure_label_fontsize)
    plt.ylabel(model_name_2, fontsize=figure_label_fontsize)
    # plt.tight_layout()

    figure_name = f'{metric:s}_{tvfc_summary_measure:s}_ICCs__scatter_{model_name_1:s}_{model_name_2:s}.pdf'
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
    plt.close()


def _plot_violin():
    raise NotImplementedError


def _plot_bland_altman():
    raise NotImplementedError


if __name__ == "__main__":

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    metric = sys.argv[2]               # 'covariance', 'correlation'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    figures_savedir = os.path.join(cfg['figures-basedir'], 'test_retest', 'ICCs')
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:

        for model_name in cfg['test-retest-models']:
            print('\nMODEL NAME:', model_name, '\n')

            icc_scores_filepath = os.path.join(
                cfg['git-results-basedir'], 'test_retest', metric, f'{tvfc_summary_measure:s}_ICCs_{model_name:s}.csv'
            )
            if os.path.exists(icc_scores_filepath):
                summary_measure_icc_df = pd.read_csv(
                    icc_scores_filepath, 
                    index_col=0,
                )
                plot_icc_scores_per_edge(
                    config_dict=cfg,
                    icc_edgewise_df=summary_measure_icc_df,
                    tvfc_summary_measure=tvfc_summary_measure,
                    model_name=model_name,
                    metric=metric,
                    figures_savedir=figures_savedir
                )
                _plot_icc_scores_per_edge_binned(
                    config_dict=cfg,
                    icc_edgewise_df=summary_measure_icc_df,
                    tvfc_summary_measure=tvfc_summary_measure,
                    model_name=model_name,
                    metric=metric,
                    figures_savedir=figures_savedir
                )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        _plot_icc_scores_per_edge_two_model_scatterplot(
            summary_measure_icc_df_1=pd.read_csv(
                os.path.join(
                    cfg['git-results-basedir'], 'test_retest', metric,
                    f'{tvfc_summary_measure:s}_ICCs_DCC_joint.csv'
                ),
                index_col=0
            ),
            summary_measure_icc_df_2=pd.read_csv(
                os.path.join(
                    cfg['git-results-basedir'], 'test_retest', metric,
                    f'{tvfc_summary_measure:s}_ICCs_SVWP_joint.csv'
                ),
                index_col=0
            ),
            model_name_1='DCC_joint',
            model_name_2='SVWP_joint',
            tvfc_summary_measure=tvfc_summary_measure,
            metric=metric,
            figures_savedir=figures_savedir
        )
