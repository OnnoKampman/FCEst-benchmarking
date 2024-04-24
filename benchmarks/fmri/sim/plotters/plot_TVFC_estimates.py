import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split
from helpers.figures import get_palette
from helpers.plotters import plot_method_tvfc_estimates
from helpers.synthetic_covariance_structures import get_ground_truth_covariance_structure, get_ylim, to_human_readable


LINEWIDTH = 1.5


def plot_d2_all_covariance_structures(
    config_dict: dict,
    signal_to_noise_ratio: float,
    connectivity_metric: str,
    time_series_noise_type: str,
    data_split: str,
    i_trial: int,
    figsize: tuple[float] = (5.4, 10.4),
    ground_truth_linewidth: float = LINEWIDTH,
    figures_savedir: str = None,
) -> None:
    """
    Plots bivariate correlation edge for all synthetic covariance structures considered.

    Parameters
    ----------
    :param config_dict:
    :param signal_to_noise_ratio:
    :param figure_filename:
        Note that .eps files do not render transparency plots.
    :param connectivity_metric:
    :param time_series_noise_type:
    :param data_split:
    :param i_trial:
    :param figsize:
    :param figures_savedir:
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    n_covs_types = len(config_dict['plot-covs-types'])

    fig, ax = plt.subplots(
        nrows=n_covs_types,
        ncols=1,
        sharex=True,
        figsize=figsize,
    )
    for i_covs_type, covs_type in enumerate(config_dict['plot-covs-types']):

        print(f"\n> Covs type: {covs_type:s}\n")

        data_file = os.path.join(
            config_dict['data-dir'], time_series_noise_type,
            f'trial_{i_trial:03d}', f'{covs_type:s}_covariance.csv'
        )
        if not os.path.exists(data_file):
            logging.warning(f"Data file '{data_file:s}' not found.")
            if covs_type == 'boxcar':
                data_file = os.path.join(
                    config_dict['data-dir'], time_series_noise_type,
                    f'trial_{i_trial:03d}', 'checkerboard_covariance.csv'
                )
                if not os.path.exists(data_file):
                    logging.warning(f"Data file '{data_file:s}' not found.")
                    continue
            else:
                continue

        x, y = load_data(
            data_file,
            verbose=False,
        )  # (N, 1), (N, D)
        ground_truth_covariance_structure = get_ground_truth_covariance_structure(
            covs_type=covs_type,
            n_samples=len(x),
            signal_to_noise_ratio=signal_to_noise_ratio,
            data_set_name=config_dict['data-set-name'],
        )

        ax[i_covs_type].plot(
            x, [step[0, 1] for step in ground_truth_covariance_structure],
            color='dimgray',
            # linestyle='dashed',
            linewidth=ground_truth_linewidth,
            alpha=0.5,
            label='Ground\nTruth',
        )

        for i_model_name, model_name in enumerate(config_dict['plot-models']):

            plot_color = get_palette(
                config_dict['plot-models']
            )[i_model_name]

            plot_method_tvfc_estimates(
                config_dict=config_dict,
                model_name=model_name,
                x_train_locations=x,
                y_train_locations=y,
                data_split=data_split,
                i_trial=i_trial,
                noise_type=time_series_noise_type,
                covs_type=covs_type,
                metric=connectivity_metric,
                i_time_series=0,
                j_time_series=1,
                plot_color=plot_color,
                ax=ax[i_covs_type],
            )

        plot_color = get_palette(
            config_dict['plot-models']
        )[0]

        plot_method_tvfc_estimates(
            config_dict=config_dict,
            model_name='SVWP',
            x_train_locations=x,
            y_train_locations=y,
            data_split=data_split,
            i_trial=i_trial,
            noise_type=time_series_noise_type,
            covs_type=covs_type,
            metric=connectivity_metric,
            i_time_series=0,
            j_time_series=1,
            plot_color=plot_color,
            ax=ax[i_covs_type],
        )

        ax[i_covs_type].set_xlim(config_dict['plot-data-xlim'])
        ax[i_covs_type].set_ylim(
            get_ylim(covs_type=covs_type)
        )
        ax[i_covs_type].set_ylabel(
            to_human_readable(covs_type),
            rotation=0,
            labelpad=10,
        )

        if i_covs_type == 0:
            ax[i_covs_type].legend(
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
                title='TVFC\nestimator',
                alignment='left',
            )

    # plt.legend(frameon=True, title='cohort')

    ax[-1].set_xlabel('time [a.u.]')
    plt.subplots_adjust(
        hspace=0.12,
        wspace=0,
    )

    if figures_savedir is not None:
        figure_filename = f'all_covs_types_{connectivity_metric:s}s.pdf'
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        fig.savefig(
            os.path.join(figures_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_filename:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_d2_tvfc_estimates_single_covariance_structure(
    config_dict: dict,
    x_train_locations: np.array,
    y_train_locations: np.array,
    ground_truth_covariance_structure: np.array,
    figure_filename: str,
    connectivity_metric: str,
    time_series_noise_type,
    data_split: str,
    i_trial: int,
    covs_type,
    markersize=3.6,
    bbox_to_anchor=(1.19, 1.0),
    legend_fontsize=12,
    figure_savedir: str = None,
) -> None:
    """
    Plots bivariate pair of time series and the predicted covariance structure in one figure.

    Parameters
    ----------
    :param config_dict:
    :param x_train_locations:
    :param y_train_locations:
    :param ground_truth_covariance_structure:
    :param figure_filename:
    :param markersize:
    :param bbox_to_anchor:
    :param legend_fontsize:
    """
    sns.set(style="whitegrid", font_scale=1.3)
    font = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14
    }
    plt.rc('font', **font)

    plt.figure(figsize=config_dict['figure-model-estimates-figsize'])

    n_time_series = y_train_locations.shape[1]
    for i_time_series in range(n_time_series):
        plt.subplot(4, 1, i_time_series + 1)
        plt.plot(
            x_train_locations, y_train_locations[:, i_time_series],
            '-', markersize=markersize, label=f'Time series {i_time_series+1:d}'
        )
        plt.xlim(config_dict['plot-data-xlim'])
        plt.ylim(config_dict['plot-time-series-ylim'])
        plt.gca().get_xaxis().set_visible(False)
        plt.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, frameon=True)

    # Plot covariance estimates.
    plt.subplot(212)

    # Plot ground truth.
    plt.plot(
        x_train_locations, [step[0, 1] for step in ground_truth_covariance_structure],
        color='black',
        linewidth=4,
        label='GT'
    )
    for model_name in config_dict['plot-models']:
        plot_method_tvfc_estimates(
            config_dict=config_dict,
            model_name=model_name,
            x_train_locations=x_train_locations,
            y_train_locations=y_train_locations,
            data_split=data_split,
            i_trial=i_trial,
            noise_type=time_series_noise_type,
            covs_type=covs_type,
            metric=connectivity_metric,
            i_time_series=0,
            j_time_series=1
        )
    plt.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, frameon=True, fontsize=legend_fontsize)
    plt.xlim(config_dict['plot-data-xlim'])
    plt.ylim(get_ylim(covs_type=covs_type))
    plt.xlabel('time [a.u.]')
    plt.ylabel(f'{connectivity_metric:s} estimate')

    if figure_savedir is not None:
        if not os.path.exists(figure_savedir):
            os.makedirs(figure_savedir)
        plt.savefig(
            os.path.join(figure_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_filename:s}' in '{figure_savedir:s}'.")
        plt.close()


def _plot_d3_tvfc_estimates(
        config_dict: dict,
        x_train_locations: np.array, y_train_locations: np.array, ground_truth_covariance_structure: np.array,
        connectivity_metric: str, time_series_noise_type: str, data_split: str, i_trial: int, covs_type: str,
        markersize=3.6, bbox_to_anchor=(1.2, 1.0),
        figure_filename: str = None, figure_savedir: str = None
) -> None:
    """
    Plots 3 time series and the 3 (lower triangular) correlation terms in one figure.
    The covariance term estimates take up twice as much space as the time series plots.

    TODO: refactor to accept a list of models to plot
    TODO: sync ylims across trials and noise types
    TODO: fix ylims for sparse case

    Parameters
    ----------
    :param config_dict:
    :param x_train_locations:
    :param y_train_locations:
    :param ground_truth_covariance_structure:
    :param figure_filename:
    :param connectivity_metric:
    :param markersize:
    :param bbox_to_anchor: used to put the legend outside of the plot
    """
    sns.set(style="whitegrid", font_scale=1.5)
    font = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 14
    }
    plt.rc('font', **font)

    fig, axes = plt.subplots(
        6, 1, sharex=True,
        figsize=config_dict['figure-model-estimates-figsize'],
        gridspec_kw={
            'height_ratios': [1, 1, 1, 3, 3, 3]
        }
    )

    n_time_series = y_train_locations.shape[1]

    for i_time_series in range(n_time_series):
        axes[i_time_series].plot(
            x_train_locations, y_train_locations[:, i_time_series],
            '-', markersize=markersize, label=f'Node\n{i_time_series+1:d}'
        )
        axes[i_time_series].set_xlim(config_dict['plot-data-xlim'])
        axes[i_time_series].set_ylim(config_dict['plot-time-series-ylim'])
        # loc, _ = plt.xticks()
        # plt.xticks(ticks=loc, labels=[])
        axes[i_time_series].set_ylabel(f'Node\n{i_time_series+1:d}', rotation='horizontal', va='center', labelpad=30)

    # Plot covariance/correlation terms.
    i_interaction = 3
    for i_time_series in range(n_time_series):
        for j_time_series in range(i_time_series + 1, n_time_series):
            print(f'\n> (i, j) = ({i_time_series:d}, {j_time_series:d})')

            # Plot ground truth.
            axes[i_interaction].plot(
                x_train_locations, [step[i_time_series, j_time_series] for step in ground_truth_covariance_structure],
                color='black',
                linewidth=4,
                label='GT'
            )
            for model_name in config_dict['plot-models']:
                plot_method_tvfc_estimates(
                    config_dict=config_dict,
                    ax=axes[i_interaction],
                    model_name=model_name,
                    x_train_locations=x_train_locations,
                    y_train_locations=y_train_locations,
                    data_split=data_split,
                    i_trial=i_trial,
                    noise_type=time_series_noise_type,
                    covs_type=covs_type,
                    metric=connectivity_metric,
                    i_time_series=i_time_series,
                    j_time_series=j_time_series
                )
            if i_interaction == 3:
                axes[i_interaction].legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, frameon=True)
            axes[i_interaction].set_xlim(config_dict['plot-data-xlim'])
            axes[i_interaction].set_ylim(get_ylim(covs_type=covs_type))
            axes[i_interaction].set_ylabel(
                f"Edge\n{i_time_series+1:d} - {j_time_series+1:d}", rotation='horizontal', va='center', labelpad=30
            )
            i_interaction += 1
    axes[-1].set_xlabel('time [a.u.]')
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0)
    if figure_savedir is not None:
        if not os.path.exists(figure_savedir):
            os.makedirs(figure_savedir)
        fig.savefig(
            os.path.join(figure_savedir, figure_filename),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_filename:s}' in '{figure_savedir:s}'.")
        plt.close()


def _plot_d4_tvfc_estimates(
    config_dict: dict,
    x_train_locations: np.array,
    y_train_locations: np.array,
    ground_truth_covariance_structure: np.array,
    connectivity_metric: str,
    time_series_noise_type: str,
    data_split: str,
    i_trial: int,
    covs_type: str,
    markersize=3.6,
    bbox_to_anchor=(1.2, 1.0),
    figure_filename: str = None,
    figure_savedir: str = None,
) -> None:
    """
    Parameters
    ----------
    :param config_dict:
    :param x_train_locations:
    :param y_train_locations:
    :param ground_truth_covariance_structure:
    :param connectivity_metric:
    :param time_series_noise_type:
    :param data_split:
    :param i_trial:
    :param covs_type:
    :param markersize:
    :param bbox_to_anchor:
    :param figure_filename:
    :param figure_savedir:
    """
    raise NotImplementedError


if __name__ == "__main__":

    i_trial = 0

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'
    metric = sys.argv[4]           # 'correlation', or 'covariance'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )

    for noise_type in cfg['noise-types']:
        if noise_type != 'no_noise':
            SNR = float(noise_type[-1])
        else:
            SNR = None

        # Plot figure for all (bivariate) synthetic covariance structures jointly.
        if data_set_name == 'd2':  # TODO: also plot this for any sparse case
            plot_d2_all_covariance_structures(
                config_dict=cfg,
                signal_to_noise_ratio=SNR,
                connectivity_metric=metric,
                time_series_noise_type=noise_type,
                i_trial=i_trial,
                data_split=data_split,
                figures_savedir=os.path.join(
                    cfg['figures-basedir'], noise_type, data_split,
                    "TVFC_estimates", f'trial_{i_trial:03d}'
                )
            )

        # Plot individual figures for each synthetic covariance structure.
        # for covs_type in cfg['all-covs-types']:
        #     data_file = os.path.join(
        #         cfg['data-dir'], noise_type, f'trial_{i_trial:03d}', f'{covs_type:s}_covariance.csv'
        #     )
        #     if not os.path.exists(data_file):
        #         logging.warning(f"Data file '{data_file:s}' not found.")
        #         continue
        #     x, y = load_data(
        #         data_file,
        #         verbose=False,
        #     )  # (N, 1), (N, D)
        #     gt_cov_structure = get_ground_truth_covariance_structure(
        #         covs_type=covs_type,
        #         n_samples=len(x),
        #         signal_to_noise_ratio=SNR,
        #         data_set_name=data_set_name,
        #     )

        #     figure_name = f'{covs_type:s}_{metric:s}s.pdf'

        #     if data_split == 'LEOO':
        #         x_train, x_test = leave_every_other_out_split(x)
        #         y_train, y_test = leave_every_other_out_split(y)
        #         gt_cov_structure_train, covs_test = leave_every_other_out_split(gt_cov_structure)
        #         figure_name = '%s_train_locations.png' % figure_name[:-4]
        #     else:
        #         x_train = x
        #         y_train = y
        #         gt_cov_structure_train = gt_cov_structure

        #     match data_set_name:
        #         case 'd2':
        #             _plot_d2_tvfc_estimates_single_covariance_structure(
        #                 config_dict=cfg,
        #                 x_train_locations=x_train,
        #                 y_train_locations=y_train,
        #                 ground_truth_covariance_structure=gt_cov_structure,
        #                 figure_filename=figure_name,
        #                 connectivity_metric=metric,
        #                 time_series_noise_type=noise_type,
        #                 data_split=data_split,
        #                 covs_type=covs_type,
        #                 i_trial=i_trial,
        #                 figure_savedir=os.path.join(cfg['figures-basedir'], noise_type, data_split, "TVFC_estimates", f'trial_{i_trial:03d}')
        #             )
        #         case 'd3d' | 'd3s':
        #             _plot_d3_tvfc_estimates(
        #                 config_dict=cfg,
        #                 x_train_locations=x_train,
        #                 y_train_locations=y_train,
        #                 ground_truth_covariance_structure=gt_cov_structure,
        #                 connectivity_metric=metric,
        #                 covs_type=covs_type,
        #                 time_series_noise_type=noise_type,
        #                 i_trial=i_trial,
        #                 data_split=data_split,
        #                 figure_filename=figure_name,
        #                 figure_savedir=os.path.join(
        #                     cfg['figures-basedir'], noise_type, data_split,
        #                     'TVFC_estimates', f'trial_{i_trial:03d}'
        #                 )
        #             )
        #         case 'd4s':
        #             _plot_d4_tvfc_estimates(
        #                 config_dict=cfg,
        #                 x_train_locations=x_train,
        #                 y_train_locations=y_train,
        #                 ground_truth_covariance_structure=gt_cov_structure,
        #                 connectivity_metric=metric,
        #                 covs_type=covs_type,
        #                 time_series_noise_type=noise_type,
        #                 i_trial=i_trial,
        #                 data_split=data_split,
        #             )
        #         case _:
        #             raise NotImplementedError
