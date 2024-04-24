import logging
import os
import socket
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict


def plot_quantitative_results(
    config_dict: dict,
    performance_metric: str,
    noise_type: str,
    data_split: str,
    figsize: tuple[float] = (4.6, 2.6),
    bbox_to_anchor: tuple[float] = (1.01, 1.0),
    markersize: float = 3.0,
    capsize: float = 2.5,
    stddev_error_bars: bool = True,
    figures_savedir: str = None,
    is_fig_s1: bool = False,
) -> None:
    """
    Plot quantitative (reconstruction errors) results for all methods and synthetic covariance structures.

    Note that the figure will be slightly different for Fig. S1.

    TODO: replace with violin plot, after saving results for all individual trials
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    average_quantitative_results_df = pd.read_csv(
        os.path.join(
            config_dict['git-results-basedir'], noise_type, data_split,
            f'{performance_metric:s}_mean.csv'
        ),
        index_col=0
    )
    if stddev_error_bars:
        stddev_quantitative_results_df = pd.read_csv(
            os.path.join(
                config_dict['git-results-basedir'], noise_type, data_split,
                f'{performance_metric:s}_std.csv'
            ),
            index_col=0
        )
    else:
        stderr_quantitative_results_df = pd.read_csv(
            os.path.join(
                config_dict['git-results-basedir'], noise_type, data_split,
                f'{performance_metric:s}_se.csv'
            ),
            index_col=0
        )

    average_quantitative_results_df = _process_df(
        config_dict, average_quantitative_results_df, is_fig_s1
    )
    if stddev_error_bars:
        errorbars_quantitative_results_df = _process_df(
            config_dict, stddev_quantitative_results_df, is_fig_s1
        )
    else:
        errorbars_quantitative_results_df = _process_df(
            config_dict, stderr_quantitative_results_df, is_fig_s1
        )

    models = average_quantitative_results_df.index
    covs_types = average_quantitative_results_df.columns
    offsets = _get_offsets(n_models=len(models))

    if is_fig_s1:
        palette_tab10 = sns.color_palette("tab10", 10)
        cmap = matplotlib.colormaps['Set1']
        cmap = sns.color_palette(
            [palette_tab10[3], palette_tab10[5], palette_tab10[6], palette_tab10[7], palette_tab10[8], palette_tab10[4]],
            as_cmap=True
        )
    else:
        cmap = matplotlib.colormaps['tab10']
        # cmap = sns.color_palette(palette="tab10", n_colors=10, as_cmap=True)

    plt.figure(
        # figsize=config_dict['figure-quantitative-results-figsize'],
        figsize=figsize,
    )
    for i_model, model in enumerate(models):
        transform = Affine2D().translate(offsets[i_model], 0.0) + plt.gca().transData

        if is_fig_s1:
            model_color = cmap(i_model)  # for Fig. S1 only
            model_color = cmap[i_model]  # for Fig. S1 only
        else:
            model_color = cmap(i_model + 1) if config_dict['data-set-name'] == 'd2' and i_model > 1 else cmap(i_model)  # this line makes sure colors are assigned to the same model across plots

        plt.errorbar(
            x=covs_types,
            y=average_quantitative_results_df.T[model],
            yerr=errorbars_quantitative_results_df.T[model],
            capsize=capsize,  # the horizontal line at top and bottom of error bars
            color=model_color,
            label=model,
            linestyle='none',
            marker='o',
            markersize=markersize,
            transform=transform,
        )
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(performance_metric.replace('_', ' ').replace(' RMSE', '\nRMSE'))
    plt.legend(
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        title='TVFC\nestimator',
        alignment='left',
    )
    # plt.tight_layout()

    if figures_savedir is not None:
        figure_name = f'{performance_metric:s}.pdf'
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        plt.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _get_offsets(n_models: int):
    """
    Offsets to separate models from each other.
    """
    match n_models:
        case 4 | 5:
            offsets = np.linspace(-0.25, 0.25, n_models)
        case 7:
            offsets = np.linspace(-0.3, 0.3, n_models)
        case _:
            logging.error(f"Unexpected amount of models ({n_models:d}) found.")
            offsets = np.linspace(-0.2, 0.2, n_models)
    return offsets


def _process_df(
    config_dict: dict, 
    quant_results_df: pd.DataFrame,
    is_fig_s1: bool = False,
) -> pd.DataFrame:

    quant_results_df = _select_models_and_covariance_structures_to_plot(
        config_dict,
        quant_results_df,
        is_fig_s1=is_fig_s1,
    )

    # Update model labels for plot.
    # For the bivariate case, we make no distinction between joint or bivariate loop modelling.
    quant_results_df.index = quant_results_df.index.str.replace('SVWP_joint', 'WP')
    quant_results_df.index = quant_results_df.index.str.replace('SVWP', 'WP')
    quant_results_df.index = quant_results_df.index.str.replace('VWP', 'WP')
    quant_results_df.index = quant_results_df.index.str.replace('SW_cross_validated', 'SW-CV')

    match config_dict['data-set-name']:
        case 'd2':
            quant_results_df.index = quant_results_df.index.str.replace('_joint', '')
            quant_results_df.index = quant_results_df.index.str.replace('_bivariate_loop', '')
        case _:
            quant_results_df.index = quant_results_df.index.str.replace('_joint', '-J')
            quant_results_df.index = quant_results_df.index.str.replace('_bivariate_loop', '-BL')

    quant_results_df.index = quant_results_df.index.str.replace('_', '-')

    # Update covs types labels for plots.
    quant_results_df.columns = quant_results_df.columns.str.replace('periodic_1', 'periodic (slow)')
    quant_results_df.columns = quant_results_df.columns.str.replace('periodic_3', 'periodic (fast)')
    quant_results_df.columns = quant_results_df.columns.str.replace('_', ' ')

    return quant_results_df


def _select_models_and_covariance_structures_to_plot(
    config_dict: dict,
    df: pd.DataFrame,
    is_fig_s1: bool = False,
) -> pd.DataFrame:
    """
    Select models and covariance structures to plot.
    
    For the case of N=1200 we need to drop 'VWP' as well.

    Note that the range of models to plot is different in Fig. S1.
    """
    if is_fig_s1:
        models_to_plot = [
            'SW_cross_validated',
            'SW_15',
            'SW_30',
            'SW_60',
            'SW_120',
            'sFC',
        ]
    else:
        models_to_plot = config_dict['plot-models']

    return df.loc[models_to_plot, config_dict['plot-covs-types']]


if __name__ == "__main__":

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # e.g. 'N0200_T0100'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )

    for noise_type in cfg['noise-types']:
        for perform_metric in cfg['performance-metrics']:

            print(f'\n\n> Noise type:         {noise_type:s}')
            print(f'> Performance metric: {perform_metric:s}\n\n')

            plot_quantitative_results(
                config_dict=cfg,
                performance_metric=perform_metric,
                noise_type=noise_type,
                data_split=data_split,
                figures_savedir=os.path.join(
                    cfg['figures-basedir'], noise_type, data_split
                )
            )
