import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import get_palette

sns.set(style="whitegrid", font_scale=1.5)
plt.rcParams["font.family"] = 'serif'


def _plot_bar_graph_scores(config_dict: dict) -> None:
    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])

    sns.barplot(
        # x="group",
        # y="score",
        data=all_likelihoods_df,
        capsize=.1
    )
    if experiment_dimensionality == 'bivariate':
        plt.ylim([-4.5, -2.3])
    # else:
    #     plt.ylim([-25.4, -13.8])
    plt.ylabel('test log likelihood', fontsize=16)
    # plt.tight_layout()

    figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_bar.pdf'
    savedir = os.path.join(cfg['figures-basedir'], 'imputation_study')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(
        os.path.join(savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{savedir:s}'.")


def _plot_cloud(config_dict: dict) -> None:
    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])

    ax = pt.half_violinplot(
        # x=dx,
        # y=dy,
        data=all_likelihoods_df,
        # palette=pal,
        bw=.2,
        cut=0.,
        scale="area",
        width=.6,
        inner=None,
        orient="h"
    )
    if experiment_dimensionality == 'bivariate':
        plt.xlim([-4.5, -2.3])
    else:
        plt.xlim([-25.4, -13.8])
    plt.xlabel('test log likelihood', fontsize=16)
    # plt.tight_layout()

    figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_cloud.pdf'
    savedir = os.path.join(cfg['figures-basedir'], 'imputation_study')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(
        os.path.join(savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{savedir:s}'.")


def _plot_violin(config_dict: dict) -> None:
    """
    Show each distribution with both violins and points.
    """
    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])
    sns.violinplot(
        data=all_likelihoods_df,
        # palette="light:g",
        inner="points",  # 'points', 'stick'
        orient="v",
        cut=2,
        scale_hue=False,
        linewidth=0.8,
        bw=.2
    )
    if experiment_dimensionality == 'bivariate':
        plt.ylim([-4.5, -2.4])
    else:
        plt.ylim([-25.4, -13.8])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=13)
    plt.ylabel('test log likelihood', fontsize=16)
    # plt.tight_layout()

    figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_violin.pdf'
    savedir = os.path.join(cfg['figures-basedir'], 'imputation_study')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(
        os.path.join(savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{savedir:s}'.")


def _plot_likelihoods_raincloud(
    config_dict: dict,
    all_likelihoods_df: pd.DataFrame,
    data_dimensionality: str,
    experiment_dimensionality: str,
    data_split: str = 'LEOO',
    figure_savedir: str = None,
) -> None:
    """
    A "cloud", or smoothed version of a histogram, gives an idea of the distribution of scores.
    The "rain" are the individual data points, which can give an idea of outliers.
    Source: 
        https://github.com/RainCloudPlots/RainCloudPlots
    TODO: figure out the right way of setting fonts and ticks on axes
    """

    plt.rcParams["font.family"] = 'sans-serif'

    # plt.figure(figsize=config_dict['plot-likelihoods-figsize'])

    # The raincloud plot consists of 3 separate plots of the data.
    # ax = pt.half_violinplot(
    #     data=d,
        # bw=.2,
        # cut=0.,
        # scale="area",
        # width=.6,
        # inner=None,
        # orient="h"
    # )
    # ax = sns.stripplot(
        # data=d,
        # edgecolor="white",
        # size=3,
        # jitter=1,  # set jitter to 0 to put all data points on a line
        # zorder=0,
        # orient="h"
    # )
    # Add an "empty" boxplot to show median, quartiles, and outliers.
    # ax = sns.boxplot(
    #     data=d,
    #     color="black",
    #     width=.15,
    #     zorder=10,
    #     showcaps=True,
    #     boxprops={'facecolor':'none', "zorder":10},
    #     showfliers=True,
    #     whiskerprops={'linewidth':2, "zorder":10},
    #     saturation=1,
    #     orient="h"
    # )
    # This can also be summarized into a single call like below.
    fig, ax = plt.subplots(
        figsize=config_dict['plot-likelihoods-figsize']
    )

    pt.RainCloud(
        data=all_likelihoods_df,
        ax=ax,
        palette=get_palette(all_likelihoods_df.columns),
        bw=0.2,  # sets the smoothness of the distribution
        width_viol=0.6,
        orient="h",  # "v" if you want a vertical plot
        move=0.22
    )

    if data_dimensionality == 'd15':
        if experiment_dimensionality == 'bivariate':
            plt.xlim([-4.5, -2.4])
        else:
            plt.xlim([-25.9, -12.1])

    plt.xlabel('test log likelihood', fontsize=18)
    plt.ylabel('TVFC estimator', fontsize=18)
    # plt.tight_layout()

    if figure_savedir is not None:
        figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_raincloud.pdf'
        if not os.path.exists(figure_savedir):
            os.makedirs(figure_savedir)
        fig.savefig(
            os.path.join(figure_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figure_savedir:s}'.")
        plt.close()


if __name__ == '__main__':

    data_split = 'LEOO'  # leave-every-other-out

    data_dimensionality = sys.argv[1]        # 'd15', 'd50'
    experiment_dimensionality = sys.argv[2]  # 'multivariate', 'bivariate'  # TODO: remove this?

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    all_likelihoods_df = pd.DataFrame()
    for model_name in cfg['plot-models']:
        likelihoods_filename = f'LEOO_{experiment_dimensionality:s}_likelihoods_{model_name:s}.csv'
        test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
        test_likelihoods_filepath = os.path.join(test_likelihoods_savedir, likelihoods_filename)
        if os.path.exists(test_likelihoods_filepath):
            likelihoods_df = pd.read_csv(
                test_likelihoods_filepath, index_col=0
            )  # (n_subjects, n_scans)
            likelihoods_array = likelihoods_df.values.reshape(-1)  # (n_subjects * n_scans, 1)

            model_name = model_name.replace('SVWP_joint', 'WP')
            model_name = model_name.replace('DCC_joint', 'DCC-J')
            model_name = model_name.replace('DCC_bivariate_loop', 'DCC-BL')
            model_name = model_name.replace('SW_cross_validated', 'SW-CV')
            model_name = model_name.replace('_', '-')

            all_likelihoods_df[model_name] = likelihoods_array

    print(all_likelihoods_df.head())
    _plot_bar_graph_scores(config_dict=cfg)
    _plot_cloud(config_dict=cfg)
    _plot_violin(config_dict=cfg)
    _plot_likelihoods_raincloud(
        config_dict=cfg, 
        all_likelihoods_df=all_likelihoods_df,
        data_dimensionality=data_dimensionality,
        experiment_dimensionality=experiment_dimensionality,
        figure_savedir=os.path.join(cfg['figures-basedir'], 'imputation_study')
    )
