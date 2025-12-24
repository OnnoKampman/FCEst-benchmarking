import logging
import os
import socket

import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import get_palette


def save_bar_plot(config_dict: dict) -> None:
    sns.set(style="whitegrid", font_scale=1.5)
    plt.rcParams["font.family"] = 'serif'

    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])

    sns.barplot(
        # x="group",
        # y="score",
        data=all_test_likelihoods_df,
        capsize=.1
    )
    plt.ylabel('test log likelihood')
    # plt.tight_layout()

    figure_name = f'{data_split:s}_test_log_likelihoods_bar.pdf'
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")


def save_cloud_plot(config_dict: dict) -> None:
    sns.set(style="whitegrid", font_scale=1.5)
    plt.rcParams["font.family"] = 'serif'

    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])

    ax = pt.half_violinplot(
        # x=dx,
        # y=dy,
        data=all_test_likelihoods_df,
        # palette=pal,
        bw=.2,
        cut=0.,
        scale="area",
        width=.6,
        inner=None,
        orient="h"
    )
    plt.xlabel('test log likelihood')
    # plt.tight_layout()
    figure_name = f'{data_split:s}_test_log_likelihoods_cloud.pdf'
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")


def save_violin_plot(config_dict: dict) -> None:
    """
    Show each distribution with both violins and points.
    """
    sns.set(style="whitegrid", font_scale=1.5)
    plt.rcParams["font.family"] = 'serif'

    plt.figure(figsize=config_dict['plot-likelihoods-figsize'])
    sns.violinplot(
        data=all_test_likelihoods_df,
        # palette="light:g",
        inner="points",  # 'points', 'stick'
        orient="v",
        cut=2,
        scale_hue=False,
        linewidth=0.8,
        bw=.2
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=13)
    plt.ylabel('test log likelihood')
    # plt.tight_layout()
    figure_name = f'{data_split:s}_test_log_likelihoods_violin.pdf'
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")


def plot_likelihoods_raincloud(
        config_dict: dict, all_test_likelihoods_df: pd.DataFrame,
        data_split: str = 'LEOO', figures_savedir: str = None
) -> None:
    """
    A "cloud", or smoothed version of a histogram, gives an idea of the distribution of scores.
    The "rain" are the individual data points, which can give an idea of outliers.

    Source:
        https://github.com/RainCloudPlots/RainCloudPlots
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, ax = plt.subplots(
        figsize=config_dict['fig-figsize-likelihoods-raincloud'],
    )
    pt.RainCloud(
        data=all_test_likelihoods_df,
        ax=ax,
        palette=get_palette(all_test_likelihoods_df.columns),
        bw=0.2,  # sets the smoothness of the distribution
        width_viol=0.6,
        orient="h",  # "v" if you want a vertical plot
        move=0.22,
    )

    plt.xlim([-15, 1])
    plt.xlabel('test log likelihood')
    plt.ylabel('TVFC estimator')

    # plt.tight_layout()

    if figures_savedir is not None:
        figure_name = f'{data_split:s}_test_log_likelihoods_raincloud.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


if __name__ == '__main__':

    pp_pipeline = 'custom_fsl_pipeline'
    data_split = 'LEOO'  # leave-every-other-out

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )

    figures_savedir = os.path.join(cfg['figures-basedir'], pp_pipeline, 'imputation_study')
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    all_test_likelihoods_df = pd.DataFrame()
    for model_name in cfg['plot-likelihoods-models']:
        likelihoods_filename = f'{data_split:s}_likelihoods_{model_name:s}.csv'
        test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_benchmark')
        likelihoods_df = pd.read_csv(
            os.path.join(test_likelihoods_savedir, likelihoods_filename),
            index_col=0
        )  # (n_subjects, 1)
        likelihoods_array = likelihoods_df.values.flatten()  # (n_subjects, )

        model_name = model_name.replace('SVWP_joint', 'WP')
        model_name = model_name.replace('_joint', '-J')
        model_name = model_name.replace('DCC_bivariate_loop', 'DCC-BL')
        model_name = model_name.replace('SW_cross_validated', 'SW-CV')
        model_name = model_name.replace('_', '-')

        all_test_likelihoods_df[model_name] = likelihoods_array

    print('\nPlotting for the following TVFC estimation methods...')
    print(all_test_likelihoods_df.head())

    save_bar_plot(config_dict=cfg)
    save_cloud_plot(config_dict=cfg)
    save_violin_plot(config_dict=cfg)
    plot_likelihoods_raincloud(
        config_dict=cfg,
        all_test_likelihoods_df=all_test_likelihoods_df,
        figures_savedir=figures_savedir
    )
