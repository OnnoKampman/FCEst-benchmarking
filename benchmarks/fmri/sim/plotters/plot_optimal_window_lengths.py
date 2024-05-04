import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import set_size


def plot_optimal_window_lengths(
    config_dict: dict,
    optimal_window_lengths_df: pd.DataFrame,
    figure_savedir: str = None,
) -> None:
    """
    Generates raincloud plot with distribution of optimal window lengths for each synthetic covariance structure.
    """
    sns.set_style("whitegrid")
    # sns.set_context(rc={"grid.linewidth": 0.4})
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, ax = plt.subplots(
        figsize=set_size(),
    )
    pt.RainCloud(
        data=optimal_window_lengths_df,
        ax=ax,
        palette=config_dict['plot-covs-types-palette'],
        bw=0.2,  # sets the smoothness of the distribution
        linewidth=0.5,
        box_linewidth=0.5,
        point_size=1.5,
        box_fliersize=2.0,
        box_whiskerprops={
            'linewidth': 0.5,
            "zorder": 10,
        },
        width_viol=0.6,
        orient="h",  # "v" if you want a vertical plot
        move=0.22,
    )
    ax.set_xlabel('optimal window length [TRs]')
    ax.set_ylabel('covariance structure')
    ax.set_xlim(left=0.0)

    if figure_savedir is not None:
        figure_name = 'SW_cross_validated_optimal_window_lengths.pdf'
        if not os.path.exists(figure_savedir):
            os.makedirs(figure_savedir)
        fig.savefig(
            os.path.join(figure_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figure_savedir:s}'.")
        plt.close()


if __name__ == "__main__":

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )

    for noise_type in cfg['noise-types']:
        optimal_window_lengths_df = pd.read_csv(
            os.path.join(
                cfg['git-results-basedir'], noise_type, data_split, 'optimal_window_lengths.csv'
            ),
            index_col=0
        )
        optimal_window_lengths_df = optimal_window_lengths_df.loc[:, cfg['plot-covs-types']]

        # Update covs types labels for plots.
        optimal_window_lengths_df.columns = optimal_window_lengths_df.columns.str.replace('periodic_1', 'periodic (slow)')
        optimal_window_lengths_df.columns = optimal_window_lengths_df.columns.str.replace('periodic_3', 'periodic (fast)')
        optimal_window_lengths_df.columns = optimal_window_lengths_df.columns.str.replace('checkerboard', 'boxcar')
        optimal_window_lengths_df.columns = optimal_window_lengths_df.columns.str.replace('_', ' ')

        plot_optimal_window_lengths(
            config_dict=cfg,
            optimal_window_lengths_df=optimal_window_lengths_df,
            figure_savedir=os.path.join(cfg['figures-basedir'], noise_type, data_split)
        )
