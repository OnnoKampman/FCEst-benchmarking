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


def plot_kernel_lengthscales(
    config_dict: dict,
    model_name: str,
    kernel_lengthscales_df: pd.DataFrame,
    figure_savedir: str = None,
) -> None:
    """
    Generates raincloud plot with distribution of learned kernel lengthscales for each synthetic covariance structure.
    """
    sns.set_style("whitegrid")
    # sns.set_context(rc={"grid.linewidth": 0.4})
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, ax = plt.subplots(
        figsize=set_size(),
    )
    pt.RainCloud(
        data=kernel_lengthscales_df,
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
    ax.set_xlabel('learned kernel lengthscale')
    ax.set_ylabel('covariance structure')
    ax.set_xlim(left=0.0)

    if figure_savedir is not None:
        figure_name = f'{model_name:s}_kernel_lengthscales.pdf'
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

    kernel_param = 'kernel_lengthscales'

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )

    for model_name in ['VWP', 'SVWP']:
        for noise_type in cfg['noise-types']:
            kernel_lengthscales_df_filepath = os.path.join(
                cfg['git-results-basedir'], noise_type, data_split, f'{model_name:s}_{kernel_param:s}_kernel_params.csv'
            )
            if not os.path.exists(kernel_lengthscales_df_filepath):
                continue
            kernel_lengthscales_df = pd.read_csv(
                kernel_lengthscales_df_filepath,
                index_col=0
            )
            kernel_lengthscales_df = kernel_lengthscales_df.loc[:, cfg['plot-covs-types']]

            # Update covs types labels for plots.
            kernel_lengthscales_df.columns = kernel_lengthscales_df.columns.str.replace('periodic_1', 'periodic (slow)')
            kernel_lengthscales_df.columns = kernel_lengthscales_df.columns.str.replace('periodic_3', 'periodic (fast)')
            kernel_lengthscales_df.columns = kernel_lengthscales_df.columns.str.replace('_', ' ')

            plot_kernel_lengthscales(
                config_dict=cfg,
                model_name=model_name,
                kernel_lengthscales_df=kernel_lengthscales_df,
                figure_savedir=os.path.join(cfg['figures-basedir'], noise_type, data_split)
            )
