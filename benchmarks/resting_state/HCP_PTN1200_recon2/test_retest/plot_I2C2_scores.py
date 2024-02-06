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
from helpers.figures import get_palette


def _plot_i2c2_bar(
        config_dict: dict, metric: str,
        figures_savedir: str = None
) -> None:
    sns.set(style="whitegrid", font_scale=1.0)
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    all_i2c2_scores_df = pd.DataFrame()
    for model_name in config_dict['plot-models']:
        i2c2_scores_filepath = os.path.join(
            config_dict['git-results-basedir'], 'test_retest', metric,
            f'I2C2_{model_name:s}_scores.csv'
        )
        if os.path.exists(i2c2_scores_filepath):
            i2c2_scores_df = pd.read_csv(
                i2c2_scores_filepath,
                index_col=0
            )
            all_i2c2_scores_df = all_i2c2_scores_df.append(i2c2_scores_df)

    all_i2c2_scores_df = _clean_up_model_names(all_i2c2_scores_df)
    print(all_i2c2_scores_df)

    # Select summary measures to plot.
    all_i2c2_scores_df = all_i2c2_scores_df[config_dict['plot-TVFC-summary-measures']]
    all_i2c2_scores_df.columns = all_i2c2_scores_df.columns.str.replace('_', '-')

    all_i2c2_scores_df.T.plot.bar(
        figsize=(6, 2),
        color={
            "WP": get_palette(config_dict['plot-models'])[0],
            "DCC-J": get_palette(config_dict['plot-models'])[1],
            "SW-CV": get_palette(config_dict['plot-models'])[2],
            "sFC": get_palette(config_dict['plot-models'])[3],
        },
    )
    # plt.xticks(rotation=35, ha="right")
    plt.xticks(rotation=0)
    plt.xlabel('TVFC summary measure')
    plt.ylabel('I2C2 test-retest score')
    plt.ylim([0.0, 0.55])
    plt.legend(
        bbox_to_anchor=(1.01, 1.0), frameon=True,
        title='TVFC\nestimator', title_fontsize=8, alignment='left'
    )

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_name = f'{metric:s}_I2C2_scores.pdf'
        plt.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _clean_up_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change model names for plot.
    """
    df.index = df.index.str.replace('SVWP_joint', 'WP')
    df.index = df.index.str.replace('DCC_joint', 'DCC-J')
    df.index = df.index.str.replace('DCC_bivariate_loop', 'DCC-BL')
    df.index = df.index.str.replace('SW_cross_validated', 'SW-CV')
    df.index = df.index.str.replace('_', '-')
    return df


if __name__ == "__main__":

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    metric = sys.argv[2]               # 'covariance', 'correlation'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    _plot_i2c2_bar(
        config_dict=cfg,
        metric=metric,
        figures_savedir=os.path.join(
            cfg['figures-basedir'], 'test_retest', 'I2C2'
        )
    )
