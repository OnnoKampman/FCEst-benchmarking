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


def plot_brain_state_switch_count(
    config_dict: dict,
    n_basis_states: int,
    connectivity_metric: str = 'correlation',
    figure_savedir: str = None,
) -> None:
    """
    The number of brain state switches is closely related to the idea of "dwell times".

    Parameters
    ----------
    :param config_dict:
    :param n_basis_states:
    :param connectivity_metric:
        'correlation', 'covariance'
    :param figure_savedir:
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    n_brain_state_switches_savedir = os.path.join(
        config_dict['git-results-basedir'], 'brain_states', f'k{n_basis_states:02d}'
    )

    all_brain_state_switch_counts_df = pd.DataFrame()
    for model_name in config_dict['models-brain-state-analysis']:
        n_brain_state_switches_filename = f'number_of_brain_state_switches_{model_name:s}.csv'
        brain_state_switch_counts_df = pd.read_csv(
            os.path.join(n_brain_state_switches_savedir, n_brain_state_switches_filename),
            index_col=0
        )  # (n_subjects, n_scans)

        # Shorten model names for plot.
        model_name = model_name.replace('SVWP_joint', 'WP')
        model_name = model_name.replace('DCC_joint', 'DCC-J')
        model_name = model_name.replace('DCC_bivariate_loop', 'DCC-BL')
        model_name = model_name.replace('SW_cross_validated', 'SW-CV')
        model_name = model_name.replace('_', '-')

        all_brain_state_switch_counts_df[model_name] = brain_state_switch_counts_df.values.flatten()

    fig, ax = plt.subplots(
        # figsize=config_dict['plot-brain-state-switch-count-figsize']
        figsize=(6.9, 3.5)
    )
    pt.RainCloud(
        data=all_brain_state_switch_counts_df,
        ax=ax,
        palette=get_palette(config_dict['models-brain-state-analysis']),
        bw=0.2,  # sets the smoothness of the distribution
        width_viol=0.6,
        orient="h",  # "v" if you want a vertical plot
        move=0.22,
        box_fliersize=2,
        box_linewidth=0.5,
        point_linewidth=0.5,
        linewidth=0.5,
        point_size=1.5,
    )
    plt.xlim([-1, 51])
    plt.xlabel('number of brain state switches')
    plt.ylabel('TVFC estimator')
#    plt.tight_layout()

    if figure_savedir is not None:
        figure_name = 'brain_state_switch_count.pdf'
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

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_brain_states_list = cfg['n-brain-states-list']

    for n_brain_states in n_brain_states_list:
        plot_brain_state_switch_count(
            config_dict=cfg,
            n_basis_states=n_brain_states,
            figure_savedir=os.path.join(
                cfg['figures-basedir'], 'brain_states', f'k{n_brain_states:02d}'
            )
        )
