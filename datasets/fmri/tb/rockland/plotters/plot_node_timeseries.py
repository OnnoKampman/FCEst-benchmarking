import logging
import os
import socket

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import normalize_array
from helpers.figures import set_size
from helpers.rockland import get_rockland_subjects, get_convolved_stim_array


def _plot_node_timeseries(
        config_dict: dict, ts_df: pd.DataFrame, convolved_stim_array: np.array,
        mean_estimate: bool = False, subject_name: str = None,
        figures_savedir: str = None
) -> None:
    """
    Plots the BOLD signal time series per brain region of interest.

    TODO: add option to plot x-axis as minutes

    Parameters
    ----------
    :param config_dict:
    :param subject_ts_df:
    :param subject_name:
    """
    # sns.set(
    #     # style="whitegrid", 
    #     font_scale=0.05
    # )
    sns.set_style(
        "whitegrid",
        {
            'grid.linestyle': '--'
        },
        # font_scale=1.0
    )
    sns.set_context(rc={"grid.linewidth": 0.4})
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    linewidth = 1.6

    fig, axes = plt.subplots(
        # figsize=(5, 3),
        figsize=set_size(),
        nrows=ts_df.shape[1], ncols=1,
        sharex='all', sharey=True
    )

    for i_roi, region_of_interest in enumerate(ts_df.columns):
        axes[i_roi].plot(
            ts_df[region_of_interest], 
            label='BOLD time series', 
            linewidth=linewidth
        )
        axes[i_roi].plot(
            convolved_stim_array,
            label='Stimulus ' + r'$ \ast $' + ' HRF',
            linewidth=linewidth
        )
        axes[i_roi].set_ylim(config_dict['plot-time-series-ylim'])
        axes[i_roi].set_yticklabels([])
        axes[i_roi].set_ylabel(
            f'{region_of_interest:s}', rotation='horizontal', va='center', labelpad=15
        )

        for axis in ['top', 'bottom', 'left', 'right']:
            axes[i_roi].spines[axis].set_linewidth(0.4)

    axes[-1].set_xlim([0, 240])
    axes[-1].set_xticks([0, 40, 80, 120, 160, 200, 240])
    axes[-1].set_xlabel('time [scan volume]')

    axes[0].legend(bbox_to_anchor=(1.0, 1.0))

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.2)

    if figures_savedir is not None:
        if mean_estimate:
            figure_name = "mean_over_subjects.pdf"
        else:
            figure_name = f"{subject_name:s}.pdf"
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


if __name__ == '__main__':

    pp_pipeline = 'custom_fsl_pipeline'
    roi_list_name = 'final'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )

    figures_savedir = os.path.join(cfg['figures-basedir'], pp_pipeline, 'node_timeseries', roi_list_name)
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    # Get stimulus time series (convolved with HRF) and normalize for better interpretation.
    convolved_stim_array = get_convolved_stim_array(config_dict=cfg)
    convolved_stim_array = normalize_array(convolved_stim_array, verbose=False)

    all_subjects_list = get_rockland_subjects(config_dict=cfg)

    for i_subject, subject in enumerate(all_subjects_list):
        print(f'\n> Subject {i_subject+1:d} / {len(all_subjects_list):d}: {subject:s}\n')

        subject_timeseries_df = pd.read_csv(
            os.path.join(cfg['data-basedir'], pp_pipeline, 'node_timeseries', roi_list_name, subject)
        )  # (N, D)

        # _plot_node_timeseries(
        #     config_dict=cfg,
        #     subject_ts_df=subject_timeseries_df,
        #     subject_name=subject.removesuffix('.csv')
        # )

        if i_subject == 0:
            mean_over_subjects_timeseries_df = subject_timeseries_df.copy()
        else:
            mean_over_subjects_timeseries_df += subject_timeseries_df.values

    mean_over_subjects_timeseries_df /= len(all_subjects_list)
    print(mean_over_subjects_timeseries_df)

    _plot_node_timeseries(
        config_dict=cfg,
        ts_df=mean_over_subjects_timeseries_df,
        convolved_stim_array=convolved_stim_array,
        mean_estimate=True,
        figures_savedir=figures_savedir
    )
