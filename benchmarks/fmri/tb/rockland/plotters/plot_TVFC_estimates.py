import logging
import os
import socket
import sys

from fcest.helpers.data import to_3d_format
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import set_size
from helpers.plotters import convert_to_seconds, plot_method_tvfc_estimates
from helpers.rockland import get_rockland_subjects, get_convolved_stim_array, load_rockland_data


def _plot_average_over_subject_tvfc_estimates(
    config_dict: dict,
    data_split: str,
    metric: str,
    preprocessing_pipeline: str,
    model_to_plot_name: str,
    all_subjects_list: list,
    edges_to_plot_indices,
    column_names,
    figures_savedir: str = None,
) -> None:
    """
    Plots average TVFC estimates over all subjects for a single TVFC estimation method.
    """
    sns.set(style="whitegrid", font_scale=0.4)
    # sns.set_palette("colorblind")

    for plot_type in ['raw', 'detrended']:
        sns.set_palette('Set3')
        fig, ax = plt.subplots(
            figsize=set_size(fraction=0.47)
        )
        _plot_estimates(
            config_dict=config_dict,
            data_split=data_split,
            metric=metric,
            preprocessing_pipeline=preprocessing_pipeline,
            model_to_plot_name=model_to_plot_name,
            all_subjects_list=all_subjects_list,
            edges_to_plot_indices=edges_to_plot_indices,
            column_names=column_names,
            plot_type=plot_type,
            figures_savedir=figures_savedir,
            fig=fig,
            axes=ax,
        )


def plot_average_over_subject_tvfc_estimates_joint(
    config_dict: dict,
    data_split: str,
    preprocessing_pipeline: str,
    all_subjects_list: list,
    edges_to_plot_indices,
    column_names,
    metric: str = 'correlation',
    figures_savedir: str = None,
) -> None:
    """
    Plots TVFC estimates averaged over all subjects jointly for a collection of TVFC
    estimation methods.
    """
    sns.set(style="whitegrid", font_scale=1.2)

    models_to_plot = [
        'SVWP_joint',
        'DCC_joint',
        'SW_cross_validated',
        # 'sFC',
    ]

    for plot_type in ['raw', 'detrended']:
        sns.set_palette('Set3')

        if len(models_to_plot) == 3:
            fig, axes = plt.subplots(
                figsize=(12, 4),
                nrows=1, ncols=3,
                sharex=True, sharey=True
            )
            axes[0].set_ylabel('TVFC estimate')
        elif len(models_to_plot) == 4:
            fig, axes = plt.subplots(
                # figsize=set_size()
                nrows=2, ncols=2,
                sharex=True, sharey=True
            )
            # axes[0, 0].set_ylabel('TVFC estimate')
            # axes[1, 0].set_ylabel('TVFC estimate')

        for i_model_name, model_to_plot_name in enumerate(models_to_plot):
            _plot_estimates(
                config_dict=config_dict,
                data_split=data_split,
                metric=metric,
                preprocessing_pipeline=preprocessing_pipeline,
                model_to_plot_name=model_to_plot_name,
                all_subjects_list=all_subjects_list,
                edges_to_plot_indices=edges_to_plot_indices,
                column_names=column_names,
                plot_type=plot_type,
                figures_savedir=figures_savedir,
                fig=fig,
                axes=axes,
                models_to_plot=models_to_plot,
                i_model_name=i_model_name,
            )


def _plot_estimates(
    config_dict: dict,
    data_split: str,
    metric: str,
    preprocessing_pipeline: str,
    model_to_plot_name: str,
    all_subjects_list: list,
    edges_to_plot_indices,
    column_names,
    plot_type: str,
    figures_savedir: str,
    fig,
    axes,
    models_to_plot: list = None,
    i_model_name: int = None,
    plot_hrf: bool = False,
) -> None:
    """
    Plot raw (trended) or detrended TVFC estimates.
    """
    xx = _get_xx(
        config_dict=config_dict,
        all_subjects_list=all_subjects_list,
        pp_pipeline=preprocessing_pipeline
    )
    n_time_steps = len(xx)  # N
    n_time_series = len(config_dict['roi-list'])  # D

    average_tvfc_estimates = _compute_average_over_subjects_tvfc_estimates(
        config_dict=config_dict,
        n_time_steps=n_time_steps,
        n_time_series=n_time_series,
        data_split=data_split,
        metric=metric,
        pp_pipeline=preprocessing_pipeline,
        model_name=model_to_plot_name,
        all_subjects_list=all_subjects_list
    )

    for edge_to_plot_indices in edges_to_plot_indices:

        label_name = column_names[edge_to_plot_indices]
        label_name = '-'.join(label_name)

        edge_average_tvfc_estimates = average_tvfc_estimates[:, edge_to_plot_indices[0], edge_to_plot_indices[1]]  # (N, )

        if plot_type == 'detrended':
            edge_average_tvfc_estimates = scipy.signal.detrend(
                edge_average_tvfc_estimates,
                type='linear'
            )

        if models_to_plot is not None:
            if len(models_to_plot) == 3:
                i_plot = 0
                j_plot = i_model_name
                ax_to_plot = axes[j_plot]
                ax_to_plot_legend = axes[2]
            elif len(models_to_plot) == 4:
                i_plot = np.repeat(np.arange(2), 2)[i_model_name]
                j_plot = np.tile(np.arange(2), 2)[i_model_name]
                ax_to_plot = axes[i_plot, j_plot]
                ax_to_plot_legend = axes[0, 1]
        else:
            ax_to_plot = axes
            ax_to_plot_legend = axes

        match model_to_plot_name:
            case 'DCC_joint' | 'DCC_bivariate_loop' | 'SW_cross_validated' | 'SW_16' | 'SW_30' | 'SW_60':
                ax_to_plot.plot(
                    xx, edge_average_tvfc_estimates,
                    # 'x-',
                    linewidth=2.0,
                    # markersize=2.0,
                    label=label_name
                )
            case 'VWP_joint' | 'SVWP_joint' | 'sFC':
                ax_to_plot.plot(
                    xx, edge_average_tvfc_estimates,
                    linewidth=2.0,
                    label=label_name
                )
            case _:
                logging.warning(f"Model {model_to_plot_name:s} not recognized.")

    # Add stimulus presence.
    for stimuli_start in range(20, 140, 40):
        ax_to_plot.axvspan(
            stimuli_start, stimuli_start + 20,
            facecolor='b', alpha=0.1
        )

    # Add stimulus convolved with HRF.
    if plot_hrf:
        stim_hrf_array = get_convolved_stim_array(config_dict=config_dict)
        plt.plot(
            xx, stim_hrf_array, color='black', linewidth=3, label='Stimulus HRF'
        )

    ax_to_plot.grid(linestyle='dashed', linewidth=0.4)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax_to_plot.spines[axis].set_linewidth(0.4)

    ax_to_plot.set_xlim([0, 144.5])
    ax_to_plot.set_xticks([0, 20, 40, 60, 80, 100, 120, 140])
    if plot_type == 'raw':
        ax_to_plot.set_ylim([-0.15, 1.05])
    if plot_type == 'detrended':
        ax_to_plot.set_ylim([-0.36, 0.36])

    if models_to_plot is not None:
        ax_to_plot.set_xlabel('time [seconds]')
        # ax_to_plot.set_ylabel('TVFC estimate')
        ax_to_plot.set_title(
            model_to_plot_name.replace('SVWP_joint', 'WP').replace('_joint', '-J').replace('_cross_validated', '-CV')
        )
    else:
        ax_to_plot.set_xlabel('time [seconds]')
        ax_to_plot.set_ylabel('TVFC estimate')

        bbox_to_anchor = (1.61, 1.0)  # used to put legend outside of plot
        ax_to_plot_legend.legend(
            loc='upper right', bbox_to_anchor=bbox_to_anchor, frameon=True,
            title='ROI edge', alignment='left'
        )

    plt.tight_layout()

    if figures_savedir is not None:
        if models_to_plot is not None:
            if plot_type == 'raw':
                figure_name = f'all_subjects_joint_{metric:s}s.pdf'
            if plot_type == 'detrended':
                figure_name = f'all_subjects_joint_{metric:s}s_detrended.pdf'
        else:
            if plot_type == 'raw':
                figure_name = f'all_subjects_{model_to_plot_name:s}_{metric:s}s.pdf'
            if plot_type == 'detrended':
                figure_name = f'all_subjects_{model_to_plot_name:s}_{metric:s}s_detrended.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_individual_subjects(
    config_dict: dict,
    edges_to_plot_indices,
    data_split: str,
    metric: str,
    pp_pipeline: str,
    all_subjects_list:list,
    brain_regions_of_interest, 
    model_name: str,
    figures_base_savedir: str,
) -> None:
    """
    Plot TVFC estimates for each subject individually.
    """
    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'\n> Subject {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}\n')
        data_file = os.path.join(
            config_dict['data-basedir'], pp_pipeline, 'node_timeseries',
            config_dict['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        x_train = x  # (N, 1)
        y_train = y  # (N, D)
        n_time_steps = x_train.shape[0]
        n_time_series = y_train.shape[1]

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )
        xx = convert_to_seconds(
            x_train,
            repetition_time=config_dict['repetition-time'],
            data_length=n_time_steps
        )
        for edge_to_plot_indices in edges_to_plot_indices:

            label_name = brain_regions_of_interest[edge_to_plot_indices]
            label_name = '-'.join(label_name)

            sns.set(style="whitegrid", font_scale=1.4)
            # plt.rcParams["font.family"] = 'serif'
            plot_method_tvfc_estimates(
                config_dict=config_dict,
                model_name=model_name,
                i_time_series=edge_to_plot_indices[0],
                j_time_series=edge_to_plot_indices[1],
                x_train_locations=x_train,
                y_train_locations=y_train,                
                subject_name=subject_filename,
                data_split=data_split,
                metric=metric,
                pp_pipeline=pp_pipeline,
                label_name=label_name
            )

        # Add stimulus presence.
        for stimuli_start in range(20, 140, 40):
            plt.axvspan(
                stimuli_start, stimuli_start + 20, facecolor='b', alpha=0.1
            )

        # Add stimulus convolved with HRF.
        # stim_hrf_array = get_convolved_stim_array(config_dict=cfg)
        # plt.plot(xx, stim_hrf_array, color='black', linewidth=3, label='Stimulus HRF')

        bbox_to_anchor = (1.14, 1.0)  # used to put legend outside of plot
        plt.xlim([0.0, 144.5])
        # plt.ylim([-0.5, 1.1])
        plt.legend(
            loc='upper right', bbox_to_anchor=bbox_to_anchor, frameon=True
        )
        plt.ylabel(f'{metric:s} estimate')
        plt.xlabel('time [seconds]')
        # plt.tight_layout()

        figures_savedir_subject = os.path.join(figures_base_savedir, subject_filename[:-4])
        if not os.path.exists(figures_savedir_subject):
            os.makedirs(figures_savedir_subject)

        if figures_savedir_subject is not None:
            figure_name = f'{model_name:s}_{metric:s}s.pdf'
            plt.savefig(
                os.path.join(figures_savedir_subject, figure_name),
                format='pdf',
                bbox_inches='tight'
            )
            logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir_subject:s}'.")
            plt.close()


def _compute_average_over_subjects_tvfc_estimates(
    config_dict: dict,
    n_time_steps: int,
    n_time_series: int,
    data_split: str,
    metric: str,
    pp_pipeline: str,
    model_name: str,
    all_subjects_list: list,
) -> np.array:
    """
    Compute average of TVFC estimates across all subjects.
    """
    average_tvfc_estimates = np.zeros(shape=(n_time_steps, n_time_series, n_time_series))

    tvfc_estimates_savedir = os.path.join(
        config_dict['experiments-basedir'], pp_pipeline, 'TVFC_estimates', data_split, metric, model_name
    )

    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'> Subject {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}')
        estimated_tvfc_df = pd.read_csv(
            os.path.join(tvfc_estimates_savedir, subject_filename),
            index_col=0
        )  # (D*D, N)
        estimated_tvfc = to_3d_format(estimated_tvfc_df.values)  # (N, D, D)
        assert estimated_tvfc.shape == (n_time_steps, n_time_series, n_time_series)
        average_tvfc_estimates += estimated_tvfc
    average_tvfc_estimates /= len(all_subjects_list)  # (N, D, D)

    return average_tvfc_estimates


def _get_xx(
        config_dict: dict, all_subjects_list: list, pp_pipeline: str
) -> np.array:

    example_subject_filename = all_subjects_list[0]
    example_data_file = os.path.join(
        config_dict['data-basedir'], pp_pipeline, 'node_timeseries',
        config_dict['roi-list-name'], example_subject_filename
    )
    x, _ = load_rockland_data(example_data_file)  # (N, 1), (N, D)
    x_train = x  # (N, 1)

    return convert_to_seconds(
        x_train,
        repetition_time=config_dict['repetition-time'],
        data_length=x_train.shape[0]
    )


if __name__ == "__main__":

    data_split = 'all'
    metric = 'correlation'
    pp_pipeline = 'custom_fsl_pipeline'

    model_name = sys.argv[1]       # 'VWP_joint', 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_16', 'SW_30', 'SW_60', or 'sFC'
    repetition_time = sys.argv[2]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    brain_regions_of_interest = cfg['roi-list']
    edges_to_plot_indices = cfg['roi-edges-list']
    figures_base_savedir = os.path.join(
        cfg['figures-basedir'], pp_pipeline, 'TVFC_estimates', cfg['roi-list-name'], data_split
    )
    if not os.path.exists(figures_base_savedir):
        os.makedirs(figures_base_savedir)

    # _plot_average_over_subject_tvfc_estimates(
    #     config_dict=cfg,
    #     data_split=data_split,
    #     metric=metric,
    #     preprocessing_pipeline=pp_pipeline,
    #     model_to_plot_name=model_name,
    #     all_subjects_list=all_subjects_list,
    #     edges_to_plot_indices=edges_to_plot_indices,
    #     column_names=brain_regions_of_interest,
    #     figures_savedir=figures_base_savedir
    # )
    plot_average_over_subject_tvfc_estimates_joint(
        config_dict=cfg,
        data_split=data_split,
        metric=metric,
        preprocessing_pipeline=pp_pipeline,
        all_subjects_list=all_subjects_list,
        edges_to_plot_indices=edges_to_plot_indices,
        column_names=brain_regions_of_interest,
        figures_savedir=figures_base_savedir
    )
    # _plot_individual_subjects(
    #     config_dict=cfg,
    #     data_split=data_split,
    #     metric=metric,
    #     all_subjects_list=all_subjects_list,
    # )
