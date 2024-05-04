import copy
import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects, rename_variables_for_plots
from helpers.figures import get_palette, set_size


def plot_out_of_sample_prediction_scores_joint(
    config_dict: dict,
    scores_savedir: str,
    subject_measures_list: list[str],
    tvfc_estimation_methods: list[str],
    connectivity_metric: str = 'correlation',
    performance_metric: str = 'r2_scores',
    figures_savedir: str = None,
) -> None:
    """
    Plot all TVFC summary measures jointly in a single figure.

    Parameters
    ----------
    :param config_dict:
        Configuration dictionary.
    :param performance_metric:
        'r2_scores', 'variance_explained', or 'prediction_accuracy'.
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, axes = plt.subplots(
        # figsize=set_size(subplots=(3, 1)),
        figsize=(9.6, 8),
        # figsize=config_dict['plot-subject-measures-figsize'],
        nrows=len(config_dict['plot-TVFC-summary-measures']),
        ncols=1,
        sharex=True,
    )

    for i_tvfc_summary_measure, tvfc_summary_measure in enumerate(config_dict['plot-TVFC-summary-measures']):

        # Gather results.
        all_results = pd.DataFrame()
        for tvfc_estimation_method in tvfc_estimation_methods:

            if tvfc_estimation_method == 'sFC' and tvfc_summary_measure != 'mean':
                logging.info("Not including sFC here.")
                continue

            out_of_sample_prediction_results_df = pd.read_csv(
                os.path.join(
                    scores_savedir, f'{connectivity_metric:s}_linear_ridge_model_{performance_metric:s}_TVFC_{tvfc_summary_measure:s}_{tvfc_estimation_method:s}.csv'
                ),
                index_col=0
            )  # (n_permutations, n_subject_measures)

            out_of_sample_prediction_results_df = out_of_sample_prediction_results_df.loc[:, subject_measures_list]
            out_of_sample_prediction_results_df = rename_variables_for_plots(out_of_sample_prediction_results_df, axis=0)

            reformatted_single_method_results_df = pd.DataFrame()
            for sm in out_of_sample_prediction_results_df.columns:

                nnndf = pd.DataFrame()

                nnndf['permutation'] = out_of_sample_prediction_results_df.index
                nnndf['Subject measure'] = sm
                nnndf[performance_metric] = out_of_sample_prediction_results_df.loc[:, sm].values

                reformatted_single_method_results_df = pd.concat(
                    [reformatted_single_method_results_df, nnndf]
                )

            tvfc_estimation_method = tvfc_estimation_method.replace('SVWP_joint', 'WP')
            tvfc_estimation_method = tvfc_estimation_method.replace('DCC_joint', 'DCC-J')
            tvfc_estimation_method = tvfc_estimation_method.replace('DCC_bivariate_loop', 'DCC-BL')
            tvfc_estimation_method = tvfc_estimation_method.replace('SW_cross_validated', 'SW-CV')
            tvfc_estimation_method = tvfc_estimation_method.replace('_', '-')

            reformatted_single_method_results_df['TVFC estimator'] = tvfc_estimation_method

            all_results = pd.concat(
                [all_results, reformatted_single_method_results_df]
            )

        print(all_results)

        # Remove outliers for more interpretable plots.
        # TODO: this is based on a visual inspection, be careful with this
        if performance_metric == 'r2_scores':
            outlier_cutoff = -0.04
            all_results.loc[all_results.loc[:, performance_metric] < outlier_cutoff, performance_metric] = np.nan

        print(all_results)

        sns.violinplot(
            ax=axes[i_tvfc_summary_measure],
            data=all_results,
            x="Subject measure",
            y=performance_metric,
            hue="TVFC estimator",
            # bw_method="silverman",
            cut=0,
            fill=False,
            gridsize=100,
            inner='quart',
            linewidth=1.0,
            orient="v",
            palette=get_palette(config_dict['plot-models']),
            saturation=0.75,
            split=False,
        )

        axes[i_tvfc_summary_measure].set_ylabel(
            f"TVFC {tvfc_summary_measure.replace('_', '-'):s}\n{performance_metric.replace('r2_scores', 'R2 scores').replace('variance_explained', 'Variance explained').replace('prediction_accuracy', 'Prediction accuracy'):s}"
        )

        if performance_metric == 'r2_scores':
            axes[i_tvfc_summary_measure].set_ylim([-0.05, 0.18])
        if performance_metric == 'prediction_accuracy':
            axes[i_tvfc_summary_measure].set_ylim([-0.3, 0.7])

        if not i_tvfc_summary_measure + 1 == len(config_dict['plot-TVFC-summary-measures']):
            axes[i_tvfc_summary_measure].set_xticklabels([])

        # Leave the legend only in the first subplot.
        if not i_tvfc_summary_measure == 0:
            axes[i_tvfc_summary_measure].get_legend().remove()

        axes[i_tvfc_summary_measure].set(xlabel=None)
        axes[i_tvfc_summary_measure].grid(True)

    # axes[0].legend(
    #     bbox_to_anchor=(1.01, 1.0), frameon=True,
    #     title='TVFC\nestimator', alignment='left'
    # )
    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_name = f'out_of_sample_prediction_{performance_metric:s}_all_TVFC_summary_measures.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def plot_out_of_sample_prediction_scores(
    config_dict: dict,
    out_of_sample_prediction_results_df: pd.DataFrame,
    tvfc_summary_measure: str,
    figures_savedir: str = None,
) -> None:
    """
    Plot morphometricity results for single TVFC summary measure.
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, ax = plt.subplots(
        figsize=set_size()
    )

    print(out_of_sample_prediction_results_df)

    sns.violinplot(
        ax=ax,
        data=out_of_sample_prediction_results_df,
        # x="Age",
        # y="class",
        cut=0,
        fill=False,
        inner='quart',
        linewidth=1.0,
        gridsize=100,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
    )
    ax.set_ylabel('Prediction accuracy')

    # ax.legend(
    #     bbox_to_anchor=(1.21, 1.0), frameon=True,
    #     title='TVFC\nestimator', alignment='left'
    # )

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_name = f'out_of_sample_prediction_TVFC_{tvfc_summary_measure:s}.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _rename_models_for_plots(config_dict: dict, original_df: pd.DataFrame) -> pd.DataFrame:
    renamed_df = original_df.copy()

    # Select models to be included in plots.
    plot_models = copy.deepcopy(config_dict['plot-models'])
    if 'sFC' in plot_models and 'sFC' not in renamed_df.columns.tolist():
        print('Not including sFC estimates here.')
        plot_models.remove('sFC')
    renamed_df = renamed_df.loc[:, plot_models]

    renamed_df.columns = renamed_df.columns.str.replace('SVWP_joint', 'WP')
    renamed_df.columns = renamed_df.columns.str.replace('DCC_joint', 'DCC-J')
    renamed_df.columns = renamed_df.columns.str.replace('DCC_bivariate_loop', 'DCC-BL')
    renamed_df.columns = renamed_df.columns.str.replace('SW_cross_validated', 'SW-CV')
    renamed_df.columns = renamed_df.columns.str.replace('_', '-')
    return renamed_df


if __name__ == '__main__':

    metric = 'correlation'
    tvfc_estimation_method = 'SVWP_joint'

    data_dimensionality = sys.argv[1]      # 'd15', 'd50'
    subject_measures_subset = sys.argv[2]  # 'cognitive', 'other', 'personality', 'psychiatric', or 'social-emotional'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=n_subjects,
        as_ints=True
    )

    subject_measures_list = cfg[f"subject-measures-{subject_measures_subset:s}"]
    if subject_measures_subset == 'cognitive':
        subject_measures_list = cfg['subject-measures-nuisance-variables'] + subject_measures_list

    scores_savedir = os.path.join(
        cfg['git-results-basedir'], 'subject_measure_prediction',
        'out_of_sample_prediction', subject_measures_subset
    )

    plot_out_of_sample_prediction_scores_joint(
        cfg,
        scores_savedir,
        subject_measures_list,
        tvfc_estimation_method,
        figures_savedir=os.path.join(
            cfg['figures-basedir'], 'subject_measure_prediction', subject_measures_subset
        )
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:

        out_of_sample_prediction_results_df = pd.read_csv(
            os.path.join(scores_savedir, f'{metric:s}_linear_ridge_model_prediction_accuracy_TVFC_{tvfc_summary_measure:s}_{tvfc_estimation_method:s}.csv'),
            index_col=0
        )  # (n_permutations, n_subjects)
        print('results')
        print(out_of_sample_prediction_results_df)

        # Re-order subject measures.
        out_of_sample_prediction_results_df = out_of_sample_prediction_results_df.loc[:, subject_measures_list]

        out_of_sample_prediction_results_df = rename_variables_for_plots(out_of_sample_prediction_results_df)
        out_of_sample_prediction_results_df = _rename_models_for_plots(cfg, out_of_sample_prediction_results_df)

        plot_out_of_sample_prediction_scores(
            cfg, out_of_sample_prediction_results_df,
            tvfc_summary_measure=tvfc_summary_measure,
            figures_savedir=os.path.join(
                cfg['figures-basedir'], 'subject_measure_prediction', subject_measures_subset
            )
        )
