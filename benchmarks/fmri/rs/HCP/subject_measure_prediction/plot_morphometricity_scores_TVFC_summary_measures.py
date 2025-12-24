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


def plot_morphometricity_scores_joint(
    config_dict: dict,
    scores_savedir: str,
    subject_measures_list: list[str],
    connectivity_metric: str = 'correlation',
    figures_savedir: str = None,
) -> None:
    """
    Plot all TVFC summary measures jointly in a single figure.

    Parameters
    ----------
    config_dict : dict
    scores_savedir : str
    subject_measures_list : list[str]
    connectivity_metric : str, optional
    figures_savedir : str, optional
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, axes = plt.subplots(
        # figsize=set_size(subplots=(3, 1)),
        # figsize=config_dict['plot-morphometricity-scores-figsize'],
        figsize=(8.4, 7.0),
        nrows=len(config_dict['plot-TVFC-summary-measures']),
        ncols=1,
    )
    for i_tvfc_summary_measure, tvfc_summary_measure in enumerate(config_dict['plot-TVFC-summary-measures']):
        morphometricity_results_df = pd.read_csv(
            os.path.join(
                scores_savedir, f'{connectivity_metric:s}_morphometricity_scores_TVFC_{tvfc_summary_measure:s}.csv'
            ),
            index_col=0
        )  # (n_subject_measures, n_models)
        morphometricity_results_standard_error_df = pd.read_csv(
            os.path.join(
                scores_savedir, f'{connectivity_metric:s}_morphometricity_scores_standard_error_TVFC_{tvfc_summary_measure:s}.csv'
            ),
            index_col=0
        )  # (n_subject_measures, n_models)

        # Set standard error to zero if variance explained is very small.
        morphometricity_results_standard_error_df[morphometricity_results_df < 0.03] = 0.0

        # Re-order subject measures.
        morphometricity_results_df = morphometricity_results_df.loc[subject_measures_list, :]
        morphometricity_results_standard_error_df = morphometricity_results_standard_error_df.loc[subject_measures_list, :]

        morphometricity_results_df = rename_variables_for_plots(morphometricity_results_df)
        morphometricity_results_standard_error_df = rename_variables_for_plots(morphometricity_results_standard_error_df)
        morphometricity_results_df = _rename_models_for_plots(config_dict, morphometricity_results_df)
        morphometricity_results_standard_error_df = _rename_models_for_plots(config_dict, morphometricity_results_standard_error_df)

        # Add mean morphometricity score across subject measures.
        empty_array = np.zeros_like(np.mean(morphometricity_results_df, axis=0))
        empty_array[:] = np.nan
        morphometricity_results_df.loc['', :] = empty_array
        morphometricity_results_df.loc['Mean', :] = np.mean(morphometricity_results_df, axis=0)
        empty_array = np.zeros_like(np.mean(morphometricity_results_standard_error_df, axis=0))
        empty_array[:] = np.nan
        morphometricity_results_standard_error_df.loc['', :] = empty_array
        morphometricity_results_standard_error_df.loc['Mean', :] = np.mean(morphometricity_results_standard_error_df, axis=0)

        morphometricity_results_df.plot(
            ax=axes[i_tvfc_summary_measure],
            kind='bar',
            # linestyle='none',
            capsize=2.0,
            color={
                "WP": get_palette(config_dict['plot-models'])[0],
                "DCC-J": get_palette(config_dict['plot-models'])[1],
                "SW-CV": get_palette(config_dict['plot-models'])[2],
                "sFC": get_palette(config_dict['plot-models'])[3],
            },
            width=0.72,
            # xticks=xticks,
            legend=not i_tvfc_summary_measure,  # only add legend in first subplot
            # figsize=(10, 4),
            error_kw={
                'elinewidth': 1,
                'markeredgewidth': 1,
            },
            yerr=morphometricity_results_standard_error_df,
        )
        axes[i_tvfc_summary_measure].set_ylabel(f"TVFC {tvfc_summary_measure.replace('_', '-'):s}\nVariance explained")
        axes[i_tvfc_summary_measure].set_ylim([-0.03, 1.19])
        if not i_tvfc_summary_measure + 1 == len(config_dict['plot-TVFC-summary-measures']):
            axes[i_tvfc_summary_measure].set_xticklabels([])

    axes[0].legend(
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        title='TVFC\nestimator',
        alignment='left',
    )
    plt.xticks(
        rotation=35,
        ha="right",
    )
    # plt.tight_layout()

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_name = 'morphometricity_all_TVFC_summary_measures.pdf'
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_morphometricity_scores(
        config_dict: dict,
        morphometricity_results_df: pd.DataFrame, morphometricity_results_standard_error_df: pd.DataFrame,
        tvfc_summary_measure: str, figures_savedir: str = None
) -> None:
    """
    Plot morphometricity results for single TVFC summary measure.
    """
    sns.set(style="whitegrid")
    plt.style.use(os.path.join(config_dict['git-basedir'], 'configs', 'fig.mplstyle'))

    fig, ax = plt.subplots(
        figsize=set_size()
    )
    morphometricity_results_df.plot(
        ax=ax,
        kind='bar',
        # kind='line',
        # linestyle='none',
        capsize=2.0,
        color={
            "WP": get_palette(config_dict['plot-models'])[0],
            "DCC-J": get_palette(config_dict['plot-models'])[1],
            "SW-CV": get_palette(config_dict['plot-models'])[2],
            "sFC": get_palette(config_dict['plot-models'])[3],
        },
        width=0.72,
        yerr=morphometricity_results_standard_error_df
    )
    plt.xticks(rotation=35, ha="right")
    ax.set_ylabel('Variance explained')
    ax.set_ylim([-0.03, 1.19])
    ax.legend(
        bbox_to_anchor=(1.21, 1.0), frameon=True,
        title='TVFC\nestimator', alignment='left'
    )
    # plt.tight_layout()

    if figures_savedir is not None:
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        figure_name = f'morphometricity_TVFC_{tvfc_summary_measure:s}.pdf'
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

    data_dimensionality = sys.argv[1]      # 'd15', 'd50'
    subject_measures_subset = sys.argv[2]  # 'cognitive', 'other', 'personality', 'psychiatric', or 'social-emotional'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    num_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=num_subjects,
        as_ints=True
    )

    subject_measures_list = cfg[f"subject-measures-{subject_measures_subset:s}"]
    if subject_measures_subset == 'cognitive':
        subject_measures_list = cfg['subject-measures-nuisance-variables'] + subject_measures_list

    scores_savedir = os.path.join(
        cfg['git-results-basedir'], 'subject_measure_prediction', 
        'morphometricity', subject_measures_subset
    )

    plot_morphometricity_scores_joint(
        cfg,
        scores_savedir,
        subject_measures_list,
        figures_savedir=os.path.join(
            cfg['figures-basedir'], 'subject_measure_prediction', subject_measures_subset
        )
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        morphometricity_results_df = pd.read_csv(
            os.path.join(scores_savedir, f'{metric:s}_morphometricity_scores_TVFC_{tvfc_summary_measure:s}.csv'),
            index_col=0
        )  # (n_subject_measures, n_models)
        print('results')
        print(morphometricity_results_df)
        morphometricity_results_standard_error_df = pd.read_csv(
            os.path.join(scores_savedir, f'{metric:s}_morphometricity_scores_standard_error_TVFC_{tvfc_summary_measure:s}.csv'),
            index_col=0
        )  # (n_subject_measures, n_models)
        print('results SE')
        print(morphometricity_results_standard_error_df)

        # Re-order subject measures.
        morphometricity_results_df = morphometricity_results_df.loc[subject_measures_list, :]
        morphometricity_results_standard_error_df = morphometricity_results_standard_error_df.loc[subject_measures_list, :]

        morphometricity_results_df = rename_variables_for_plots(morphometricity_results_df)
        morphometricity_results_standard_error_df = rename_variables_for_plots(morphometricity_results_standard_error_df)
        morphometricity_results_df = _rename_models_for_plots(cfg, morphometricity_results_df)
        morphometricity_results_standard_error_df = _rename_models_for_plots(cfg, morphometricity_results_standard_error_df)

        # Add mean morphometricity score across subject measures (with some empty white space before it).
        morphometricity_results_df.loc['', :] = np.full_like(np.mean(morphometricity_results_df, axis=0), np.nan)
        morphometricity_results_df.loc['Mean', :] = np.mean(morphometricity_results_df, axis=0)
        morphometricity_results_standard_error_df.loc['', :] = np.full_like(np.mean(morphometricity_results_standard_error_df, axis=0), np.nan)
        morphometricity_results_standard_error_df.loc['Mean', :] = np.mean(morphometricity_results_standard_error_df, axis=0)

        _plot_morphometricity_scores(
            cfg, morphometricity_results_df, morphometricity_results_standard_error_df,
            tvfc_summary_measure=tvfc_summary_measure,
            figures_savedir=os.path.join(
                cfg['figures-basedir'], 'subject_measure_prediction', subject_measures_subset
            )
        )
