import copy
import logging
import os
import socket
import sys

import numpy as np
import pandas as pd
import pingouin
import scipy

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects, rename_variables_for_plots


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
    prediction_task = 'morphometricity'

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
        prediction_task, subject_measures_subset
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:

        logging.info("TVFC summary measure: %s", tvfc_summary_measure)

        # Load results.
        prediction_task_results_df = pd.read_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_morphometricity_scores_TVFC_{tvfc_summary_measure:s}.csv'
            ),
            index_col=0
        )  # (n_subject_measures, n_models)
        print('\nresults\n')
        print(prediction_task_results_df)

        prediction_task_results_standard_error_df = pd.read_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_morphometricity_scores_standard_error_TVFC_{tvfc_summary_measure:s}.csv'
            ),
            index_col=0
        )  # (n_subject_measures, n_models)
        print('\nresults SE\n')
        print(prediction_task_results_standard_error_df)

        # Re-order subject measures.
        prediction_task_results_df = prediction_task_results_df.loc[subject_measures_list, :]
        prediction_task_results_standard_error_df = prediction_task_results_standard_error_df.loc[subject_measures_list, :]

        prediction_task_results_df = rename_variables_for_plots(prediction_task_results_df)
        prediction_task_results_standard_error_df = rename_variables_for_plots(prediction_task_results_standard_error_df)
        prediction_task_results_df = _rename_models_for_plots(cfg, prediction_task_results_df)
        prediction_task_results_standard_error_df = _rename_models_for_plots(cfg, prediction_task_results_standard_error_df)

        # Compute p-values.
        # https://www.bmj.com/content/343/bmj.d2304
        models_to_compare = ['WP', 'DCC-J']

        estimates = prediction_task_results_df.loc[:, models_to_compare].values  # (n_subject_measures, 2)
        # print(estimates)
        difference_in_means = estimates[:, 0] - estimates[:, 1]  # (n_subject_measures, )
        print(difference_in_means)
        standard_errors = prediction_task_results_standard_error_df.loc[:, models_to_compare].values  # (n_subject_measures, 2)
        print(standard_errors)

        test_statistics = estimates / standard_errors
        print(test_statistics)

        p_values = np.exp(-0.717 * test_statistics - 0.416 * test_statistics ** 2)
        print('\nOne-sided p-values:\n')
        p_values_df = pd.DataFrame(p_values, index=subject_measures_list, columns=models_to_compare)
        print(p_values_df.round(4))

        # Run t-test between two methods for a single subject measure.
        # https://select-statistics.co.uk/calculators/two-sample-t-test-calculator/
        scipy_ttest_results = scipy.stats.ttest_ind_from_stats(
            mean1=estimates[:, 0],
            std1=standard_errors[:, 0] * np.sqrt(n_subjects),
            nobs1=n_subjects,
            mean2=estimates[:, 1],
            std2=standard_errors[:, 1] * np.sqrt(n_subjects),
            nobs2=n_subjects,
            equal_var=False,
            alternative='two-sided'
        )
        scipy_ttest_results_df = pd.DataFrame(
            np.concatenate(
                (scipy_ttest_results.statistic.reshape(-1, 1), scipy_ttest_results.pvalue.reshape(-1, 1)),
                axis=1
            ),
            index=subject_measures_list,
            columns=['t-statistic', 'p-value']
        )
        print(f"\nSciPy t-test results between TVFC estimation methods '{models_to_compare[0]:s}' and '{models_to_compare[1]:s}':\n")
        print(scipy_ttest_results_df)
        scipy_ttest_results_df.to_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_morphometricity_scores_TVFC_{tvfc_summary_measure:s}_{models_to_compare[0]:s}_{models_to_compare[1]:s}_ttest_individual_subject_measures.csv'
            ),
            float_format='%.5f'
        )

        # Run t-test across subject measures.
        pairwise_ttest_results_pingouin_df = pingouin.ptests(
            prediction_task_results_df,
            decimals=5,
            padjust=None,
            stars=False,
        )
        print(f"\nPairwise t-tests over all subject measures TVFC '{tvfc_summary_measure:s}':\n")
        print(pairwise_ttest_results_pingouin_df)
        pairwise_ttest_results_pingouin_df.to_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_morphometricity_scores_TVFC_{tvfc_summary_measure:s}_ttest_all_subject_measures.csv'
            ),
            float_format='%.5f'
        )
