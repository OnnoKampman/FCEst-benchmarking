import logging
import os
import socket
import sys

from fcest.helpers.array_operations import get_all_lower_triangular_indices_tuples
import numpy as np
import pandas as pd
import pingouin

from configs.configs import get_config_dict


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'
    metric = 'correlation'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    icc_edgewise_savedir = os.path.join(cfg['git-results-basedir'], 'test_retest', metric)
    models_list = cfg['plot-model-estimates-methods']
    n_time_series = int(data_dimensionality[1:])

    all_method_significances_df = pd.DataFrame(
        np.nan,
        index=models_list,
        columns=models_list
    )
    all_method_cod_df = pd.DataFrame(
        np.nan,
        index=models_list,
        columns=models_list
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        for (i_model, j_model) in get_all_lower_triangular_indices_tuples(len(models_list)):
            print(i_model, j_model)
            i_model_name = models_list[i_model]
            j_model_name = models_list[j_model]
            print(i_model_name, j_model_name)
            if (i_model_name == 'sFC' or j_model_name == 'sFC') and tvfc_summary_measure != 'mean':
                continue

            all_methods_edgewise_icc_scores = []
            for model_name in [i_model_name, j_model_name]:

                icc_matrix_filename = f'{tvfc_summary_measure:s}_ICCs_{model_name:s}.csv'
                icc_matrix_filepath = os.path.join(icc_edgewise_savedir, icc_matrix_filename)
                if not os.path.exists(icc_matrix_filepath):
                    logging.warning(f"ICC scores TVFC {tvfc_summary_measure:s} {model_name:s} not found.")
                    all_methods_edgewise_icc_scores.append(np.full([n_time_series, n_time_series], np.nan).flatten())
                    continue
                summary_measure_icc_df = pd.read_csv(
                    icc_matrix_filepath,
                    index_col=0
                )
                # print(summary_measure_icc_df)
                all_methods_edgewise_icc_scores.append(summary_measure_icc_df.values.flatten())

            ttest_results_pingouin_df = pingouin.ttest(
                all_methods_edgewise_icc_scores[0], all_methods_edgewise_icc_scores[1],
                alternative='two-sided',
                correction='auto'
            )
            print(ttest_results_pingouin_df.round(3))

            all_method_significances_df.iloc[i_model, j_model] = all_method_significances_df.iloc[j_model, i_model] = ttest_results_pingouin_df['p-val'].values
            all_method_cod_df.iloc[i_model, j_model] = all_method_cod_df.iloc[j_model, i_model] = ttest_results_pingouin_df['cohen-d'].values

        print(all_method_significances_df.round(3))
        print(all_method_cod_df.round(3))

        icc_scores_pvals_filename = f'{data_split:s}_{experiment_dimensionality:s}_TVFC_{tvfc_summary_measure:s}_ICC_scores_pvals.csv'
        all_method_significances_df.to_csv(
            os.path.join(icc_edgewise_savedir, icc_scores_pvals_filename),
            index=True,
            # float_format='%.3f'
        )
        logging.info(f"Saved {data_split:s} ICC scores p-values '{icc_scores_pvals_filename:s}' in '{icc_edgewise_savedir:s}'.")

        icc_scores_cod_filename = f'{data_split:s}_{experiment_dimensionality:s}_TVFC_{tvfc_summary_measure:s}_ICC_scores_cod.csv'
        all_method_cod_df.to_csv(
            os.path.join(icc_edgewise_savedir, icc_scores_cod_filename),
            index=True,
            float_format='%.3f'
        )
        logging.info(f"Saved {data_split:s} ICC scores Cohen's D '{icc_scores_cod_filename:s}' in '{icc_edgewise_savedir:s}'.")
