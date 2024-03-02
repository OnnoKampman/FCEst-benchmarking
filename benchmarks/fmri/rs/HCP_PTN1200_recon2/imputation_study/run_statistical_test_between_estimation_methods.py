import logging
import os
import socket
import sys

import numpy as np
import pandas as pd
import pingouin

from configs.configs import get_config_dict
from helpers.array_operations import get_all_lower_triangular_indices_tuples


if __name__ == "__main__":

    data_split = 'LEOO'  # leave-every-other-out

    data_dimensionality = sys.argv[1]        # 'd15', 'd50'
    experiment_dimensionality = sys.argv[2]  # 'multivariate', 'bivariate'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    models_list = cfg['plot-model-estimates-methods']
    test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
    test_likelihoods_ttests_savedir = os.path.join(test_likelihoods_savedir, 'ttests')

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
    for (i_model, j_model) in get_all_lower_triangular_indices_tuples(len(models_list)):
        print(i_model, j_model)

        all_methods_test_likelihoods = []
        for model_name in [models_list[i_model], models_list[j_model]]:
            likelihoods_filename = f'{data_split}_{experiment_dimensionality:s}_likelihoods_{model_name:s}.csv'
            test_likelihoods_savepath = os.path.join(test_likelihoods_savedir, likelihoods_filename)
            if not os.path.exists(test_likelihoods_savepath):
                logging.info("Test likelihoods not found.")
                continue
            test_likelihoods_df = pd.read_csv(
                test_likelihoods_savepath,
                index_col=0
            )
            print(test_likelihoods_df)
            logging.info(f"Loaded '{likelihoods_filename:s}' from '{test_likelihoods_savedir:s}'.")
            all_methods_test_likelihoods.append(test_likelihoods_df.values.flatten())

        ttest_results_pingouin_df = pingouin.ttest(
            all_methods_test_likelihoods[0], all_methods_test_likelihoods[1],
            alternative='two-sided',
            correction='auto'
        )
        print(ttest_results_pingouin_df.round(3))

        all_method_significances_df.iloc[i_model, j_model] = all_method_significances_df.iloc[j_model, i_model] = ttest_results_pingouin_df['p-val'].values
        all_method_cod_df.iloc[i_model, j_model] = all_method_cod_df.iloc[j_model, i_model] = ttest_results_pingouin_df['cohen-d'].values
    print(all_method_significances_df.round(3))
    print(all_method_cod_df.round(3))

    likelihoods_pvals_filename = f'{data_split:s}_{experiment_dimensionality:s}_likelihoods_pvals.csv'
    all_method_significances_df.to_csv(
        os.path.join(test_likelihoods_ttests_savedir, likelihoods_pvals_filename),
        index=True,
        float_format='%.3f'
    )
    logging.info(f"Saved {data_split:s} likelihoods p-values '{likelihoods_pvals_filename:s}' in '{test_likelihoods_ttests_savedir:s}'.")

    likelihoods_cod_filename = f'{data_split:s}_{experiment_dimensionality:s}_likelihoods_cod.csv'
    all_method_cod_df.to_csv(
        os.path.join(test_likelihoods_ttests_savedir, likelihoods_cod_filename),
        index=True,
        float_format='%.3f'
    )
    logging.info(f"Saved {data_split:s} likelihoods Cohen's D '{likelihoods_cod_filename:s}' in '{test_likelihoods_ttests_savedir:s}'.")
