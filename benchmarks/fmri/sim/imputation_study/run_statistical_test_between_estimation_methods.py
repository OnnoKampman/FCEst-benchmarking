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

    data_split = 'LEOO'  # leave-every-other-out

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    experiment_data = sys.argv[2]  # e.g. 'N0200_T0003'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    models_list = cfg['plot-models']
    num_trials = int(experiment_data[-4:])
    test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')

    for noise_type in cfg['noise-types']:

        for covs_type in cfg['plot-covs-types']:

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
                    likelihoods_filename = f'{data_split:s}_{noise_type:s}_likelihoods_{model_name:s}.csv'
                    test_likelihoods_savepath = os.path.join(
                        test_likelihoods_savedir, likelihoods_filename
                    )
                    if not os.path.exists(test_likelihoods_savepath):
                        logging.warning(
                            f"Test likelihoods not found: '{test_likelihoods_savepath:s}'"
                        )
                        continue
                    test_likelihoods_df = pd.read_csv(
                        test_likelihoods_savepath,
                        index_col=0
                    )
                    print(test_likelihoods_df)
                    logging.info(
                        f"Loaded '{likelihoods_filename:s}' from '{test_likelihoods_savedir:s}'."
                    )
                    all_methods_test_likelihoods.append(test_likelihoods_df[covs_type].values)

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

            likelihoods_pvals_filename = f'{data_split:s}_{noise_type:s}_{covs_type:s}_likelihoods_pvals.csv'
            all_method_significances_df.to_csv(
                os.path.join(test_likelihoods_savedir, likelihoods_pvals_filename),
                index=True,
                float_format='%.3f'
            )
            logging.info(f"Saved {data_split:s} likelihoods p-values '{likelihoods_pvals_filename:s}' in '{test_likelihoods_savedir:s}'.")

            likelihoods_cod_filename = f'{data_split:s}_{noise_type:s}_{covs_type:s}_likelihoods_cod.csv'
            all_method_cod_df.to_csv(
                os.path.join(test_likelihoods_savedir, likelihoods_cod_filename),
                index=True,
                float_format='%.3f'
            )
            logging.info(f"Saved {data_split:s} likelihoods Cohen's D '{likelihoods_cod_filename:s}' in '{test_likelihoods_savedir:s}'.")
