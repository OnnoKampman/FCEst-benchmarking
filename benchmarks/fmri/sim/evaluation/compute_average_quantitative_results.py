import logging
import os
import socket
import sys

from fcest.helpers.array_operations import get_all_lower_triangular_indices_tuples
import numpy as np
import pandas as pd
import pingouin

from configs.configs import get_config_dict


# TODO: instead of saving a mean and std/ste csv file, it would be good to save all individual trials to allow for violin plot


if __name__ == "__main__":

    hostname = socket.gethostname()

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    data_split = sys.argv[2]       # 'all', or 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    models_list = cfg['all-quantitative-results-models']
    num_trials = int(experiment_data[-4:])

    for noise_type in cfg['noise-types']:

        git_results_savedir = os.path.join(
            cfg['git-results-basedir'], noise_type, data_split
        )
        if not os.path.exists(git_results_savedir):
            os.makedirs(git_results_savedir)

        for performance_metric in cfg['performance-metrics']:

            print(f"\n> Noise type:         {noise_type:s}")
            print(f"> Performance metric: {performance_metric:s}")

            all_trials_results = []
            for i_trial in range(num_trials):
                quantitative_results_savepath = os.path.join(
                    cfg['experiments-basedir'], noise_type, data_split,
                    f'trial_{i_trial:03d}', f'{performance_metric:s}.csv'
                )

                single_trial_results_df = pd.read_csv(
                    quantitative_results_savepath,
                    index_col=0,
                )  # (num_models, num_covs_types) - can contain NaNs!
                assert single_trial_results_df.shape == (len(models_list), len(cfg['all-covs-types']))

                # missing_cov_types = pd.isnull(single_trial_results_df).all().nonzero()[0]
                # missing_models = pd.isnull(single_trial_results_df).all().nonzero()[0]

                all_trials_results.append(single_trial_results_df.values)

            all_trials_results = np.array(all_trials_results)  # (num_trials, num_models, num_covs_types)
            print('all_results:', all_trials_results.shape)

            # Run statistical significance tests between all estimation methods.
            for _, cov_type in enumerate(cfg['plot-covs-types']):

                i_all_cov_type = cfg['all-covs-types'].index(cov_type)
                print(i_all_cov_type, cov_type, cfg['all-covs-types'])

                all_results_cov_type_df = pd.DataFrame(
                    all_trials_results[:, :, i_all_cov_type],
                    columns=models_list
                )  # (num_trials, n_models)

                all_method_t_values_df = pd.DataFrame(
                    np.nan,
                    index=models_list,
                    columns=models_list
                )
                all_method_significances_df = pd.DataFrame(
                    np.nan,
                    index=models_list,
                    columns=models_list
                )
                all_method_effect_sizes_df = pd.DataFrame(
                    np.nan,
                    index=models_list,
                    columns=models_list
                )
                for (i_model, j_model) in get_all_lower_triangular_indices_tuples(len(models_list)):
                    results_i = all_trials_results[:, i_model, i_all_cov_type]
                    results_j = all_trials_results[:, j_model, i_all_cov_type]
                    if np.isnan(results_i).all() or np.isnan(results_j).all():
                        continue
                    ttest_results_pingouin_df = pingouin.ttest(
                        results_i, results_j,
                        alternative='two-sided',
                        correction='auto'
                    )
                    all_method_t_values_df.iloc[i_model, j_model] = all_method_t_values_df.iloc[j_model, i_model] = ttest_results_pingouin_df['T'].values
                    all_method_significances_df.iloc[i_model, j_model] = all_method_significances_df.iloc[j_model, i_model] = ttest_results_pingouin_df['p-val'].values
                    all_method_effect_sizes_df.iloc[i_model, j_model] = all_method_effect_sizes_df.iloc[j_model, i_model] = ttest_results_pingouin_df['cohen-d'].values
                print(all_method_t_values_df.round(2))
                print(all_method_significances_df.round(4))
                print(all_method_effect_sizes_df.round(2))

                tvals_filename = f'{cov_type:s}_between_method_performance_T.csv'
                pvals_filename = f'{cov_type:s}_between_method_performance_pvals.csv'
                effect_sizes_filename = f'{cov_type:s}_between_method_performance_cohen_d.csv'
                ttest_results_savedir = os.path.join(
                    git_results_savedir, 'ttests', performance_metric
                )
                if not os.path.exists(ttest_results_savedir):
                    os.makedirs(ttest_results_savedir)
                all_method_t_values_df.to_csv(
                    os.path.join(ttest_results_savedir, tvals_filename),
                    index=True,
                    float_format='%.2f',
                )
                all_method_significances_df.to_csv(
                    os.path.join(ttest_results_savedir, pvals_filename),
                    index=True,
                    float_format='%.4e',
                )
                all_method_effect_sizes_df.to_csv(
                    os.path.join(ttest_results_savedir, effect_sizes_filename),
                    index=True,
                    float_format='%.2f',
                )
                logging.info(f"Saved {data_split:s} t-test results '{tvals_filename:s}', '{pvals_filename:s}', and '{effect_sizes_filename:s}' in '{ttest_results_savedir:s}'.")

                pairwise_ttest_results_pingouin_df = pingouin.ptests(
                    all_results_cov_type_df,
                    decimals=4,
                    padjust='bonf',
                    stars=False,
                )
                print(pairwise_ttest_results_pingouin_df.round(4))

                ptests_results_filename = f'{cov_type:s}_between_method_performance_ptests.csv'
                pairwise_ttest_results_pingouin_df.to_csv(
                    os.path.join(ttest_results_savedir, ptests_results_filename),
                    index=True,
                    float_format='%.4f',
                )
                logging.info(f"Saved {data_split:s} t-test ptests results '{ptests_results_filename:s}' in '{ttest_results_savedir:s}'.")

            # re-use the last df for its row and column names
            mean_df = single_trial_results_df.copy()
            mean_df.iloc[:, :] = np.nanmean(
                all_trials_results,
                axis=0
            )
            mean_df['average'] = mean_df.mean(axis=1)
            mean_df.to_csv(
                os.path.join(git_results_savedir, f'{performance_metric:s}_mean.csv'),
                index=True,
                float_format='%.3f'
            )

            # re-use the last df for its row and column names
            std_df = single_trial_results_df.copy()
            std_df.iloc[:, :] = np.nanstd(
                all_trials_results,
                axis=0
            )
            std_df['average'] = std_df.mean(axis=1)
            std_df.to_csv(
                os.path.join(git_results_savedir, f'{performance_metric:s}_std.csv'),
                index=True,
                float_format='%.3f'
            )

            # Standard error.
            # re-use the last df for its row and column names
            se_df = single_trial_results_df.copy()
            se_df.iloc[:, :] = np.nanstd(all_trials_results, axis=0) / np.sqrt(all_trials_results.shape[0])
            se_df['average'] = se_df.mean(axis=1)
            se_df.to_csv(
                os.path.join(git_results_savedir, f'{performance_metric:s}_se.csv'),
                index=True,
                float_format='%.3f'
            )

            logging.info(f"Saved average '{performance_metric:s}' results in '{git_results_savedir:s}'.")
