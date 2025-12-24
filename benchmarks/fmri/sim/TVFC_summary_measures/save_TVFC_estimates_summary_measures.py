import logging
import os
import socket
import sys

from fcest.helpers.data import to_3d_format
from fcest.helpers.summary_measures import summarize_tvfc_estimates
from nilearn import connectome
import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.array_operations import reconstruct_symmetric_summary_measure_matrix_from_tril


if __name__ == "__main__":

    metric = 'correlation'

    data_set_name = sys.argv[1]    # 'd2', 'd3d', 'd{%d}s'
    data_split = sys.argv[2]       # 'all', 'LEOO'
    experiment_data = sys.argv[3]  # 'Nxxxx_Txxxx'
    model_name = sys.argv[4]       # 'SVWP_joint', 'DCC', 'DCC_joint', 'SW_cross_validated', 'sFC'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    num_trials = int(experiment_data[-4:])
    num_time_series = int(data_set_name[1:-1])

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:

        if model_name == 'sFC' and tvfc_summary_measure != 'mean':
            continue

        for noise_type in cfg['noise-types']:

            for covs_type in cfg['all-covs-types']:

                for i_trial in range(num_trials):

                    edgewise_tvfc_summary_per_trial_df = []

                    tvfc_estimates_savedir = os.path.join(
                        cfg['experiments-basedir'], noise_type, f'trial_{i_trial:03d}', 'TVFC_estimates',
                        data_split, metric, model_name
                    )
                    tvfc_estimates_filepath = os.path.join(
                        tvfc_estimates_savedir, f'{covs_type:s}.csv'
                    )
                    if not os.path.exists(tvfc_estimates_filepath):
                        logging.warning(f"Could not find TVFC estimates '{tvfc_estimates_filepath:s}'.")
                        tvfc_estimates_array = np.empty((int(num_time_series * (num_time_series - 1) / 2), ))
                        tvfc_estimates_array[:] = np.nan
                    else:
                        tvfc_estimates_df = pd.read_csv(tvfc_estimates_filepath, index_col=0)  # (D*D, N)
                        tvfc_estimates = to_3d_format(tvfc_estimates_df.values)  # (N, D, D)
                        tvfc_estimates_summary = summarize_tvfc_estimates(
                            full_covariance_structure=tvfc_estimates,
                            tvfc_summary_metric=tvfc_summary_measure
                        )  # (D, D)
                        tvfc_estimates_array = connectome.sym_matrix_to_vec(tvfc_estimates_summary, discard_diagonal=True)  # (D*(D-1)/2, )
                    edgewise_tvfc_summary_per_trial_df.append(tvfc_estimates_array)

                edgewise_tvfc_summary_per_trial_df = pd.DataFrame(
                    edgewise_tvfc_summary_per_trial_df,
                    index=range(num_trials)
                )  # (num_trials, D*(D-1)/2)
                print(edgewise_tvfc_summary_per_trial_df)

                tvfc_estimates_summaries_savedir = os.path.join(
                    cfg['experiments-basedir'], 'TVFC_estimates_summary_measures', noise_type, f'trial_{i_trial:03d}',
                    data_split, metric, model_name
                )
                if not os.path.exists(tvfc_estimates_summaries_savedir):
                    os.makedirs(tvfc_estimates_summaries_savedir)
                file_name = f"TVFC_{tvfc_summary_measure:s}.csv"
                edgewise_tvfc_summary_per_trial_df.to_csv(
                    os.path.join(tvfc_estimates_summaries_savedir, file_name),
                    float_format='%.3f'
                )
                logging.info(f"Saved TVFC summaries '{file_name:s}' in '{tvfc_estimates_summaries_savedir:s}'.")

                mean_over_trials_edgewise_summarized_tvfc_df = edgewise_tvfc_summary_per_trial_df.mean(axis=0)  # (D*(D-1)/2, )
                mean_over_trials_edgewise_summarized_tvfc_array = pd.DataFrame(
                    reconstruct_symmetric_summary_measure_matrix_from_tril(
                        mean_over_trials_edgewise_summarized_tvfc_df.values,
                        tvfc_summary_measure=tvfc_summary_measure,
                        num_time_series=num_time_series,
                    )
                )  # (D, D)

                tvfc_estimates_git_savedir = os.path.join(
                    cfg['git-results-basedir'], noise_type, data_split, 'TVFC_estimates_summary_measures', f'trial_{i_trial:03d}', metric, covs_type
                )
                if not os.path.exists(tvfc_estimates_git_savedir):
                    os.makedirs(tvfc_estimates_git_savedir)
                file_name = f"{model_name:s}_TVFC_{tvfc_summary_measure:s}_mean_over_trials.csv"
                mean_over_trials_edgewise_summarized_tvfc_array.to_csv(
                    os.path.join(tvfc_estimates_git_savedir, file_name),
                    float_format='%.3f'
                )
                logging.info(f"Saved TVFC summaries '{file_name:s}' in '{tvfc_estimates_git_savedir:s}'.")
