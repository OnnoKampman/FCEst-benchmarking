import logging
import os
import socket
import sys

from fcest.helpers.data import to_3d_format
from fcest.helpers.summary_measures import summarize_tvfc_estimates
from nilearn import connectome
import numpy as np
import pandas as pd

from ......configs.configs import get_config_dict
from ....helpers.array_operations import reconstruct_symmetric_summary_measure_matrix_from_tril
from .....helpers.hcp import get_human_connectome_project_subjects


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    metric = sys.argv[2]               # 'covariance', 'correlation'
    model_name = sys.argv[3]           # 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_30', 'SW_60', 'SW_cross_validated', 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_subjects = cfg['n-subjects']
    n_time_series = int(data_dimensionality[1:])
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'], first_n_subjects=n_subjects
    )

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        if model_name == 'sFC' and tvfc_summary_measure != 'mean':
            continue
        for scan_id in cfg['scan-ids']:

            # edgewise_tvfc_summary_per_subject_df = pd.DataFrame(
            #     np.nan,
            #     index=[subject_filename.removesuffix('.txt') for subject_filename in all_subject_filenames_list],
            #     columns=range(n_trils)
            # )  # (n_subjects, D*(D-1)/2)
            edgewise_tvfc_summary_per_subject_df = []

            tvfc_estimates_savedir = os.path.join(
                cfg['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, metric, model_name
            )
            for i_subject, subject_filename in enumerate(all_subjects_list):
                print('')
                logging.info(f'> SUMMARY MEASURE: {tvfc_summary_measure:s}')
                logging.info(f'> MODEL NAME:      {model_name:s}')
                logging.info(f'> SCAN ID:         {scan_id:d}')
                logging.info(f'> SUBJECT {i_subject+1: 3d} / {n_subjects:d}: {subject_filename:s}')

                # Load TVFC estimates - some may be missing.
                tvfc_estimates_filepath = os.path.join(
                    tvfc_estimates_savedir, f"{subject_filename.removesuffix('.txt'):s}.csv"
                )
                if not os.path.exists(tvfc_estimates_filepath):
                    logging.warning(f"Could not find TVFC estimates '{tvfc_estimates_filepath:s}'.")
                    tvfc_estimates_array = np.empty((int(n_time_series * (n_time_series - 1) / 2), ))
                    tvfc_estimates_array[:] = np.nan
                else:
                    tvfc_estimates_df = pd.read_csv(tvfc_estimates_filepath, index_col=0)  # (D*D, N)
                    tvfc_estimates = to_3d_format(tvfc_estimates_df.values)  # (N, D, D)
                    tvfc_estimates_summary = summarize_tvfc_estimates(
                        full_covariance_structure=tvfc_estimates,
                        tvfc_summary_metric=tvfc_summary_measure
                    )  # (D, D)
                    tvfc_estimates_array = connectome.sym_matrix_to_vec(tvfc_estimates_summary, discard_diagonal=True)  # (D*(D-1)/2, )
                edgewise_tvfc_summary_per_subject_df.append(tvfc_estimates_array)
            edgewise_tvfc_summary_per_subject_df = pd.DataFrame(
                edgewise_tvfc_summary_per_subject_df,
                index=all_subjects_list
            )  # (n_subjects, D*(D-1)/2)
            print(edgewise_tvfc_summary_per_subject_df)

            tvfc_estimates_summaries_savedir = os.path.join(
                cfg['experiments-basedir'], 'TVFC_estimates_summary_measures', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, metric, model_name
            )
            if not os.path.exists(tvfc_estimates_summaries_savedir):
                os.makedirs(tvfc_estimates_summaries_savedir)
            file_name = f"TVFC_{tvfc_summary_measure:s}.csv"
            edgewise_tvfc_summary_per_subject_df.to_csv(
                os.path.join(tvfc_estimates_summaries_savedir, file_name),
                float_format='%.3f'
            )
            logging.info(f"Saved TVFC summaries '{file_name:s}' in '{tvfc_estimates_summaries_savedir:s}'.")

            mean_over_subjects_edgewise_summarized_tvfc_df = edgewise_tvfc_summary_per_subject_df.mean(axis=0)  # (D*(D-1)/2, )
            mean_over_subjects_edgewise_summarized_tvfc_array = pd.DataFrame(
                reconstruct_symmetric_summary_measure_matrix_from_tril(
                    mean_over_subjects_edgewise_summarized_tvfc_df.values,
                    tvfc_summary_measure=tvfc_summary_measure,
                    n_time_series=n_time_series
                )
            )  # (D, D)

            tvfc_estimates_git_savedir = os.path.join(
                cfg['git-results-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, metric
            )
            if not os.path.exists(tvfc_estimates_git_savedir):
                os.makedirs(tvfc_estimates_git_savedir)
            file_name = f"{model_name:s}_TVFC_{tvfc_summary_measure:s}_mean_over_subjects.csv"
            mean_over_subjects_edgewise_summarized_tvfc_array.to_csv(
                os.path.join(tvfc_estimates_git_savedir, file_name),
                float_format='%.3f'
            )
            logging.info(f"Saved TVFC summaries '{file_name:s}' in '{tvfc_estimates_git_savedir:s}'.")
