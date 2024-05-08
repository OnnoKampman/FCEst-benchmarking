import logging
import os
import socket
import sys

from fcest.helpers.data import to_3d_format
from nilearn import connectome
import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.brain_states import compute_basis_state
from helpers.hcp import get_human_connectome_project_subjects


if __name__ == "__main__":

    connectivity_metric = 'correlation'
    data_split = 'all'
    experiment_dimensionality = 'multivariate'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    model_name = sys.argv[2]           # 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_30', 'SW_60'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    num_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=num_subjects,
        as_ints=True,
    )

    num_brain_states_list = cfg['n-brain-states-list']
    distortions_df = pd.DataFrame(
        index=num_brain_states_list,
        columns=cfg['scan-ids'],
    )
    for scan_id in cfg['scan-ids']:
        num_subjects = len(all_subjects_list)
        all_subjects_tril_tvfc_per_time_step = []
        for i_subject, subject in enumerate(all_subjects_list):
            logging.info(f'> SUBJECT {i_subject+1:d} / {num_subjects:d}: {subject:d}')
            tvfc_estimates_savedir = os.path.join(
                cfg['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, connectivity_metric, model_name
            )
            tvfc_estimates_filepath = os.path.join(tvfc_estimates_savedir, f'{subject:d}.csv')
            single_subject_tvfc_estimates_df = pd.read_csv(tvfc_estimates_filepath, index_col=0)  # (D*D, N)
            single_subject_tvfc_estimates = to_3d_format(single_subject_tvfc_estimates_df.values)  # (N, D, D)

            # Select lower triangular values only (e.g. 105 for D=15).
            subject_tril_tvfc_per_time_step = connectome.sym_matrix_to_vec(
                single_subject_tvfc_estimates, discard_diagonal=True
            )  # (N, D*(D-1)/2)
            all_subjects_tril_tvfc_per_time_step.append(subject_tril_tvfc_per_time_step)

        num_time_steps = single_subject_tvfc_estimates.shape[0]
        num_time_series = single_subject_tvfc_estimates.shape[1]

        # Aggregates all observed 'states' over time and over subjects.
        all_subjects_tril_tvfc_per_time_step = np.array(all_subjects_tril_tvfc_per_time_step)  # (n_subjects, N, D*(D-1)/2)
        all_subjects_tril_tvfc_per_time_step = all_subjects_tril_tvfc_per_time_step.reshape(-1, all_subjects_tril_tvfc_per_time_step.shape[-1])  # (n_subjects*N, D*(D-1)/2)
        assert all_subjects_tril_tvfc_per_time_step.shape == (num_subjects * num_time_steps, int(num_time_series * (num_time_series-1) / 2))

        for num_brain_states in num_brain_states_list:
            num_brain_states_inertia, _, _ = compute_basis_state(
                config_dict=cfg,
                all_subjects_tril_tvfc=all_subjects_tril_tvfc_per_time_step,
                scan_session_id=scan_id,
                model_name=model_name,
                n_basis_states=num_brain_states,
                n_time_series=num_time_series,
                n_time_steps=num_time_steps,
            )
            distortions_df.loc[num_brain_states, scan_id] = num_brain_states_inertia / num_subjects

    distortions_df.astype(float).round(2).to_csv(
        os.path.join(
            cfg['git-results-basedir'], 'brain_states', f'{connectivity_metric:s}_inertias_{model_name:s}.csv'
        ),
        float_format='%.2f'
    )
    logging.info("Inertias saved.")
