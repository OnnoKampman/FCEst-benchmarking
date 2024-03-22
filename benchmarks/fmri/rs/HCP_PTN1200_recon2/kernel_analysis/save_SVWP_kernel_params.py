import json
import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'
    model_name = 'SVWP_joint'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    # We save the kernel parameters with the experiments in the git repo.
    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis', data_split)
    if not os.path.exists(kernel_params_savedir):
        os.makedirs(kernel_params_savedir)

    n_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'], first_n_subjects=n_subjects, as_ints=True
    )

    # TODO: refactor so we only load each model once?
    for kernel_param in cfg['kernel-params']:
        kernel_params_df = pd.DataFrame(index=all_subjects_list, columns=cfg['scan-ids'])
        print('\nKERNEL PARAM:', kernel_param, '\n')

        for i_subject, subject in enumerate(all_subjects_list):
            print(f'\n> SUBJECT {i_subject+1:d} / {n_subjects:d}: {subject:d}')

            for scan_id in cfg['scan-ids']:
                model_savedir = os.path.join(
                    cfg['experiments-basedir'], 'saved_models', f'scan_{scan_id:d}',
                    data_split, experiment_dimensionality, model_name
                )
                with open(os.path.join(model_savedir, f'{subject:d}.json')) as f:
                    m_dict = json.load(f)  # a dictionary
                kernel_params_df.loc[subject, scan_id] = m_dict[kernel_param]

        kernel_params_df_filename = f'{kernel_param:s}_kernel_params.csv'
        kernel_params_df.to_csv(
            os.path.join(kernel_params_savedir, kernel_params_df_filename),
            float_format='%.3f'
        )
        logging.info(f"Saved kernel params table '{kernel_params_df_filename:s}' to '{kernel_params_savedir:s}'.")
