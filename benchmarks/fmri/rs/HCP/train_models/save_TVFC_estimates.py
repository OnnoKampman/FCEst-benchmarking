import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split, get_tvfc_estimates
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data


if __name__ == "__main__":

    experiment_dimensionality = 'multivariate'

    data_dimensionality = sys.argv[1]  # 'd15' or 'd50'
    data_split = sys.argv[2]           # 'all' or 'LEOO'
    metric = sys.argv[3]               # 'correlation', 'covariance'
    model_name = sys.argv[4]           # 'SVWP_joint', 'SW_30', 'SW_60', or 'sFC'

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

    for i_subject, subject in enumerate(all_subjects_list):

        print(f'\n> SUBJECT {i_subject+1:d} / {num_subjects:d}: {subject:d}\n')

        data_file = os.path.join(cfg['data-dir'], f'{subject:d}.txt')
        for scan_id in cfg['scan-ids']:
            print(f'\nSCAN ID {scan_id:d}\n')

            tvfc_estimates_savedir = os.path.join(
                cfg['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, metric, model_name
            )
            if not os.path.exists(tvfc_estimates_savedir):
                os.makedirs(tvfc_estimates_savedir)

            x, y = load_human_connectome_project_data(
                data_file,
                scan_id=scan_id,
                verbose=False,
            )  # (N, 1), (N, D)
            n_time_steps = x.shape[0]

            match data_split:
                case 'LEOO':
                    x_train, x_test = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                    y_train, y_test = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
                case _:
                    x_train = x  # (N, 1)
                    y_train = y  # (N, D)

            estimated_tvfc = get_tvfc_estimates(
                config_dict=cfg,
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                subject=subject,
                metric=metric,
                scan_id=scan_id,
                data_split=data_split,
                experiment_dimensionality=experiment_dimensionality,
            )

            # Convert predictions to 2D array to save it to disk.
            estimated_tvfc_df = pd.DataFrame(
                estimated_tvfc.reshape(len(estimated_tvfc), -1).T
            )  # (D*D, N)

            estimated_tvfc_df.to_csv(os.path.join(tvfc_estimates_savedir, f'{subject:d}.csv'))
            logging.info(f"Saved {model_name:s} TVFC estimates in '{tvfc_estimates_savedir:s}'.")
