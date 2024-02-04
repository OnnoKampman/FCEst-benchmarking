import logging
import os
import socket
import sys

from fcest.models.mgarch import MGARCH
import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    data_dimensionality = sys.argv[1]        # 'd15' or 'd50'
    data_split = sys.argv[2]                 # 'all' or 'LEOO'
    experiment_dimensionality = sys.argv[3]  # 'multivariate' or 'bivariate'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=hostname
    )
    all_subjects_list = get_human_connectome_project_subjects(data_dir=cfg['data-dir'])
    n_subjects = len(all_subjects_list)

    # Allow for local and CPU cluster training.
    # When running on the Hivemind with SLURM, only one model is trained here.
    if hostname == 'hivemind':
        try:
            i_subject = os.environ['SLURM_ARRAY_TASK_ID']
            i_subject = int(i_subject) - 1  # to make zero-index
            print('SLURM trial ID', i_subject)
            subjects = [
                all_subjects_list[i_subject]
            ]
        except KeyError:
            subjects = all_subjects_list
    else:
        print('Running locally...')
        subjects = all_subjects_list

    for i_subject, subject_filename in enumerate(subjects):

        print(f'\n> SUBJECT {i_subject+1:d} / {n_subjects:d}: {subject_filename:s}')

        data_file = os.path.join(cfg['data-dir'], subject_filename)
        for scan_id in cfg['scan-ids']:
            print(f'\nSCAN {scan_id:d} / 3\n')

            x, y = load_human_connectome_project_data(
                data_file, scan_id=scan_id, verbose=False
            )  # (N, 1), (N, D)

            # TODO: pick two time series at random if 'experiment_dimensionality' == 'bivariate'?
            if experiment_dimensionality == 'bivariate':
                chosen_indices = [0, 1]
                # chosen_indices_df = cfg['chosen-indices']
                # chosen_indices = chosen_indices_df.loc[subject, scan_id]
                y = y[:, chosen_indices]
                print('y', y.shape)

            # Select train and test data through a leave-every-other-out split.
            if data_split == 'LEOO':
                x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), _
                y_train, _ = leave_every_other_out_split(y)  # (N/2, D), _
            else:
                x_train = x  # (N, 1)
                y_train = y  # (N, D)

            for model_name in cfg['mgarch-models']:

                for training_type in cfg['mgarch-training-types']:

                    # Save model estimates both correlation and covariance.
                    for metric in ['correlation', 'covariance']:

                        tvfc_estimates_savedir = os.path.join(
                            cfg['experiments-basedir'], 'TVFC_estimates', f'scan_{scan_id:d}',
                            data_split, experiment_dimensionality, metric, f'{model_name:s}_{training_type:s}'
                        )
                        tvfc_estimates_savepath = os.path.join(tvfc_estimates_savedir, f"{subject_filename.removesuffix('.txt'):s}.csv")
                        if not os.path.exists(tvfc_estimates_savepath):
                            print(f'\nMODEL {model_name:s} | TRAINING {training_type:s}')
                            m = MGARCH(mgarch_type=model_name)
                            m.fit_model(training_data_df=pd.DataFrame(y_train), training_type=training_type)
                            m.save_tvfc_estimates(
                                savedir=tvfc_estimates_savedir,
                                model_name=f"{subject_filename.removesuffix('.txt'):s}.csv",
                                connectivity_metric=metric
                            )
                        else:
                            logging.info(f"Skipping training: existing model found in '{tvfc_estimates_savedir:s}'.")
