import logging
import os
import socket
import sys

from fcest.models.mgarch import MGARCH
import pandas as pd

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.rockland import get_rockland_subjects, load_rockland_data


if __name__ == "__main__":

    pp_pipeline = 'custom_fsl_pipeline'

    data_split = sys.argv[1]       # 'all' or 'LEOO'
    repetition_time = sys.argv[2]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )

    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    n_subjects = len(all_subjects_list)

    # Allow for local and CPU cluster training.
    if socket.gethostname() == 'hivemind':
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
        subjects = all_subjects_list

    for i_subject, subject_filename in enumerate(subjects):
        print(f'\n> SUBJECT {i_subject+1:d} / {n_subjects:d}: {subject_filename:s}\n')

        data_file = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        if data_split == 'LEOO':
            x_train, _ = leave_every_other_out_split(x)  # (N/2, 1)
            y_train, _ = leave_every_other_out_split(y)  # (N/2, D)
        else:
            x_train = x  # (N, 1)
            y_train = y  # (N, D)
        n_time_series = y_train.shape[1]

        for model_name in cfg['mgarch-models']:

            for training_type in cfg['mgarch-training-types']:

                # Save model estimates, both correlation and covariance.
                for metric in ['correlation', 'covariance']:
                    tvfc_estimates_savedir = os.path.join(
                        cfg['experiments-basedir'], pp_pipeline, 'TVFC_estimates',
                        data_split, metric, f'{model_name:s}_{training_type:s}'
                    )
                    if not os.path.exists(tvfc_estimates_savedir):
                        os.makedirs(tvfc_estimates_savedir)
                    tvfc_estimates_savepath = os.path.join(tvfc_estimates_savedir, subject_filename)
                    if not os.path.exists(tvfc_estimates_savepath):
                        print(f'\nMODEL {model_name:s} | TRAINING {training_type:s}')
                        m = MGARCH(mgarch_type=model_name)
                        m.fit_model(training_data_df=pd.DataFrame(y_train), training_type=training_type)
                        m.save_tvfc_estimates(
                            savedir=tvfc_estimates_savedir,
                            model_name=subject_filename,
                            connectivity_metric=metric
                        )
                    else:
                        logging.info(f"Skipping training: existing model found in '{tvfc_estimates_savedir:s}'.")
