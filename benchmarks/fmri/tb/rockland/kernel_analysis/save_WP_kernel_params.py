import json
import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.rockland import get_rockland_subjects


if __name__ == "__main__":

    pp_pipeline = 'custom_fsl_pipeline'

    model_name = sys.argv[1]  # 'VWP_joint' or 'SVWP_joint'
    data_split = sys.argv[2]       # 'all' or 'LEOO'
    repetition_time = sys.argv[3]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    n_subjects = len(all_subjects_list)

    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis', data_split, model_name)
    if not os.path.exists(kernel_params_savedir):
        os.makedirs(kernel_params_savedir)

    model_savedir = os.path.join(
        cfg['experiments-basedir'], pp_pipeline, 'saved_models', data_split, model_name
    )

    kernel_param = 'kernel_lengthscales'
    kernel_params_df = pd.DataFrame()
    for i_subject, subject_filename in enumerate(all_subjects_list):
        print(f'\n> SUBJECT {i_subject+1:d} / {n_subjects:d}: {subject_filename:s}\n')
        with open(os.path.join(model_savedir, f"{subject_filename.removesuffix('.csv'):s}.json")) as f:
            m_dict = json.load(f)  # a dictionary
        kernel_params_df.loc[subject_filename.removesuffix('.csv'), kernel_param] = m_dict[kernel_param]

    kernel_params_df_filename = f'{kernel_param:s}_kernel_params.csv'
    kernel_params_df.to_csv(
        os.path.join(kernel_params_savedir, kernel_params_df_filename),
        float_format="%.3f"
    )
    logging.info(f"Saved kernel params table '{kernel_params_df_filename:s}' to '{kernel_params_savedir:s}'.")
