import logging
import os
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.icc import compute_icc_scores_pingouin


if __name__ == "__main__":

    model_name = 'SVWP_joint'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality
    )

    # The kernel parameters were saved with the experiments in the git repo.
    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis')

    icc_scores_df = pd.DataFrame(index=cfg['kernel-params'], columns=['ICC2'])
    for kernel_param in cfg['kernel-params']:
        kernel_params_df_filename = f'{kernel_param:s}_kernel_params.csv'
        kernel_params_df = pd.read_csv(
            os.path.join(kernel_params_savedir, kernel_params_df_filename),
            index_col=0
        )
        print(kernel_params_df)

        icc_score = compute_icc_scores_pingouin(kernel_params_df.values, icc_type='ICC2')
        icc_scores_df.loc[kernel_param, 'ICC2'] = icc_score
    icc_scores_df_filename = 'kernel_params_icc_scores.csv'
    icc_scores_df.to_csv(
        os.path.join(kernel_params_savedir, icc_scores_df_filename),
        float_format='%.3f'
    )
    logging.info(f"Saved kernel params ICC scores '{icc_scores_df_filename:s}' to '{kernel_params_savedir:s}'.")
