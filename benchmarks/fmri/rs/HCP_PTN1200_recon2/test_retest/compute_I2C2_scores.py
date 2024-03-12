import logging
import os
import socket
import sys

import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.test_retest import compute_tvfc_summary_measure_test_retest_scores


if __name__ == "__main__":

    data_split = 'all'
    experiment_dimensionality = 'multivariate'
    metric = 'correlation'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    model_name = sys.argv[2]           # 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_30', 'SW_60', 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_time_series = int(data_dimensionality[1:])

    i2c2_scores_df = pd.DataFrame(
        index=[model_name], 
        columns=cfg['TVFC-summary-measures']
    )
    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        if model_name == 'sFC' and tvfc_summary_measure != 'mean':
            i2c2_scores_df.loc[model_name, tvfc_summary_measure] = np.nan
            continue
        i2c2_scores_df.loc[model_name, tvfc_summary_measure] = compute_tvfc_summary_measure_test_retest_scores(
            config_dict=cfg,
            test_retest_metric='I2C2',
            model_name=model_name,
            tvfc_summary_measure=tvfc_summary_measure,
            n_time_series=n_time_series,
            experiment_dimensionality=experiment_dimensionality,
            data_split=data_split,
            connectivity_metric=metric
        )

    i2c2_scores_savedir = os.path.join(cfg['git-results-basedir'], 'test_retest', metric)
    if not os.path.exists(i2c2_scores_savedir):
        os.makedirs(i2c2_scores_savedir)
    i2c2_scores_filename = f'I2C2_{model_name:s}_scores.csv'
    i2c2_scores_df = i2c2_scores_df.astype(float).round(3)
    i2c2_scores_df.to_csv(
        os.path.join(i2c2_scores_savedir, i2c2_scores_filename),
        float_format='%.3f'
    )
    logging.info(f"Saved I2C2 scores table '{i2c2_scores_filename:s}' to '{i2c2_scores_savedir:s}'.")
