import logging
import os
import socket
import sys

from configs.configs import get_config_dict
from helpers.test_retest import compute_tvfc_summary_measure_test_retest_scores


if __name__ == "__main__":

    experiment_dimensionality = 'multivariate'
    metric = 'correlation'

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'
    data_split = sys.argv[2]           # 'all', 'LEOO'
    model_name = sys.argv[3]           # 'SVWP_joint', 'DCC_joint', 'SW_cross_validated', 'SW_30', 'SW_60', 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_time_series = int(data_dimensionality[1:])

    icc_edgewise_savedir = os.path.join(cfg['git-results-basedir'], 'test_retest', metric)
    if not os.path.exists(icc_edgewise_savedir):
        os.makedirs(icc_edgewise_savedir)

    for tvfc_summary_measure in cfg['TVFC-summary-measures']:
        if model_name == 'sFC' and tvfc_summary_measure != 'mean':
            continue
        summary_measure_icc_df = compute_tvfc_summary_measure_test_retest_scores(
            config_dict=cfg,
            test_retest_metric='ICC',
            model_name=model_name,
            tvfc_summary_measure=tvfc_summary_measure,
            n_time_series=n_time_series,
            experiment_dimensionality=experiment_dimensionality,
            data_split=data_split,
            connectivity_metric=metric
        )
        print(summary_measure_icc_df.shape)

        icc_matrix_filename = f'{tvfc_summary_measure:s}_ICCs_{model_name:s}.csv'
        summary_measure_icc_df.to_csv(
            os.path.join(icc_edgewise_savedir, icc_matrix_filename),
            float_format='%.3f'
        )
        logging.info(f"Saved ICC matrix '{icc_matrix_filename:s}' to '{icc_edgewise_savedir:s}'.")
