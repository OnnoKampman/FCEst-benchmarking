import logging
import os
import socket

import numpy as np
import pandas as pd

from configs.configs import get_config_dict
from helpers.rockland import get_rockland_subjects, get_convolved_stim_array


if __name__ == '__main__':

    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )

    # Get stimulus time series (convolved with HRF).
    convolved_stim_array = get_convolved_stim_array(config_dict=cfg)

    all_subjects_list = get_rockland_subjects(config_dict=cfg)

    correlation_results_df = pd.DataFrame()
    for i_subject, subject_filename in enumerate(all_subjects_list):
        subject_timeseries_df = pd.read_csv(
            os.path.join(cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename)
        )

        # Get V1 time series.
        v1_time_series = subject_timeseries_df['V1']

        # Compute correlation.
        correlation = np.corrcoef(convolved_stim_array, v1_time_series)[0, 1]
        correlation_results_df.loc[subject_filename, 'V1-stim_correlation'] = correlation

    print(correlation_results_df)
    correlations_filename = "V1_stimulus_correlations.csv"
    correlation_results_df.to_csv(
        os.path.join(cfg["git-results-basedir"], correlations_filename),
        index=True,
        float_format="%.2f"
    )
    logging.info(f"Saved V1-stimulus correlations '{correlations_filename:s}'.")

    # TODO: save histogram of correlations
