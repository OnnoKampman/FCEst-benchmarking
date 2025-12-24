import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.icc import compute_icc_scores_pingouin


def _compute_brain_state_switch_count_icc(
    config_dict: dict,
    num_basis_states: int,
    connectivity_metric: str = 'correlation',
) -> None:
    """
    The number of brain state switches is closely related to the idea of "dwell times".

    Parameters
    ----------
    :param config_dict:
    :param num_basis_states:
    :param connectivity_metric:
        'correlation', 'covariance'
    :return:
    """
    all_brain_state_switch_counts_iccs_df = pd.DataFrame()
    for model_name in config_dict['models-brain-state-analysis']:
        # Load number of switches in brain state file.
        num_brain_state_switches_savedir = os.path.join(
            config_dict['git-results-basedir'], 'brain_states', f'k{num_basis_states:02d}'
        )
        num_brain_state_switches_filename = f'number_of_brain_state_switches_{model_name:s}.csv'
        brain_state_switch_counts_df = pd.read_csv(
            os.path.join(num_brain_state_switches_savedir, num_brain_state_switches_filename),
            index_col=0
        )  # (num_subjects, num_scans)

        brain_state_switch_counts_icc = compute_icc_scores_pingouin(
            brain_state_switch_counts_df.values,
            icc_type='ICC2',
        )

        # Shorten model names for plot.
        model_name = model_name.replace('SVWP_joint', 'WP-J')
        model_name = model_name.replace('DCC_joint', 'DCC-J')
        model_name = model_name.replace('SW_cross_validated', 'SW-CV')
        model_name = model_name.replace('_', '-')

        all_brain_state_switch_counts_iccs_df.loc[model_name, 'ICC'] = brain_state_switch_counts_icc

    all_brain_state_switch_counts_iccs_df.to_csv(
        os.path.join(num_brain_state_switches_savedir, 'number_of_brain_state_switches_ICC.csv'),
        float_format="%.2f"
    )
    logging.info(f"Saved ICC scores in '{num_brain_state_switches_savedir:s}'.")


if __name__ == "__main__":

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    num_brain_states_list = cfg['n-brain-states-list']

    for n_brain_states in num_brain_states_list:
        _compute_brain_state_switch_count_icc(
            config_dict=cfg,
            num_basis_states=n_brain_states
        )
