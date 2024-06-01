import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.brain_states import extract_number_of_brain_state_switches


def _save_number_of_brain_state_switches(
    config_dict: dict,
    num_basis_states: int,
    connectivity_metric: str = 'correlation',
) -> None:
    for model_name in config_dict['models-brain-state-analysis']:
        model_number_of_switches_df = pd.DataFrame()
        for scan_id in config_dict['scan-ids']:
            brain_states_savedir = os.path.join(
                config_dict['git-results-basedir'], 'brain_states', f'k{num_basis_states:02d}', f'scan_{scan_id:d}'
            )

            # Load brain state assignment file.
            brain_state_assignment_df = pd.read_csv(
                os.path.join(brain_states_savedir, f'{connectivity_metric:s}_brain_state_assignments_{model_name:s}.csv'),
                index_col=0
            )  # (n_subjects, N)

            change_point_counts = extract_number_of_brain_state_switches(brain_state_assignment_df)  # (n_subjects, )
            model_number_of_switches_df[f'scan_{scan_id:d}'] = change_point_counts

        print(model_number_of_switches_df)

        # Save number of brain state switches for this model.
        n_brain_state_switches_savedir = os.path.join(
            config_dict['git-results-basedir'], 'brain_states', f'k{num_basis_states:02d}'
        )
        n_brain_state_switches_filename = f'number_of_brain_state_switches_{model_name:s}.csv'
        model_number_of_switches_df.to_csv(
            os.path.join(n_brain_state_switches_savedir, n_brain_state_switches_filename)
        )
        logging.info(f"Saved number of brain state switches {model_name:s} in '{n_brain_state_switches_savedir:s}'.")


if __name__ == "__main__":

    data_dimensionality = sys.argv[1]  # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_brain_states_list = cfg['n-brain-states-list']
    for n_brain_states in n_brain_states_list:
        _save_number_of_brain_state_switches(
            config_dict=cfg,
            num_basis_states=n_brain_states
        )
