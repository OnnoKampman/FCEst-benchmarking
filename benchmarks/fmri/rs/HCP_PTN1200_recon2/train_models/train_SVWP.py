import logging
import os
import socket
import sys

from fcest.models.wishart_process import SparseVariationalWishartProcess
import gpflow
from gpflow.ci_utils import ci_niter

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.hcp import get_human_connectome_project_subjects, load_human_connectome_project_data
from helpers.inference import run_adam_svwp, save_elbo_plot


if __name__ == "__main__":

    model_name = 'SVWP_joint'  # TODO: the bivariate loop SVWP has not been implemented yet

    data_dimensionality = sys.argv[1]  # 'd15' or 'd50'
    data_split = sys.argv[2]  # 'all' or 'LEOO'
    experiment_dimensionality = sys.argv[3]  # 'multivariate' or 'bivariate'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_human_connectome_project_subjects(data_dir=cfg['data-dir'])
    n_subjects = len(all_subjects_list)

    # Allow for local and CPU cluster training.
    # When running on the Hivemind with SLURM, only one model is trained here.
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

        data_file = os.path.join(cfg['data-dir'], subject_filename)
        for scan_id in cfg['scan-ids']:
            print(f'\nSCAN {scan_id+1:d} / 4\n')

            # Check if model exists already.
            model_savedir = os.path.join(
                cfg['experiments-basedir'], 'saved_models', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, model_name
            )
            model_savepath = os.path.join(model_savedir, f"{subject_filename.removesuffix('.txt'):s}.json")
            if os.path.exists(model_savepath):
                logging.info(f"Skipping training: Existing model found in '{model_savedir:s}'.")
                continue

            x, y = load_human_connectome_project_data(
                data_file, scan_id=scan_id, verbose=False
            )  # (N, 1), (N, D)
            n_time_series = y.shape[1]

            # TODO: pick two time series at random if 'experiment_dimensionality' == 'bivariate'?
            if experiment_dimensionality == 'bivariate':
                chosen_indices = [0, 1]
                # chosen_indices_df = cfg['chosen-indices']
                # chosen_indices = chosen_indices_df.loc[subject, scan_id]
                y = y[:, chosen_indices]
                n_time_series = y.shape[1]
                print('y', y.shape)

            match data_split:
                case "LEOO":
                    x_train, _ = leave_every_other_out_split(x)  # (N/2, 1), (N/2, 1)
                    y_train, _ = leave_every_other_out_split(y)  # (N/2, D), (N/2, D)
                case "all":
                    x_train = x  # (N, 1)
                    y_train = y  # (N, D)
                case _:
                    logging.error("Data split not recognized.")

            k = gpflow.kernels.Matern52()
            m = SparseVariationalWishartProcess(
                D=n_time_series,
                Z=x_train[:cfg['n-inducing-points']],
                nu=n_time_series,
                kernel=k
            )
            maxiter = ci_niter(cfg['n-iterations'])
            logf = run_adam_svwp(
                m,
                data=(x_train, y_train),
                iterations=maxiter,
                log_interval=cfg['log-interval-svwp']
            )
            m.save_model_params_dict(
                savedir=model_savedir, model_name=f"{subject_filename.removesuffix('.txt'):s}.json"
            )
            figures_savedir = os.path.join(
                cfg['figures-basedir'], 'elbo', f'scan_{scan_id:d}', data_split
            )
            if not os.path.exists(figures_savedir):
                os.makedirs(figures_savedir)
            save_elbo_plot(
                maxiter,
                cfg['log-interval-svwp'],
                logf,
                savedir=figures_savedir,
                figure_name=f"{subject_filename.removesuffix('.txt')}_{model_name:s}.png"
            )
