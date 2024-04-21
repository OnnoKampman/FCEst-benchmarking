import logging
import os
import socket
import sys

from fcest.helpers.inference import run_adam_svwp, run_adam_vwp
from fcest.models.wishart_process import SparseVariationalWishartProcess, VariationalWishartProcess
import gpflow
from gpflow.ci_utils import ci_niter

from configs.configs import get_config_dict
from helpers.data import load_data
from helpers.evaluation import leave_every_other_out_split
from helpers.inference import save_elbo_plot


if __name__ == "__main__":

    hostname = socket.gethostname()
    print('\nHostname:', hostname)

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd{%d}s'
    experiment_data = sys.argv[2]  # e.g. 'N0200_T0100'
    model_name = sys.argv[3]       # 'VWP_joint', 'SVWP_joint', 'VWP', or 'SVWP'
    data_split = sys.argv[4]       # 'all', or 'LEOO'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=hostname
    )
    n_trials = int(experiment_data[-4:])
    assert os.path.exists(os.path.join(cfg['data-dir']))

    # Allow for local and CPU cluster training.
    # When running on the Hivemind, only one model is trained here.
    if hostname == 'hivemind':
        try:
            i_trial = os.environ['SLURM_ARRAY_TASK_ID']
            # i_trial = os.environ['SLURM_ARRAY_JOB_ID']
            i_trial = int(i_trial) - 1  # to make zero-index
            print('SLURM trial ID', i_trial)

            assert len(sys.argv) == 7
            noise_type = sys.argv[5]
            covs_type = sys.argv[6]

            i_trials = [i_trial]
            noise_types = [noise_type]
            covs_types = [covs_type]
        except KeyError:
            i_trials = range(n_trials)
            noise_types = cfg['noise-types']
            covs_types = cfg['all-covs-types']
    else:
        print('Running locally...')
        i_trials = range(n_trials)
        noise_types = cfg['noise-types']
        covs_types = cfg['all-covs-types']

    for noise_type in noise_types:

        for covs_type in covs_types:

            for i_trial in i_trials:

                print('\n----------')
                print(f'Trial      {i_trial}')
                print(f'covs_type  {covs_type:s}')
                print(f'noise_type {noise_type:s}', '\n----------\n')
                data_file = os.path.join(
                    cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                    f'{covs_type:s}_covariance.csv'
                )
                if not os.path.exists(data_file):
                    if covs_type == 'boxcar':
                        data_file = os.path.join(
                            cfg['data-dir'], noise_type, f'trial_{i_trial:03d}',
                            'checkerboard_covariance.csv'
                        )

                # Check if model already exists.
                model_savedir = os.path.join(
                    cfg['experiments-basedir'], noise_type, data_split,
                    f'trial_{i_trial:03d}', model_name
                )
                model_filename = f'{covs_type:s}.json'
                if os.path.exists(os.path.join(model_savedir, model_filename)):
                    logging.info(f"Skipping training: Found existing model '{model_filename:s}' in '{model_savedir}'.")
                    continue

                tensorboard_logdir = os.path.join(
                    cfg['experiments-basedir'], 'tensorboard_logs', noise_type, data_split,
                    f'trial_{i_trial:03d}', model_name, covs_type
                )

                x, y = load_data(data_file, verbose=False)  # (N, 1), (N, D)
                n_time_series = y.shape[1]

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
                match model_name:
                    case 'VWP' | 'VWP_joint':
                        m = VariationalWishartProcess(
                            x_train, y_train,
                            nu=n_time_series,
                            kernel=k
                        )
                        maxiter = ci_niter(cfg['n-iterations-vwp'])
                        logf = run_adam_vwp(
                            m, maxiter,
                            log_interval=cfg['log-interval'],
                            log_dir=tensorboard_logdir
                        )
                    case 'SVWP' | 'SVWP_joint':
                        m = SparseVariationalWishartProcess(
                            D=n_time_series,
                            Z=x[:cfg['n-inducing-points']],  # TODO: select equally spaced X for Z init?
                            # Z=x[::int(len(x) / n_inducing_points)],
                            nu=n_time_series,
                            kernel=k,
                            train_additive_noise=True
                        )
                        maxiter = ci_niter(cfg['n-iterations-svwp'])
                        logf = run_adam_svwp(
                            m,
                            data=(x_train, y_train),
                            iterations=maxiter,
                            log_interval=cfg['log-interval'],
                            log_dir=tensorboard_logdir
                        )
                    case _:
                        raise NotImplementedError(f"Model name '{model_name:s}' not recognized.")
                m.save_model_params_dict(savedir=model_savedir, model_name=model_filename)
                save_elbo_plot(
                    maxiter, cfg['log-interval'], logf,
                    savedir=os.path.join(
                        cfg['figures-basedir'], 'elbo', noise_type, data_split, f'trial_{i_trial:03d}'
                    ),
                    figure_name=f'elbo_{covs_type:s}_{model_name:s}.png'
                )
