import os
import socket
import sys

from helpers.inference import run_adam_svwp, run_adam_vwp
from fcest.models.wishart_process import VariationalWishartProcess, SparseVariationalWishartProcess
import gpflow
from gpflow.ci_utils import ci_niter

from configs.configs import get_config_dict
from helpers.evaluation import leave_every_other_out_split
from helpers.inference import save_elbo_plot
from helpers.rockland import get_rockland_subjects, load_rockland_data


if __name__ == "__main__":

    pp_pipeline = 'custom_fsl_pipeline'

    model_name = sys.argv[1]       # 'VWP_joint' or 'SVWP_joint'
    data_split = sys.argv[2]       # 'all' or 'LEOO'
    repetition_time = sys.argv[3]  # in ms, either '1400' or '645'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset=repetition_time,
        hostname=socket.gethostname()
    )

    all_subjects_list = get_rockland_subjects(config_dict=cfg)
    num_subjects = len(all_subjects_list)

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
        print(f'\n> SUBJECT {i_subject+1:d} / {num_subjects:d}: {subject_filename:s}\n')

        # TODO: Check if model exists already.
        model_savedir = os.path.join(
            cfg['experiments-basedir'], pp_pipeline, 'saved_models', data_split, model_name
        )

        data_file = os.path.join(
            cfg['data-basedir'], pp_pipeline, 'node_timeseries', cfg['roi-list-name'], subject_filename
        )
        x, y = load_rockland_data(data_file)  # (N, 1), (N, D)
        if data_split == 'LEOO':
            x_train, x_test = leave_every_other_out_split(x)
            y_train, y_test = leave_every_other_out_split(y)
        else:
            x_train = x  # (N, 1)
            y_train = y  # (N, D)
        num_time_series = y_train.shape[1]

        k = gpflow.kernels.Matern52()
        match model_name:
            case 'VWP_joint':
                m = VariationalWishartProcess(
                    x_train, y_train,
                    nu=num_time_series,
                    kernel=k
                )
                maxiter = ci_niter(cfg['n-iterations'])
                logf = run_adam_vwp(
                    m,
                    maxiter,
                    log_interval=cfg['log-interval'],
                    log_dir=os.path.join(
                        cfg['experiments-basedir'], pp_pipeline, 'tensorboard_logs', data_split, model_name, subject_filename.removesuffix('.csv')
                    )
                )
            case 'SVWP_joint':
                m = SparseVariationalWishartProcess(
                    D=num_time_series,
                    Z=x[:cfg['n-inducing-points']],
                    nu=num_time_series,
                    kernel=k
                )
                maxiter = ci_niter(cfg['n-iterations-svwp'])
                logf = run_adam_svwp(
                    m,
                    data=(x_train, y_train),
                    iterations=maxiter,
                    log_interval=cfg['log-interval'],
#                    log_dir=tensorboard_logdir
                )
        m.save_model_params_dict(savedir=model_savedir, model_name=f"{subject_filename.removesuffix('.csv'):s}.json")
        save_elbo_plot(
            maxiter, cfg['log-interval'], logf,
            savedir=os.path.join(
                cfg['figures-basedir'], pp_pipeline, 'elbo_plots', cfg['roi-list-name'], data_split, model_name
            ),
            figure_name=f"{subject_filename.removesuffix('.csv'):s}.png"
        )
