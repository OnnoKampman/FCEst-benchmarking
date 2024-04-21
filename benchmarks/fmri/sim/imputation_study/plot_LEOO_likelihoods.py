import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import get_palette


def _plot_likelihoods_raincloud(
    config_dict: dict,
    test_likelihoods_df: pd.DataFrame,
    noise_type: str,
    palette: str,
    model_name: str = None,
    covs_type: str = None,
    data_split: str = 'LEOO',
    figures_savedir: str = None,
) -> None:
    """
    A "cloud", or smoothed version of a histogram, gives an idea of the distribution of scores.
    The "rain" are the individual data points, which can give an idea of outliers.

    Source:
        https://github.com/RainCloudPlots/RainCloudPlots
    """
    sns.set(style="whitegrid", font_scale=1.5)
    # plt.rcParams["font.family"] = 'serif'

    fig, ax = plt.subplots(
        figsize=config_dict['plot-likelihoods-figsize']
    )
    pt.RainCloud(
        data=test_likelihoods_df,
        ax=ax,
        palette=get_palette(test_likelihoods_df.columns),
        bw=0.2,  # sets the smoothness of the distribution
        width_viol=0.6,
        orient="h",  # "v" if you want a vertical plot
        move=0.22
    )
    plt.xlim([-3.35, -2.05])
    plt.xlabel('test log likelihood')
    plt.ylabel('TVFC estimator')

    if figures_savedir is not None:
        if model_name is not None:
            figure_name = f'{data_split:s}_{noise_type:s}_test_log_likelihoods_raincloud_{model_name:s}.pdf'
        elif covs_type is not None:
            figure_name = f'{data_split:s}_{noise_type:s}_test_log_likelihoods_raincloud_{covs_type:s}.pdf'
        else:
            raise
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figures_savedir:s}'.")
        plt.close()


def _plot_all_covs_structures_bar(
        config_dict: dict, test_likelihoods_mean_df: pd.DataFrame, test_likelihoods_sem_df: pd.DataFrame
) -> None:
    """
    Plot joint test likelihoods for all methods for all covariance structures.
    """
    raise NotImplementedError


if __name__ == '__main__':

    data_split = 'LEOO'  # leave-every-other-out

    data_set_name = sys.argv[1]    # 'd2', 'd3d', or 'd3s'
    experiment_data = sys.argv[2]  # e.g. 'N0200_T0003'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        experiment_data=experiment_data,
        hostname=socket.gethostname()
    )
    n_trials = int(experiment_data[-4:])
    test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')

    for noise_type in cfg['noise-types']:

        all_covs_models_test_likelihoods_mean_df = pd.DataFrame(
            np.nan,
            index=cfg['plot-covs-types'],
            columns=cfg['plot-models'],
        )
        all_covs_models_test_likelihoods_sem_df = pd.DataFrame(
            np.nan,
            index=cfg['plot-covs-types'],
            columns=cfg['plot-models'],
        )
        all_covs_models_test_likelihoods_mean_df.index = all_covs_models_test_likelihoods_mean_df.index.str.replace('periodic_1', 'periodic (slow)')
        all_covs_models_test_likelihoods_mean_df.index = all_covs_models_test_likelihoods_mean_df.index.str.replace('periodic_3', 'periodic (fast)')
        all_covs_models_test_likelihoods_mean_df.index = all_covs_models_test_likelihoods_mean_df.index.str.replace('_', ' ')
        all_covs_models_test_likelihoods_sem_df.index = all_covs_models_test_likelihoods_sem_df.index.str.replace('periodic_1', 'periodic (slow)')
        all_covs_models_test_likelihoods_sem_df.index = all_covs_models_test_likelihoods_sem_df.index.str.replace('periodic_3', 'periodic (fast)')
        all_covs_models_test_likelihoods_sem_df.index = all_covs_models_test_likelihoods_sem_df.index.str.replace('_', ' ')

        # Plot test likelihoods for all covariance structures for a single method.
        for model_name in cfg['plot-models']:
            likelihoods_filename = f'{data_split:s}_{noise_type:s}_likelihoods_{model_name:s}.csv'
            test_likelihoods_savepath = os.path.join(test_likelihoods_savedir, likelihoods_filename)
            if os.path.exists(test_likelihoods_savepath):
                logging.info(f"Loading '{test_likelihoods_savepath:s}'...")
                likelihoods_df = pd.read_csv(
                    test_likelihoods_savepath,
                    index_col=0,
                )  # (n_trials, n_all_covs_types)
                likelihoods_df = likelihoods_df.loc[:, cfg['plot-covs-types']]  # (n_trials, n_covs_types)

                # Update covs types labels for plots.
                likelihoods_df.columns = likelihoods_df.columns.str.replace('periodic_1', 'periodic (slow)')
                likelihoods_df.columns = likelihoods_df.columns.str.replace('periodic_3', 'periodic (fast)')
                likelihoods_df.columns = likelihoods_df.columns.str.replace('_', ' ')

                _plot_likelihoods_raincloud(
                    config_dict=cfg,
                    test_likelihoods_df=likelihoods_df,
                    noise_type=noise_type,
                    palette=cfg['plot-covs-types-palette'],
                    model_name=model_name,
                    figures_savedir=os.path.join(cfg['figures-basedir'], 'imputation_study')
                )

                likelihoods_mean_df = likelihoods_df.mean(axis=0)
                likelihoods_sem_df = likelihoods_df.sem(axis=0)
                all_covs_models_test_likelihoods_mean_df.loc[:, model_name] = likelihoods_mean_df
                all_covs_models_test_likelihoods_sem_df.loc[:, model_name] = likelihoods_sem_df

            else:
                logging.warning(f"{test_likelihoods_savepath:s} not found.")

        print(f'\n\n[{noise_type:s}] ALL MODELS AND COVARIANCE STRUCTURES:')
        print(all_covs_models_test_likelihoods_mean_df.round(3))
        print(all_covs_models_test_likelihoods_sem_df.round(3))
        all_covs_models_test_likelihoods_mean_df.to_csv(
            os.path.join(test_likelihoods_savedir, f'LEOO_{noise_type:s}_likelihoods_all_methods_mean.csv'),
            float_format="%.3f"
        )
        all_covs_models_test_likelihoods_sem_df.to_csv(
            os.path.join(test_likelihoods_savedir, f'LEOO_{noise_type:s}_likelihoods_all_methods_sem.csv'),
            float_format="%.3f"
        )

        # Plot test likelihoods for all methods for all covariance structures.
        _plot_all_covs_structures_bar(
            config_dict=cfg,
            test_likelihoods_mean_df=all_covs_models_test_likelihoods_mean_df,
            test_likelihoods_sem_df=all_covs_models_test_likelihoods_sem_df
        )

        # Plot test likelihoods for all methods for a single covariance structure.
        for covs_type in cfg['plot-covs-types']:
            covs_type_df = pd.DataFrame(
                np.nan,
                index=np.arange(n_trials), columns=cfg['plot-models']
            )
            for model_name in cfg['plot-models']:
                likelihoods_filename = f'{data_split:s}_{noise_type:s}_likelihoods_{model_name:s}.csv'
                test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
                test_likelihoods_savepath = os.path.join(test_likelihoods_savedir, likelihoods_filename)
                if os.path.exists(test_likelihoods_savepath):
                    likelihoods_df = pd.read_csv(
                        test_likelihoods_savepath, index_col=0
                    )  # (n_trials, n_train_covs_types)
                    if not likelihoods_df.shape == (n_trials, len(cfg['all-covs-types'])):
                        logging.warning("Unexpected number of covariance structures found.")
                        print(likelihoods_df.round(3))
                    likelihoods_df = likelihoods_df.loc[:, cfg['plot-covs-types']]
                    # print(likelihoods_df)
                    covs_type_df.loc[:, model_name] = likelihoods_df.loc[:, covs_type]
                else:
                    logging.warning(f"{test_likelihoods_savepath:s} not found.")

            # Update model names labels for plots.
            covs_type_df.columns = covs_type_df.columns.str.replace('SVWP_joint', 'WP')
            covs_type_df.columns = covs_type_df.columns.str.replace('SW_cross_validated', 'SW-CV')
            if data_set_name == 'd2':
                covs_type_df.columns = covs_type_df.columns.str.replace('_joint', '')
            else:
                covs_type_df.columns = covs_type_df.columns.str.replace('_joint', '-J')
                covs_type_df.columns = covs_type_df.columns.str.replace('_bivariate_loop', '-BL')
            covs_type_df.columns = covs_type_df.columns.str.replace('_', ' ')

            _plot_likelihoods_raincloud(
                config_dict=cfg,
                test_likelihoods_df=covs_type_df,
                noise_type=noise_type,
                palette=cfg['plot-methods-palette'],
                covs_type=covs_type,
                figures_savedir=os.path.join(cfg['figures-basedir'], 'imputation_study')
            )
