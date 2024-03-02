import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.data import reorder_ica_components

sns.set(style="whitegrid", font_scale=1.3)
plt.rcParams["font.family"] = 'serif'


def _plot_edgewise_imputation_benchmark_scores(
        config_dict: dict, edgewise_likelihoods: pd.DataFrame, model_name: str,
        figure_savedir: str = None
) -> None:
    n_time_series = edgewise_likelihoods.shape[0]
    if data_dimensionality == 'd15':
        edgewise_likelihoods, new_rsn_names = reorder_ica_components(
            config_dict=config_dict,
            original_matrix=edgewise_likelihoods.values,
            n_time_series=n_time_series,
            # lower_triangular=True
        )
    else:
        # TODO: add RSN names map for d50
        edgewise_likelihoods = edgewise_likelihoods.values
        new_rsn_names = np.arange(n_time_series)

    # Define mask for upper triangular values.
    mask = np.zeros_like(edgewise_likelihoods)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        edgewise_likelihoods,
        cmap='jet',
        mask=mask,
        vmin=-4.0,
        vmax=-2.0,
        xticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        yticklabels=new_rsn_names if data_dimensionality == 'd15' else False,
        square=True,
        cbar_kws={
            'label': "test log likelihood",
            'shrink': 0.6
        }
    )
    # plt.tight_layout()
    if figure_savedir is not None:
        figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_edgewise_{model_name:s}.pdf'
        if not os.path.exists(figure_savedir):
            os.makedirs(figure_savedir)
        plt.savefig(
            os.path.join(figure_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name:s}' in '{figure_savedir:s}'.")
        plt.close()


def _plot_performance_difference(config_dict: dict, edgewise_likelihoods: (pd.DataFrame, pd.DataFrame)) -> None:
    """
    Plots difference in performance between two methods.
    """
    edgewise_likelihoods_first_method, new_rsn_names = reorder_ica_components(
        original_matrix=edgewise_likelihoods[0].values,
        n_time_series=edgewise_likelihoods[0].shape[0],
        config_dict=config_dict,
        # lower_triangular=True
    )
    edgewise_likelihoods_second_method, new_rsn_names = reorder_ica_components(
        original_matrix=edgewise_likelihoods[1].values,
        n_time_series=edgewise_likelihoods[1].shape[0],
        config_dict=config_dict,
        # lower_triangular=True
    )
    performance_differences = (edgewise_likelihoods_first_method - edgewise_likelihoods_second_method) / abs(edgewise_likelihoods_second_method)
    print(performance_differences)

    # Define mask for upper triangular values.
    mask = np.zeros_like(performance_differences)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        performance_differences,
        cmap='RdYlGn',
        mask=mask,
        vmin=-0.08,
        vmax=0.08,
        xticklabels=new_rsn_names,
        # xticklabels=False,
        yticklabels=new_rsn_names,
        # yticklabels=False,
        square=True,
        cbar_kws={
            'label': "relative outperformance test log likelihood"
        }
    )
    # plt.tight_layout()

    figure_name = f'{data_split:s}_{experiment_dimensionality:s}_test_log_likelihoods_edgewise_SVWP_joint_sFC.pdf'
    savedir = os.path.join(config_dict['figures-basedir'], 'imputation_study')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(
        os.path.join(savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved figure '{figure_name:s}' in '{savedir:s}'.")
    plt.close()


if __name__ == '__main__':

    data_split = 'LEOO'  # leave-every-other-out

    data_dimensionality = sys.argv[1]        # 'd15', 'd50'
    experiment_dimensionality = sys.argv[2]  # 'multivariate', 'bivariate'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )

    all_likelihoods_df = pd.DataFrame()
    test_likelihoods_savedir = os.path.join(cfg['git-results-basedir'], 'imputation_study')
    for model_name in cfg['plot-models']:
        likelihoods_filename = f'{data_split:s}_{experiment_dimensionality:s}_likelihoods_{model_name:s}_edgewise.csv'
        likelihoods_df = pd.read_csv(
            os.path.join(test_likelihoods_savedir, likelihoods_filename),
            index_col=0
        )  # (D, D)
        print(likelihoods_df)
        _plot_edgewise_imputation_benchmark_scores(
            config_dict=cfg,
            edgewise_likelihoods=likelihoods_df,
            model_name=model_name,
            figure_savedir=os.path.join(cfg['figures-basedir'], 'imputation_study')
        )

    # WP outperformance over sFC.
    likelihoods_filename_wp = f'{data_split:s}_{experiment_dimensionality:s}_likelihoods_SVWP_joint_edgewise.csv'
    likelihoods_wp_df = pd.read_csv(
        os.path.join(test_likelihoods_savedir, likelihoods_filename_wp),
        index_col=0
    )  # (D, D)
    likelihoods_filename_sfc = f'{data_split:s}_{experiment_dimensionality:s}_likelihoods_sFC_edgewise.csv'
    likelihoods_sfc_df = pd.read_csv(
        os.path.join(test_likelihoods_savedir, likelihoods_filename_sfc),
        index_col=0
    )  # (D, D)
    _plot_performance_difference(
        config_dict=cfg,
        edgewise_likelihoods=(likelihoods_wp_df, likelihoods_sfc_df)
    )
