import logging
import os
import socket

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.configs import get_config_dict
from helpers.figures import set_size
from helpers.rockland import get_edges_names


def _plot_glm_beta_heatmap(
    config_dict: dict, betas_df: pd.DataFrame, model_name: str, 
    figures_savedir: str = None
) -> None:
    sns.set(style="whitegrid", font_scale=0.6)
    plt.rcParams["font.family"] = 'serif'

    fig, ax = plt.subplots(figsize=set_size(fraction=0.47))
    sns.heatmap(
        betas_df,
        ax=ax,
        cmap='jet',
        annot=True,
        fmt='.2f',
        linewidth=1.0,
        # robust=True,
        vmin=-0.5,
        vmax=1.0,
        cbar_kws={
            'label': "beta",
            'shrink': 0.6
        }
    )
    ax.set_xlabel('regressors')
    ax.set_ylabel('edges of interest')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # ax.figure.axes[-1].tick_params(labelsize=12)  # TODO: test this

    if figures_savedir is not None:
        figure_name = f'GLM_beta_{model_name:s}.pdf'
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        fig.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved GLM beta matrix in '{figures_savedir:s}'.")
        plt.close()


def _plot_glm_beta_bar(
    config_dict: dict, stimulus_betas_df: pd.DataFrame, 
    figures_savedir: str = None
) -> None:
    sns.set(style="whitegrid", font_scale=1.1)
    # plt.rcParams["font.family"] = 'serif'
    sns.set_palette('Set3')

    fig, ax = plt.subplots(
        figsize=(12, 4)
        # figsize=set_size(fraction=1.0)
    )
    stimulus_betas_df.plot.bar(
        ax=ax,
        # cmap='Set3',
    )
    plt.xticks(rotation=0)
    plt.xlabel('TVFC estimator')
    plt.ylabel('beta parameters from GLM')
    plt.legend(title='ROI edge', alignment='left')

    if figures_savedir is not None:
        figure_name = 'GLM_stimulus_beta.pdf'
        if not os.path.exists(figures_savedir):
            os.makedirs(figures_savedir)
        plt.savefig(
            os.path.join(figures_savedir, figure_name),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved GLM stimulus betas in '{figures_savedir:s}'.")
        plt.close()


def _clean_up_regressor_names(df: pd.DataFrame) -> pd.DataFrame:
    # print(df.round(3))
    df.columns = df.columns.str.replace('constant', 'offset')
    df.columns = df.columns.str.replace('drift_1', 'trend')
    # print(df.round(3))
    return df


def _clean_up_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change model names for plot.
    """
    df.index = df.index.str.replace('SVWP_joint', 'WP')
    df.index = df.index.str.replace('_joint', '-J')
    df.index = df.index.str.replace('DCC_bivariate_loop', 'DCC-BL')
    df.index = df.index.str.replace('SW_cross_validated', 'SW-CV')
    df.index = df.index.str.replace('_', '-')
    return df


if __name__ == "__main__":
    
    # TODO: add significance stars to bar plot

    data_set_name = 'rockland'
    data_split = 'all'
    metric = 'correlation'
    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        subset='645',
        hostname=socket.gethostname()
    )

    all_glm_betas_df = pd.DataFrame(
        columns=get_edges_names(cfg)
    )
    for model_name in cfg['stimulus-prediction-models']:
        print(f'\n> Model: {model_name:s}\n')

        betas_savedir = os.path.join(cfg['git-results-basedir'], 'prediction_benchmark')
        betas_df = pd.read_csv(
            os.path.join(betas_savedir, f'betas_df_{model_name:s}.csv'),
            index_col=0
        )  # (n_edges, n_regressors)
        betas_df = _clean_up_regressor_names(betas_df)
        logging.info(f"Loaded '{model_name:s}' stimulus GLM analysis benchmark results from '{betas_savedir:s}'.")

        _plot_glm_beta_heatmap(
            config_dict=cfg,
            betas_df=betas_df,
            model_name=model_name,
            figures_savedir=os.path.join(
                cfg['figures-basedir'], pp_pipeline, 'prediction_benchmark', cfg['roi-list-name'], data_split
            )
        )
        all_glm_betas_df.loc[model_name, :] = betas_df['stim']
        print(all_glm_betas_df)

    all_glm_betas_df = all_glm_betas_df.loc[cfg['plot-stimulus-prediction-models'], :]
    all_glm_betas_df = _clean_up_model_names(all_glm_betas_df)
    print(all_glm_betas_df)

    _plot_glm_beta_bar(
        config_dict=cfg,
        stimulus_betas_df=all_glm_betas_df,
        figures_savedir=os.path.join(
            cfg['figures-basedir'], pp_pipeline, 'prediction_benchmark', cfg['roi-list-name'], data_split
        )
    )
