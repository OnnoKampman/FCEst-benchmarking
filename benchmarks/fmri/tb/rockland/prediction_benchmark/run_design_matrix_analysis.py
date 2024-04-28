import logging
import os
from pprint import pprint
import socket

from fcest.helpers.data import to_3d_format
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm
from nilearn.plotting import plot_design_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels
import statsmodels.api as sm

from configs.configs import get_config_dict
from helpers.rockland import get_rockland_subjects, get_edges_names


def _get_beta(
    config_dict: dict,
    mean_tvfc_estimates: np.array,
    design_matrix_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each TVFC estimation method, run GLM and extract beta (the estimated coefficients).
    Generate trends:
        https://github.com/asoroosh/SALSA/blob/327cec9f53449bedf662538c2b9919417870f2b1/utils/Trend/GenerateTemporalTrends.m

    Sources:
        https://nilearn.github.io/dev/modules/generated/nilearn.glm.first_level.FirstLevelModel.html

    Parameters
    ----------
    :param config_dict:
    :param mean_tvfc_estimates:
        TVFC estimates for edges of interest, of shape (n_edges_of_interest, N)
    :param design_matrix_df:
        Design matrix DataFrame of shape (n_regressors, n_time_steps).
    :return:
        DataFrame of shape (n_edges_of_interest, n_regressors).
    """
    Y = mean_tvfc_estimates.T  # (N, n_edges_of_interest)
    X = design_matrix_df.values.T  # (N, n_regressors/n_predictors)
    print('Y:', Y.shape, 'X:', X.shape)
    assert len(Y) == len(X)

    labels, results_dict = run_glm(
        Y=Y,
        X=X,
        noise_model='ar1',
        verbose=1
    )  # array of shape (n_edges_of_interest, ), dict with labels as values ()
    print(labels)
    pprint(results_dict)
    print('')

    beta_df = pd.DataFrame(
        np.nan,
        index=edge_names,
        columns=design_matrix_df.index
    )  # (n_edges_of_interest, n_regressors)

    # Loop over RegressionResults objects.
    for key, regression_results in results_dict.items():
        print('')
        print(key)  # TODO: what is the key here?
        print(regression_results.MSE)
        print(regression_results.r_square)
        logging.info(f"MSE: {regression_results.MSE[0]:.2f} | r_square: {regression_results.r_square[0]:.2f}")

        # Beta/theta parameters.
        print(regression_results.theta)  # these are the beta values, of shape (n_regressors=6, ?)
        # print(value.Y)
        # print(value.model)
        # print(value.dispersion)
        # print(value.nuisance)
        assert regression_results.df_total == n_time_steps
        print(regression_results.df_model)
        print(regression_results.df_residuals)

        # Significance.
        print('\nConfidence interval:')
        print(regression_results.conf_int(cols=(1, 2)))

        edge_indices = np.atleast_1d(np.squeeze(np.argwhere(labels == key)))
        beta_df.iloc[edge_indices, :] = regression_results.theta.T
        print(beta_df.round(3))

    # Alternative method #1.
    print('\nAlternative method 1:')
    labels, results_dict = run_glm(
        Y=Y,
        X=X,
        noise_model='ols',
        verbose=1
    )  # array of shape (n_edges_of_interest, ), dict with labels as values ()
    print(labels)
    pprint(results_dict)
    alt_beta = results_dict[0.0].theta.T  # (5, 4)
    print(alt_beta)
    results_conf_int = results_dict[0.0].conf_int(cols=(0, 1, 2)).T  # np.array of shape (5, 2, 4)
    print(results_conf_int)
    print(results_conf_int.shape)
    results_alt_se = (results_conf_int[:, 1, :] - results_conf_int[:, 0, :]) / (2 * 1.96)  # (5, 4)
    print(results_alt_se)
    results_conf_int_test_statistic = alt_beta / results_alt_se
    alt_pvals = np.exp(-0.717 * results_conf_int_test_statistic - 0.416 * results_conf_int_test_statistic**2)
    alt_pvals_df = pd.DataFrame(
        alt_pvals,
        index=edge_names,
        columns=design_matrix_df.index
    )  # (n_edges_of_interest, n_regressors)
    print(alt_pvals_df.round(3))
    print('')

    print("\nFINAL BETA AND P-VALUES:\n")
    print(beta_df.round(3))
    print(alt_pvals_df.round(3))

    # Alternative method #2: statsmodels.
    # Statsmodels expects Y as a 1-dimensional array, so we loop over all edges.
    beta_df_sm = pd.DataFrame(
        np.nan,
        index=edge_names,
        columns=design_matrix_df.index
    )  # (n_edges_of_interest, n_regressors)
    beta_pval_df_sm = pd.DataFrame(
        np.nan,
        index=edge_names,
        columns=design_matrix_df.index
    )  # (n_edges_of_interest, n_regressors)
    X = sm.add_constant(X)  # (N, n_regressors)
    print(X.shape)
    assert len(X.shape) == 2
    for i_edge in range(Y.shape[1]):
        try:
            gamma_model = sm.GLM(
                endog=Y[:, i_edge],
                exog=X,
                # family=sm.families.Gamma()
            )
            gamma_results = gamma_model.fit()
        except statsmodels.tools.sm_exceptions.PerfectSeparationError:
            logging.warning("Skipping model.")
            continue
        except ValueError:
            logging.warning("Skipping model.")
            continue
        # print(gamma_results.summary())
        beta_df_sm.iloc[i_edge, :] = gamma_results.params
        beta_pval_df_sm.iloc[i_edge, :] = gamma_results.pvalues

    print("\nSTATSMODELS BETA AND P-VALUES:\n")
    print(beta_df_sm.round(3))
    print(beta_pval_df_sm.round(3))

    return beta_df_sm, beta_pval_df_sm


def _generate_design_matrix(
    config_dict: dict,
    hrf_model: str = 'glover',
) -> pd.DataFrame:
    """
    Generate design matrix.

    Sources:
        https://github.com/asoroosh/SALSA/blob/327cec9f53449bedf662538c2b9919417870f2b1/NullRealfMRI/Img/NullSim_Img_bmrc.m#L132
        https://github.com/asoroosh/SALSA/blob/327cec9f53449bedf662538c2b9919417870f2b1/mis/GenerateED.m

    Parameters
    ----------
    :param config_dict:
    :param hrf_model:
    :return:
        design matrix DataFrame of shape (n_regressors, n_time_steps).
    """
    frame_times = np.arange(n_time_steps) * config_dict['repetition-time']  # (n_frames, )
    n_polynomials = _determine_n_polynomials(config_dict['repetition-time'], config_dict['n-time-steps'])
    drift_order = n_polynomials - 1

    # We have only one type of trial: the checkerboard visual stimulus.
    conditions = ['rest', 'stim', 'rest', 'stim', 'rest', 'stim', 'rest']
    duration = [20., 20., 20., 20., 20., 20., 20.]  # condition durations in seconds
    onsets = [0., 20., 40., 60., 80., 100., 120.]  # corresponding onset times in seconds
    events = pd.DataFrame({
        'trial_type': conditions,
        'onset': onsets,
        'duration': duration}
    )
    print(events)

    design_matrix_df = make_first_level_design_matrix(
        frame_times,
        events=events,
        drift_model='polynomial',
        drift_order=drift_order,
        hrf_model=hrf_model
    ).T  # (n_regressors, N)
    design_matrix_df = design_matrix_df.drop(index='rest')  # exclude 'rest' as a regressor
    design_matrix_df.index = design_matrix_df.index.str.replace('constant', 'offset')
    design_matrix_df.index = design_matrix_df.index.str.replace('drift_1', 'trend')
    print(design_matrix_df)
    _plot_design_matrix(
        config_dict=config_dict,
        df=design_matrix_df,
        figure_name='design_matrix.pdf'
    )

    return design_matrix_df


def _determine_n_polynomials(repetition_time: float, n_time_steps: int) -> int:
        return 1 + np.floor(repetition_time * n_time_steps / 150)


def _plot_design_matrix(
    config_dict: dict,
    df: pd.DataFrame,
    figure_name: str,
) -> None:
    """
    Plot the design matrix.

    Sources:
        https://nilearn.github.io/dev/auto_examples/04_glm_first_level/plot_design_matrix.html

    Parameters
    ----------
    :param df:
        DataFrame of shape (n_regressors, N).
    """
    sns.set(style="white", font_scale=1.5)
    plt.rcParams["font.family"] = 'serif'

    figures_savedir = os.path.join(
        config_dict['figures-basedir'], pp_pipeline, 'prediction_benchmark', config_dict['roi-list-name'], data_split
    )
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    ax = sns.heatmap(
        df.T,
        cmap='gray'
    )
    ax.set_xlabel('regressors')
    ax.set_ylabel('time [volumes]')
    plt.savefig(
        os.path.join(figures_savedir, figure_name),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved design matrix in '{figures_savedir:s}'.")
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_design_matrix(
        df.T,
        ax=ax
    )
    fig.savefig(
        os.path.join(figures_savedir, figure_name.replace('.pdf', '_nilearn.pdf')),
        format='pdf',
        bbox_inches='tight'
    )
    logging.info(f"Saved NiLearn design matrix in '{figures_savedir:s}'.")
    plt.close()


if __name__ == "__main__":

    data_set_name = 'rockland'
    data_split = 'all'
    metric = 'correlation'
    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name=data_set_name,
        subset='645',
        hostname=socket.gethostname()
    )

    all_subjects_list = get_rockland_subjects(config_dict=cfg)

    brain_regions_of_interest = cfg['roi-list']
    edges_to_plot_indices = cfg['roi-edges-list']
    edge_names = get_edges_names(cfg)
    print(edge_names)
    assert len(edge_names) == len(edges_to_plot_indices)

    n_time_steps = n_scans = n_frames = n_time_points = cfg['n-time-steps']
    n_time_series = len(brain_regions_of_interest)  # D

    design_matrix = _generate_design_matrix(
        config_dict=cfg
    )  # (n_regressors, N)

    for model_name in cfg['stimulus-prediction-models']:
        print(f'\n> Model {model_name:s}\n')

        average_tvfc_estimates = np.zeros(shape=(n_time_steps, n_time_series, n_time_series))  # (N, D, D)
        tvfc_estimates_savedir = os.path.join(
            cfg['experiments-basedir'], pp_pipeline, 'TVFC_estimates', data_split, metric, model_name
        )
        for i_subject, subject_filename in enumerate(all_subjects_list):
            # logging.info(f'> Subject {i_subject+1:d} / {len(all_subjects_list):d}: {subject_filename:s}')
            estimated_tvfc_df = pd.read_csv(
                os.path.join(tvfc_estimates_savedir, subject_filename),
                index_col=0
            )  # (D*D, N)
            subject_tvfc_estimates = to_3d_format(estimated_tvfc_df.values)  # (N, D, D)
            average_tvfc_estimates += subject_tvfc_estimates
        average_tvfc_estimates /= len(all_subjects_list)  # (N, D, D)
        average_tvfc_estimates_edges_of_interest = np.array([
            average_tvfc_estimates[:, indices[0], indices[1]] for indices in edges_to_plot_indices
        ])  # (n_edges_of_interest, N)
        assert average_tvfc_estimates_edges_of_interest.shape == (len(edges_to_plot_indices), n_time_steps)

        betas_df, betas_pvals_df = _get_beta(
            cfg,
            mean_tvfc_estimates=average_tvfc_estimates_edges_of_interest,
            design_matrix_df=design_matrix
        )
        betas_savedir = os.path.join(cfg['git-results-basedir'], 'prediction_benchmark')
        if not os.path.exists(betas_savedir):
            os.makedirs(betas_savedir)
        betas_df.to_csv(
            os.path.join(betas_savedir, f'betas_df_{model_name:s}.csv'),
            float_format="%.3f"
        )
        betas_pvals_df.to_csv(
            os.path.join(betas_savedir, f'betas_pvals_df_{model_name:s}.csv'),
            float_format="%.3f"
        )
        betas_pvals_bonferroni_df = len(betas_pvals_df) * betas_pvals_df
        betas_pvals_bonferroni_df.to_csv(
            os.path.join(betas_savedir, f'betas_pvals_Bonferroni_df_{model_name:s}.csv'),
            float_format="%.3f"
        )
        logging.info(f"Saved '{model_name:s}' stimulus design matrix analysis benchmark results in '{betas_savedir:s}'.")
