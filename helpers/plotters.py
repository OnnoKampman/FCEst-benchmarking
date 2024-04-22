import logging
import os

from fcest.models.sliding_windows import SlidingWindows
from fcest.models.wishart_process import VariationalWishartProcess, SparseVariationalWishartProcess
from fcest.helpers.array_operations import to_correlation_structure
from fcest.helpers.data import to_3d_format
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.array_operations import convert_to_seconds, convert_to_minutes


def plot_node_timeseries(
        config_dict: dict, x_plot: np.array, y_locations: np.array, figures_savedir: str = None
) -> None:
    """
    Plots the synthetic time series.

    :param config_dict:
    :param x_plot:
    :param y_locations:
    :param figures_savedir:
    """
    sns.set(style="whitegrid", font_scale=1.8)
    # plt.rcParams["font.family"] = 'serif'
    font = {
        'family': 'Times New Roman',
        #     'weight': 'normal',
        #     'size': 14
    }
    plt.rc('font', **font)
    figure_name_time_series = "time_series.pdf"

    n_time_series = y_locations.shape[1]

    plt.figure(figsize=config_dict['plot-time-series-figsize'])
    for i_time_series in range(n_time_series):
        plt.subplot(n_time_series, 1, i_time_series+1)
        plt.plot(
            x_plot, y_locations[:, i_time_series], 'x-',
            markersize=0, label=f"TS_{i_time_series+1}"
        )
        # plt.xlim(config_dict['plot-time-series-xlim'])
        plt.ylim(config_dict['plot-time-series-ylim'])
        if not i_time_series == (n_time_series - 1):
            plt.gca().get_xaxis().set_ticklabels([])
    plt.xlabel("time [a.u.]")
    # plt.tight_layout()

    if figures_savedir is not None:
        plt.savefig(
            os.path.join(figures_savedir, figure_name_time_series),
            format='pdf',
            bbox_inches='tight'
        )
        logging.info(f"Saved figure '{figure_name_time_series:s}' in '{figures_savedir:s}'.")
        plt.close()


def plot_method_tvfc_estimates(
    config_dict: dict,
    model_name: str,
    i_time_series: int,
    j_time_series: int,
    x_train_locations: np.array,
    y_train_locations: np.array,
    data_split: str,
    metric: str,
    subject_name: str = None,
    noise_type: str = None,
    i_trial: int = None,
    covs_type: str = None,
    pp_pipeline: str = None,
    label_name: str = None,
    plot_color: str = None,
    ax=None,
) -> None:
    """
    Plots a time series of TVFC estimates for a given estimation method.
    For the leave-every-other-out scheme, we only plot at the train locations.

    TODO: add compatability with HCP benchmarks
    TODO: merge with helpers.evaluation.get_tvfc_estimates

    Parameters
    ----------
    :param config_dict:
    :param model_name:
    :param x_train_locations:
        Array of shape (N, 1).
    :param y_train_locations:
        Array of shape (N, D).
    :param noise_type:
    :param data_split:
    :param i_trial:
    :param covs_type:
    :param pp_pipeline:
    :param label_name:
    :param metric:
        'correlation' or 'covariance'.
    :param subject_name:
    :param i_time_series:
        By default we expect the bivariate case.
    :param j_time_series:
        By default we expect the bivariate case.
    """
    n_time_series = y_train_locations.shape[1]
    data_set_name = config_dict['data-set-name']
    match data_set_name:
        case 'd2' | 'd3d' | 'd3s' | 'd4s' | 'd6s' | 'd9s' | 'd15s':
            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type, data_split,
                f'trial_{i_trial:03d}', model_name
            )
            wp_model_filename = f'{covs_type:s}.json'

            tvfc_estimates_savedir = os.path.join(
                config_dict['experiments-basedir'], noise_type,
                f'trial_{i_trial:03d}', 'TVFC_estimates',
                data_split, metric, model_name
            )
            tvfc_estimates_filepath = os.path.join(
                tvfc_estimates_savedir, f"{covs_type:s}.csv"
            )

            # Fix renaming issue.
            if not os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                logging.warning(f"Model file {os.path.join(wp_model_savedir, wp_model_filename):s} not found.")
                if covs_type == 'boxcar':
                    wp_model_filename = 'checkerboard.json'
            if not os.path.exists(tvfc_estimates_filepath):
                if covs_type == 'boxcar':
                    tvfc_estimates_filepath = os.path.join(
                        tvfc_estimates_savedir, 'checkerboard.csv'
                    )

            rescale_x_axis = None
        case 'HCP_PTN1200_recon2':
            raise NotImplementedError
        case 'rockland':
            wp_model_savedir = os.path.join(
                config_dict['experiments-basedir'], pp_pipeline, 'saved_models',
                data_split, model_name
            )
            wp_model_filename = f"{subject_name.removesuffix('.csv'):s}.json"
            tvfc_estimates_filepath = os.path.join(
                config_dict['experiments-basedir'], pp_pipeline, "TVFC_estimates",
                data_split, metric, model_name, subject_name
            )
            xx = convert_to_seconds(
                x_train_locations, repetition_time=config_dict['repetition-time'],
                data_length=x_train_locations.shape[0]
            )
            # edge_to_plot_indices = [i_time_series, j_time_series]
            # y_edge = y_train_locations[:, edge_to_plot_indices]  # (N, 2)
            # print(y_edge.shape)
        case _:
            logging.error(f"Dataset '{data_set_name:s}' not recognized.")
            return

    match model_name:
        case 'VWP' | 'VWP_joint':
            if os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                k = gpflow.kernels.Matern52()
                m = VariationalWishartProcess(
                    x_train_locations, y_train_locations,
                    nu=n_time_series,
                    kernel=k
                )
                m.load_from_params_dict(
                    savedir=wp_model_savedir,
                    model_name=wp_model_filename,
                )
                x_predict = np.linspace(0., 1., config_dict['wp-n-predict-samples']).reshape(-1, 1)
                plot_wishart_process_covariances_pairwise(
                    x_predict, m,
                    i=i_time_series,
                    j=j_time_series,
                    repetition_time=config_dict['repetition-time'] if data_set_name == 'rockland' else None,
                    data_length=y_train_locations.shape[0] if data_set_name == 'rockland' else None,
                    rescale_x_axis='seconds' if data_set_name == 'rockland' else None,
                    connectivity_metric=metric,
                    label=label_name if data_set_name == 'rockland' else 'WP',
                    ax=ax
                )
                del m
            else:
                logging.warning(f"VWP model in '{wp_model_savedir:s}' not found.")
        case 'SVWP' | 'SVWP_joint':
            if os.path.exists(os.path.join(wp_model_savedir, wp_model_filename)):
                k = gpflow.kernels.Matern52()
                m = SparseVariationalWishartProcess(
                    D=n_time_series,
                    Z=x_train_locations[:config_dict['n-inducing-points']],
                    nu=n_time_series,
                    kernel=k,
                    verbose=False
                )
                m.load_from_params_dict(
                    savedir=wp_model_savedir,
                    model_name=wp_model_filename,
                )
                x_predict = np.linspace(0., 1., config_dict['wp-n-predict-samples']).reshape(-1, 1)
                plot_wishart_process_covariances_pairwise(
                    x_predict, m,
                    i=i_time_series,
                    j=j_time_series,
                    rescale_x_axis='seconds' if data_set_name == 'rockland' else rescale_x_axis,
                    connectivity_metric=metric,
                    repetition_time=config_dict['repetition-time'] if data_set_name[0] != 'd' else None,
                    data_length=x_train_locations.shape[0],
                    label=label_name if data_set_name == 'rockland' else 'WP',  # use SVWP-J to refer to specific implementation
                    ax=ax,
                )
                del m
            else:
                logging.warning(f"SVWP model not found in '{wp_model_savedir:s}'.")
        case 'DCC' | 'DCC_joint' | 'DCC_bivariate_loop':
            if os.path.exists(tvfc_estimates_filepath):
                df = pd.read_csv(tvfc_estimates_filepath, index_col=0)  # (D*D, N)

                # Remove zero padding required to run these models (only for TR=1.4).
                if data_set_name == 'rockland' and config_dict['repetition-time'] == 1.4:
                    df = df.iloc[:, :-2]
                    print(df)
                    print(df.shape)

                cov_structure = to_3d_format(df.values)  # (N, D, D)
                plot_mgarch_estimated_covariance_structure(
                    estimated_tvfc_array=cov_structure,
                    xx=xx if data_set_name == 'rockland' else x_train_locations,
                    model_name=model_name,
                    i=i_time_series,
                    j=j_time_series,
                    connectivity_metric=metric,
                    markersize=3.6 if data_set_name not in ['d2'] else 0,
                    ax=ax
                )
            else:
                logging.warning(f"MGARCH model estimates '{tvfc_estimates_filepath:s}' not found.")
        case "SW_cross_validated":
            if os.path.exists(tvfc_estimates_filepath):
                covariance_structure_df = pd.read_csv(
                    tvfc_estimates_filepath,
                    index_col=0
                )  # (D*D, N)
                covariance_structure = to_3d_format(covariance_structure_df.values)  # (N, D, D)
                if data_set_name == 'rockland':
                    plot_estimated_covariance_structure_edge(
                        estimated_tvfc_array=covariance_structure,
                        xx=xx,
                        label_name=label_name,
                        i=i_time_series,
                        j=j_time_series,
                        connectivity_metric=metric
                    )
                else:
                    plot_cross_validated_sliding_windows_estimated_covariance_structure(
                        estimated_tvfc_array=covariance_structure,
                        xx=x_train_locations,
                        model_name=model_name,
                        i=i_time_series,
                        j=j_time_series,
                        connectivity_metric=metric,
                        markersize=3.6 if data_set_name not in ['d2'] else 0,
                        plot_color=plot_color,
                        ax=ax
                    )
            else:
                logging.warning(f"SW-CV TVFC estimates '{tvfc_estimates_filepath:s}' not found.")
        case 'SW_15' | 'SW_16' | 'SW_30' | 'SW_60' | 'SW_120':
            window_length = int(model_name.removeprefix('SW_'))
            window_length = int(np.floor(window_length / config_dict['repetition-time']))  # TODO: check this
            if data_split == 'LEOO':
                window_length = int(window_length / 2)
            plot_sliding_windows_estimated_covariance_structure(
                xx=xx if data_set_name == 'rockland' else x_train_locations,
                y=y_train_locations,
                window_length=window_length,
                repetition_time=config_dict['repetition-time'],
                i=i_time_series,
                j=j_time_series,
                label=model_name.replace('_', '-'),
                connectivity_metric=metric,
                markersize=3.6,
                ax=ax,
            )
        case 'SW':  # TODO: should we divide by 2 for leave-one-out?
            for window_length in config_dict['window-lengths']:
                plot_sliding_windows_estimated_covariance_structure(
                    xx=x_train_locations,
                    y=y_train_locations,
                    window_length=window_length,
                    i=i_time_series,
                    j=j_time_series,
                    label=f'SW-{window_length:d}',
                    connectivity_metric=metric,
                    ax=ax,
                )
        case "sFC":
            plot_static_estimated_covariance_structure(
                xx=xx if data_set_name == 'rockland' else x_train_locations,
                y=y_train_locations,
                i=i_time_series,
                j=j_time_series,
                connectivity_metric=metric,
                repetition_time=config_dict['repetition-time'],
                plot_color=plot_color,
                ax=ax,
            )
        case _:
            logging.error(f"Model name '{model_name:s}' not recognized.")
            exit()


def plot_wishart_process_covariances_pairwise(
    test_locations: np.array,
    m,
    n_mc_samples: int = 3000,
    i: int = 0,
    j: int = 1,
    ax=None,
    rescale_x_axis: str = None,
    repetition_time: float = None,
    data_length: int = None,
    connectivity_metric: str = 'correlation',
    linewidth: float = 1.5,
    label: str = 'WP',
) -> None:
    """
    We plot the mean of the predictive posterior as well as a 2 standard deviations (95%) confidence interval.
    We don't plot crosses at test locations, since these are arbitrary.
    """
    if connectivity_metric == 'correlation':
        all_covs_means, all_covs_stddevs = m.predict_corr(
            test_locations, n_mc_samples=n_mc_samples
        )  # (N, D, D), (N, D, D)
    else:
        all_covs_means, all_covs_stddevs = m.predict_cov(
            test_locations, n_mc_samples=n_mc_samples
        )  # (N, D, D), (N, D, D)

    # In the 2-dimensional case there is no distinction between joint or pairwise modeling.
    n_time_series = all_covs_means.shape[-1]
    if n_time_series == 2:
        label = label.removesuffix('-J').removesuffix('-BL')

    all_covs_means = [cov[i, j].numpy() for cov in all_covs_means]
    all_covs_means = np.array(all_covs_means)

    all_covs_stddevs = [cov[i, j].numpy() for cov in all_covs_stddevs]
    all_covs_stddevs = np.array(all_covs_stddevs)

    plot_wishart_process_covariances(
        xx=test_locations,
        all_covs_means_pair=all_covs_means,
        all_covs_stddevs_pair=all_covs_stddevs,
        ax=ax,
        rescale_x_axis=rescale_x_axis,
        repetition_time=repetition_time,
        data_length=data_length,
        linewidth=linewidth,
        label=label,
    )


def plot_wishart_process_covariances(
    xx: np.array,
    all_covs_means_pair: np.array,
    all_covs_stddevs_pair: np.array,
    ax=None,
    rescale_x_axis: str = None,
    repetition_time: float = None,
    data_length=None,
    linewidth: float = 1.5,
    alpha: float = 0.2,
    label: str = 'WP',
) -> None:
    """
    TODO: refactor this - change scale at the end of plotting
    """
    if rescale_x_axis == 'seconds':
        xx = convert_to_seconds(
            xx, repetition_time=repetition_time, data_length=data_length
        )
    if rescale_x_axis == 'minutes':
        xx = convert_to_minutes(
            xx, repetition_time=repetition_time, data_length=data_length
        )

    if ax is not None:
        (line, ) = ax.plot(
            xx, all_covs_means_pair,
            linewidth=linewidth,
            label=label,
            alpha=1.0,
        )
        col = line.get_color()
        ax.fill_between(
            xx[:, 0],
            (all_covs_means_pair - 2 * all_covs_stddevs_pair),
            (all_covs_means_pair + 2 * all_covs_stddevs_pair),
            color=col,
            alpha=alpha,
            linewidth=linewidth
        )
    else:
        (line,) = plt.plot(
            xx, all_covs_means_pair,
            linewidth=linewidth,
            label=label,
            alpha=1.0,
        )
        col = line.get_color()
        plt.fill_between(
            xx[:, 0],
            (all_covs_means_pair - 2 * all_covs_stddevs_pair),
            (all_covs_means_pair + 2 * all_covs_stddevs_pair),
            color=col,
            alpha=alpha,
            linewidth=linewidth
        )


def plot_wishart_process_variances() -> None:
    raise NotImplementedError


def plot_estimated_covariance_structure_edge(
    estimated_tvfc_array: np.array,
    xx: np.array,
    label_name: str,
    i: int,
    j: int,
    markersize: float = 3.6,
    linewidth: float = 2.5,
    connectivity_metric: str = 'correlation',
) -> None:
    """
    This only plots the estimated TVFC array.
    """
    if connectivity_metric == 'correlation':
        estimated_tvfc_array = to_correlation_structure(estimated_tvfc_array)  # (N, D, D)

    covariance_estimates = [step[i, j] for step in estimated_tvfc_array]
    covariance_estimates = np.array(covariance_estimates)  # (N, )

    plt.plot(
        xx, covariance_estimates, 'x-',
        linewidth=linewidth, markersize=markersize, label=label_name
    )


def plot_mgarch_estimated_covariance_structure(
    estimated_tvfc_array: np.array,
    xx: np.array,
    model_name: str,
    i: int,
    j: int,
    markersize: float = 3.6,
    linewidth: float = 1.5,
    connectivity_metric: str = 'correlation',
    ax=None,
) -> None:
    """
    We want the covariance plot to take up twice the space of the time series plots.
    """
    if connectivity_metric == 'correlation':
        estimated_tvfc_array = to_correlation_structure(estimated_tvfc_array)  # (N, D, D)

    # In the 2-dimensional case there is no distinction between joint or pairwise modeling.
    n_time_series = estimated_tvfc_array.shape[-1]
    if n_time_series == 2:
        model_name = model_name.removesuffix('_joint').removesuffix('_bivariate_loop')

    covariance_estimates = [step[i, j] for step in estimated_tvfc_array]
    covariance_estimates = np.array(covariance_estimates)  # (N, )

    match model_name:
        case 'DCC':
            model_name_str = 'DCC'
        case 'DCC_joint':
            model_name_str = 'DCC-J'
        case 'DCC_bivariate_loop':
            model_name_str = 'DCC-BL'
        case 'GO_joint':
            model_name_str = 'GO-J'
        case 'GO_bivariate_loop':
            model_name_str = 'GO-BL'
        case _:
            raise NotImplementedError(f"Model name '{model_name:s}' not recognized.")

    if ax is not None:
        ax.plot(
            xx, covariance_estimates, 'x-',
            linewidth=linewidth, 
            markersize=markersize, 
            label=model_name_str,
            alpha=0.7,
        )
    else:
        plt.plot(
            xx, covariance_estimates, 'x-',
            linewidth=linewidth,
            markersize=markersize,
            label=model_name_str,
            alpha=0.7,
        )


def plot_cross_validated_sliding_windows_estimated_covariance_structure(
    estimated_tvfc_array: np.array,
    xx: np.array,
    model_name: str,
    i: int,
    j: int,
    markersize: float = 3.6,
    linewidth: float = 1.5,
    connectivity_metric: str = 'correlation',
    plot_color: str = None,
    ax=None,
) -> None:
    """
    Simple function to plot cross validated sliding window estimates.
    """
    if connectivity_metric == 'correlation':
        estimated_tvfc_array = to_correlation_structure(estimated_tvfc_array)  # (N, D, D)

    covariance_estimates = [step[i, j] for step in estimated_tvfc_array]
    covariance_estimates = np.array(covariance_estimates)  # (N, )

    if model_name == 'SW_cross_validated':
        model_name_str = 'SW-CV'
    else:
        logging.error(f"Model name '{model_name:s}' not recognized.")
    if ax is not None:
        ax.plot(
            xx, covariance_estimates, 'x-',
            linewidth=linewidth,
            markersize=markersize,
            color=plot_color,
            label=model_name_str,
            alpha=0.7,
        )
    else:
        plt.plot(
            xx, covariance_estimates, 'x-',
            linewidth=linewidth,
            markersize=markersize,
            color=plot_color,
            label=model_name_str,
            alpha=0.7
        )


def plot_sliding_windows_estimated_covariance_structure(
    xx: np.array,
    y: np.array,
    window_length: int,
    i: int = 0,
    j: int = 1,
    repetition_time: float = None,
    markersize: float = 3.6,
    label: str = 'SW',
    linewidth: float = 1.5,
    connectivity_metric: str = 'correlation',
    plot_color: str = None,
    ax=None,
) -> None:
    """
    Plot covariance estimates from sliding window approach.
    """
    sw = SlidingWindows(
        x_train_locations=xx,
        y_train_locations=y,
        repetition_time=repetition_time,
    )
    estimated_tvfc_array = sw.overlapping_windowed_cov_estimation(
        window_length=window_length,
        repetition_time=repetition_time,
        connectivity_metric=connectivity_metric
    )  # (N, D, D)
    if ax is not None:
        ax.plot(
            xx, [cov_estimate[i, j] for cov_estimate in estimated_tvfc_array], 'x-',
            markersize=markersize,
            linewidth=linewidth,
            color=plot_color,
            label=label,
            alpha=0.7,
        )
    else:
        plt.plot(
            xx, [cov_estimate[i, j] for cov_estimate in estimated_tvfc_array], 'x-',
            markersize=markersize,
            linewidth=linewidth,
            color=plot_color,
            label=label,
            alpha=0.7,
        )


def plot_windowed_covariances(
    xx: np.array, 
    y: np.array,
    n_windows_list: list = [1, 4, 8, 12],
    repetition_time: float = None,
    i: int = 0, 
    j: int = 1,
    connectivity_metric: str = 'correlation', 
    ax=None,
) -> None:
    """
    Plot covariance estimates from windowed approach, which divides the data into non-overlapping regions..

    :param xx: only used for plotting.
    :param y: array of shape (N, D).
    :param n_windows_list:
    :param i: row index
    :param j: col index
    :param connectivity_metric: whether we want to plot correlations instead of covariances
    :param ax:
    """
    for n_windows in n_windows_list:
        sw = SlidingWindows(
            x_train_locations=xx,
            y_train_locations=y,
            repetition_time=repetition_time
        )
        estimated_tvfc_array = sw.windowed_cov_estimation(n_windows=n_windows)  # (N, D, D)

        if connectivity_metric == 'correlation':
            estimated_tvfc_array = to_correlation_structure(estimated_tvfc_array)  # (N, D, D)

        # Select a single covariance term.
        cov_array = [cov_estimate[i, j] for cov_estimate in estimated_tvfc_array]  # (N, )
        if n_windows == 1:
            label = 'sFC'
        else:
            label = f'W{n_windows:d}'

        if ax is not None:
            ax.plot(
                xx, cov_array, label=label
            )
        else:
            plt.plot(
                xx, cov_array, label=label
            )


def plot_static_estimated_covariance_structure(
    xx: np.array,
    y: np.array,
    i: int = 0,
    j: int = 1,
    connectivity_metric: str = 'correlation',
    repetition_time: float = None,
    plot_color: str = None,
    linewidth: float = 1.5,
    label: str = 'sFC',
    ax=None,
) -> None:
    """
    Plot static covariance estimates.

    :param xx: only used for plotting.
    :param y: array of shape (N, D).
    :param i: row index
    :param j: col index
    :param connectivity_metric: whether we want to plot correlations instead of covariances
    :param repetition_time:
    :param label:
    :param ax:
    """
    sw = SlidingWindows(
        x_train_locations=xx,
        y_train_locations=y,
        repetition_time=repetition_time,
    )
    estimated_tvfc_array = sw.estimate_static_functional_connectivity(
        connectivity_metric=connectivity_metric
    )  # (N, D, D)

    # Select a single covariance term.
    estimated_cov_array = estimated_tvfc_array[:, i, j]  # (N, )

    if ax is not None:
        ax.plot(
            xx, estimated_cov_array,
            color=plot_color,
            linewidth=linewidth,
            label=label,
            alpha=0.7,
        )
    else:
        plt.plot(
            xx, estimated_cov_array,
            color=plot_color,
            linewidth=linewidth,
            label=label,
            alpha=0.7,
        )
