import logging

import numpy as np


def get_ground_truth_covariance_structure(
    covs_type: str,
    num_samples: int,
    signal_to_noise_ratio,
    data_set_name: str,
) -> np.array:
    """
    Return full covariance structure of shape (N, D, D).

    These are the 'synthetic', 'simulated', or 'oracle' covariance structures.

    TODO: make this more flexible

    Parameters
    ----------
    covs_type: str
    num_samples: int
    signal_to_noise_ratio: float
    data_set_name: str
    return: np.array
    """
    covariance_structure = get_covariance_time_series(
        covs_type,
        num_samples=num_samples,
        signal_to_noise_ratio=signal_to_noise_ratio
    )  # (N, )
    match [data_set_name[0], data_set_name[1:-1], data_set_name[-1]]:
        case ['d', '2', 's' | 'd'] | ['d', '', '2']:
            covariance_structure = get_d2_covariance_structure(covariance_structure)  # (N, D, D)
        case ['d', '3', 'd']:
            covariance_structure = get_d3d_covariance_structure(covariance_structure)  # (N, D, D)
        case ['d', num_time_series, 's']:
            covariance_structure = get_sparse_covariance_structure(
                covariance_structure,
                num_time_series=int(num_time_series)
            )  # (N, D, D)
        case _:
            logging.warning(f"Data set name '{data_set_name:s}' not recognized.")
            exit()
    return covariance_structure


def get_covariance_time_series(
    cov_type: str,
    num_samples: int,
    signal_to_noise_ratio=None,
) -> np.array:
    """
    This function is only called from the plotting and quantitative results scripts.

    :param cov_type:
    :param num_samples:
    :param signal_to_noise_ratio:
    :return:
    """
    covariance_time_series = get_covariance_structure(cov_type, num_samples)
    if signal_to_noise_ratio is not None:
        covariance_time_series = _adjust_with_signal_to_noise_ratio(
            covariance_time_series,
            signal_to_noise_ratio=signal_to_noise_ratio
        )
    return covariance_time_series


def get_covariance_structure(cov_type: str, num_samples: int) -> np.array:
    """
    Selects the correct covariance structure of interest.

    Parameters
    ----------
    :param cov_type:
    :param num_samples:
    :return:
        Array of shape (N, ).
    """
    match cov_type:
        case 'null':
            covariance_time_series = get_constant_covariances(num_samples, covariance=0)
        case 'change_point':
            covariance_time_series = get_change_point_covariances(num_samples)
        case 'constant':
            covariance_time_series = get_constant_covariances(num_samples, covariance=0.8)
        case 'stepwise':
            covariance_time_series = get_stepwise_covariances(num_samples)
        case 'periodic_1':
            covariance_time_series = get_periodic_covariances(num_samples, num_periods=1)
        case 'periodic_2':
            covariance_time_series = get_periodic_covariances(num_samples, num_periods=2)
        case 'periodic_3':
            covariance_time_series = get_periodic_covariances(num_samples, num_periods=3)
        case 'periodic_4':
            covariance_time_series = get_periodic_covariances(num_samples, num_periods=4)
        case 'periodic_5':
            covariance_time_series = get_periodic_covariances(num_samples, num_periods=5)
        case 'state_transition':
            covariance_time_series = get_state_transition_covariances(num_samples)
        case 'boxcar' | 'checkerboard':
            covariance_time_series = get_boxcar_covariances(num_samples)
        case _:
            logging.error(f"cov_type '{cov_type:s}' not recognized.")
            covariance_time_series = None
    return covariance_time_series


def _adjust_with_signal_to_noise_ratio(covariances_ts: np.array, signal_to_noise_ratio) -> np.array:
    """
    When we add noise to our original time series, we need to adjust the ground truth covariance structure.

    :param covariances_ts:
    :param signal_to_noise_ratio:
    :return:
    """
    mixing_coefficient = signal_to_noise_ratio / (1 + signal_to_noise_ratio)  # \alpha
    covariance_normalization = mixing_coefficient**2 / (mixing_coefficient**2 + (1 - mixing_coefficient)**2)
    covariances_ts *= covariance_normalization
    return covariances_ts


def get_constant_covariances(num_samples: int, covariance: float) -> np.array:
    """
    Constant, time-invariant (static) covariance.

    :param num_samples: N, number of data points over time.
    :param covariance: if set to 0 this is the null model (uncorrelated time series).
    :return:
        array of shape (N, ).
    """
    return covariance * np.ones(num_samples)


def get_periodic_covariances(
    num_samples: int,
    num_periods: int,
) -> np.array:
    """
    Simple dynamic periodic covariance.

    :return:
        array of shape (N, ).
    """
    return np.sin(num_periods * 2 * np.pi / num_samples * np.arange(num_samples))


def get_stepwise_covariances(num_samples: int) -> np.array:
    """
    Simple dynamic covariance.
    Perfect correlation up and till a certain point, then perfectly uncorrelated, then perfectly anti-correlated.
    To avoid floating point issues, we make sure the values are never exactly 1 or -1.

    :return:
        array of shape (N, ).
    """
    num_steps = 3
    covs = np.zeros(num_samples)
    covs[:int(num_samples/num_steps)] = 1 - 1e-3
    covs[-int(num_samples/num_steps):] = -1 + 1e-3
    return covs


def get_state_transition_covariances(num_samples: int) -> np.array:
    """
    This mimics when two regions are jointly active in a brain state or not.
    TODO: this does not work for all N, for example not for N=100 (it does for 80 and 120)

    :return:
        array of shape (N, ).
    """
    off_covariance = 0.2
    on_covariance = 0.6

    durations = [5, 15, 10, 25, 20, 5, 15, 30, 10, 25, 20, 20]
    durations = [int(d * num_samples / 200) for d in durations]  # scale to make it fit
    assert sum(durations) == num_samples

    covs = np.array([])
    state = 0
    for state_duration in durations:
        if state == 0:
            covs = np.concatenate((covs, off_covariance * np.ones(state_duration)))
        if state == 1:
            covs = np.concatenate((covs, on_covariance * np.ones(state_duration)))
        state = 1 - state
    return covs


def get_boxcar_covariances(num_samples: int) -> np.array:
    """
    Synthetic covariance structure inspired by Rockland experiments.
    That is, the external visual stimuli is a simple off-on-off-on-off-on-off boxcar function (7 segments).
    The final segment can be of variable length to make sure the total number of samples is correct.
    The signal is also convolved with a hemodynamic response function.

    Parameters
    ----------
    num_samples: int
    :return:
        array of shape (N, ).
    """
    off_covariance = 0.0
    on_covariance = 0.8
    num_segments = 7
    segment_duration = int(num_samples / num_segments)
    covs = np.array([])
    visual_stimulus = 0
    for _ in range(num_segments - 1):
        if visual_stimulus == 0:
            covs = np.concatenate((covs, off_covariance * np.ones(segment_duration)))
        if visual_stimulus == 1:
            covs = np.concatenate((covs, on_covariance * np.ones(segment_duration)))
        visual_stimulus = 1 - visual_stimulus
    final_segment_duration = num_samples - len(covs)
    covs = np.concatenate((covs, off_covariance * np.ones(final_segment_duration)))
    assert len(covs) == num_samples

    # Convolve with hemodynamic response function.
    hrf_signal = hrf(np.arange(num_samples))
    convolved_covs = np.convolve(covs, hrf_signal)[:num_samples]
    convolved_covs = (convolved_covs / np.max(convolved_covs)) * (on_covariance - off_covariance) + off_covariance

    return convolved_covs


def get_change_point_covariances(num_samples: int) -> np.array:
    """
    Covariance structure to study change points with.

    :param num_samples: N, number of data points over time.
    :return:
        array of shape (N, ).
    """
    covariance_array = 0.8 * np.ones(num_samples)
    covariance_array[:int(num_samples/2)] = 0.2
    return covariance_array


def hrf(t):
    """
    A hemodynamic (impulse) response function.
    """
    return t ** 8.6 * np.exp(-t / 0.547)


def get_d2_covariance_structure(
    covariance_time_series: np.array,
    num_time_series: int = 2,
) -> np.array:
    """
    Returns full covariance structure.
    The diagonal (time series variances) are set to 1.

    :param covariance_time_series: array of shape (N, ).
    :return:
        array of shape (N, D, D), i.e. (N, 2, 2) in this case.
    """
    return get_sparse_covariance_structure(covariance_time_series, num_time_series=2)


def get_d3d_covariance_structure(covariance_time_series: np.array, num_time_series: int = 3) -> np.array:
    """
    Returns the full covariance structure (a series of covariance matrices).
    All off-diagonals are identical here.
    The allowed range for p(t) is -0.5 to 1.

    :param covariance_time_series: time series of varying covariance term of shape (N, ).
    :param num_time_series:
    :return:
        array of shape (N, D, D), i.e. (N, 3, 3) in this case.
    """
    num_time_steps = len(covariance_time_series)

    # Make sure matrix is positive semi-definite.
    covariance_time_series[covariance_time_series == 1] -= 1e-3
    covariance_time_series[covariance_time_series == -1] += 1e-3
    if covariance_time_series.max() > 1:
        logging.warning('Invalid covariance structure.')
    if covariance_time_series.min() < -0.5:
        logging.warning('Invalid covariance structure.')
        covariance_time_series = covariance_time_series * 3 / 4 + 1 / 4  # re-scale to allowed range

    covariance_structure = []
    for sample_step in range(num_time_steps):
        time_step_covariance_matrix = covariance_time_series[sample_step] * np.ones((num_time_series, num_time_series))
        np.fill_diagonal(time_step_covariance_matrix, 1)
        covariance_structure.append(time_step_covariance_matrix)
    return np.array(covariance_structure)


def get_sparse_covariance_structure(
    covariance_time_series: np.array,
    num_time_series: int,
) -> np.array:
    """
    Returns the full covariance structure (a series of covariance matrices).
    The first two time series are correlated, but the others are uncorrelated with both of these.

    Parameters
    ----------
    :param covariance_time_series:
        Time series of varying covariance term of shape (N, ).
    :param num_time_series:
    :return:
        covariance structure array of shape (N, D, D).
    """
    num_time_steps = len(covariance_time_series)  # N

    # Make sure matrix is positive semi-definite.
    covariance_time_series[covariance_time_series == 1] -= 1e-3
    covariance_time_series[covariance_time_series == -1] += 1e-3

    covariance_structure = []
    for sample_step in range(num_time_steps):
        time_step_covariance_matrix = np.eye(num_time_series)
        time_step_covariance_matrix[0, 1] = time_step_covariance_matrix[1, 0] = covariance_time_series[sample_step]
        covariance_structure.append(time_step_covariance_matrix)
    return np.array(covariance_structure)


def get_ylim(covs_type: str) -> list[float]:
    match covs_type:
        case 'constant':
            return [0.0, 1.0]
        case 'null':
            return [-0.35, 0.35]
        case 'state_transition':
            return [0.0, 1.0]
        case _:
            return [-1.0, 1.0]


def to_human_readable(covs_type: str) -> str:
    """
    Convert covs_type to human readable format used in figures.
    """
    match covs_type:
        case 'periodic_1':
            return 'periodic (slow)'
        case 'periodic_3':
            return 'periodic (fast)'
        case 'checkerboard':
            return 'boxcar'
        case _:
            return covs_type.replace('_', ' ')
