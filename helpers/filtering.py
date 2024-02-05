from fcest.helpers.filtering import _compute_lower_frequency_cutoff
import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def bandpass_filter_data(
        y_observed: np.array, window_length: int, repetition_time: float,
        freq_high: float = 0.08
):
    """
    We want to remove frequencies between 1 / window length (in seconds) and 0.08 Hz.

    :param y_observed:
    :param window_length: in TRs.
    :param repetition_time: TR in seconds.
    :param freq_high: typically 0.08 Hz or 0.1 Hz for resting-state data.
    :return:
    """
    # logging.info('Band-pass filtering data...')
    sampling_rate = 1.0 / repetition_time
    # logging.info('Sampling rate of %.3f Hz.', sampling_rate)
    nyquist_frequency = sampling_rate / 2
    # logging.info('Nyquist frequency is %.3f Hz.', nyquist_frequency)

    y_filtered = np.zeros_like(y_observed)
    for i_time_series, single_time_series in enumerate(y_observed.T):
        y_filtered[:, i_time_series] = butter_bandpass_filter(
            single_time_series,
            cutoff_low=_compute_lower_frequency_cutoff(window_length, repetition_time),
            cutoff_high=freq_high,
            nyquist_freq=nyquist_frequency
        )
    return y_filtered


def butter_bandpass_filter(data, cutoff_low, cutoff_high, nyquist_freq):
    b, a = _butter_bandpass(cutoff_low, cutoff_high, nyquist_freq=nyquist_freq)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def _butter_bandpass(cutoff_low, cutoff_high, nyquist_freq, order=5):
    normal_cutoff_low = cutoff_low / nyquist_freq
    normal_cutoff_high = cutoff_high / nyquist_freq
    b, a = butter(order, [normal_cutoff_low, normal_cutoff_high], btype='band')
    return b, a
