from collections import Counter
import logging

import numpy as np

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


def compute_i2c2(
        y: np.array, n_subjects: int, n_scans: int,
        demean: bool = True, twoway: bool = True
) -> float:
    """
    Computes image intra-class correlation coefficient (I2C2).

    This is an adaptation of code in R, with several assumptions made to simplify the code.

    Sources:
        https://github.com/muschellij2/I2C2/blob/3cb224d6c20d3dce240552f0f857623351d7c042/R/I2C2_orig.R
        https://www.smart-stats.org/wiki/image-intra-class-correlation-coefficient-i2c2

    :param y: A data matrix of shape (n, p) containing n vectorized image data with p voxels.
        Each row contains one observed image data at a particular visit for one subject.
        Each column contains image values for all subjects and visits at a particular voxel.
        The rows are organized by subjects and then visits, EX)
        (Y11, Y12, Y21, Y22, ... , YI1, YI2)
        Here it is assumed that each subject does each repetition.
        That is, y is an array of shape (n_scans*n_subjects, D*(D-1)/2)
    :param n_subjects: number of subjects. Represented as I in Shou et al. (2013).
    :param n_scans: number of repetitions. Represented as J in Shou et al. (2013).
    :param demean: If demean == TRUE, we calculate the overall mean function and subtract the mean function from the data.
    :param twoway: If twoway mean subtraction is needed ("twoway==TRUE"), the visit specific mean function and the deviation from
        the overall mean to visit specific mean functions are also computed.
    :return:
        estimated I2C2 score
    """
    n, p = y.shape
    print('n =', n, 'p =', p)

    scan_ids = np.repeat(
        np.arange(n_subjects), n_scans
    )  # e.g. (1, 1, 2, 2, 3, 3, 4, 4, ... , I, I) for J=2 repetitions/visits/scans
    visit = np.tile(
        np.arange(n_scans), n_subjects
    )  # e.g. (1, 2, 1, 2, 1, 2, ... , 1, 2) for J=2 repetitions/visits/scans

    unique_visits = np.unique(visit)  # array of shape (n_visits, ) e.g. [0, 1, 2, 3]
    assert len(unique_visits) == n_scans

    if not demean:
        W = y
    else:
        mu = np.mean(y, axis=0)  # (n_voxels, )
        resd = np.zeros((n_subjects*n_scans, p))
        resd = y - mu  # (n_scans*n_subjects, D*(D-1)/2)

        if twoway:
            T = visit
            eta = np.zeros((n_scans, p))
            for j in unique_visits:
                visit_indices = np.where(T == j)[0]
                eta[j, :] = np.mean(y[visit_indices, :], axis=0) - mu
                # eta[which(unique(T) == j),] <- apply(y[T == j,], 2, mean) - mu

            # Calculate residuals by subtracting visit-specific mean from original functions for
            # 'twoway == TRUE', or subtracting overall mean function for 'twoway == FALSE'.
            for j in unique_visits:
                visit_indices = np.where(T == j)[0]  # (n_subjects, )
                resd[visit_indices, :] = y[visit_indices, :] - mu + eta[j, :]
        W = resd

    # Reset the id number to be arithmetic sequence starting from 1.
    # unique_ids = np.unique(ids)
    # assert unique_id == np.arange(I)
    # ids = np.array(match(ids, unique_ids))  # e.g. (1, 1, 2, 2, 3, 3, 4, 4, ... , I, I) for J=2 repetitions

    # n_I0 = np.array(table(ids))  # visit number for each id cluster
    n_I0 = Counter(scan_ids)
    # k2 = sum(n_I0 ^ 2)

    Wdd = np.mean(W, axis=0)  # (n_voxels, ) population average for the demeaned dataset W
    assert Wdd.shape == (p, )

    # TODO: check if this is really the equivalent of the statement in R
    print('W', W.shape)
    Si = W.reshape(n_subjects, n_scans, -1).sum(1)  # (n_subjects, n_voxels)
    print('Si', Si.shape)
    # Si = rowsum(W, ids)  # subject-specific sum for the demeaned dataset W

    # Use the method of moments estimator formula from the manuscript.
    # Wi = Si / n_I0  # (n_subjects, n_voxels)
    Wi = Si / n_scans  # (n_subjects, n_voxels) this is equivalent if each subject has the same number of visits

    # Compute traces: we expect (approximately) that trKw = trKx + trKu.
    print('W', W.shape, 'Wi', Wi.shape)

    trace_Ku = np.sum((W - Wi[scan_ids, :])**2) / (n - n_subjects)
    print('trKu', trace_Ku)

    trace_Kw = np.sum((W - Wdd)**2) / (n - 1)
    print('trKw', trace_Kw)

    trace_Kx = (trace_Kw - trace_Ku)  # / (1 + (1 - k2 / n) / (n - 1))  # remove the constant in the denominator
    print('trKx', trace_Kx)

    i2c2_score = trace_Kx / (trace_Kx + trace_Ku)

    return i2c2_score


def to_i2c2_format(original_format_array: np.array) -> np.array:
    """
    Prepares an array for I2C2 computation.
    TODO: do we run I2C2 on the full correlation matrix or just the lower triangular values?
        It should not really make a difference actually.

    :param original_format_array: shape of (n_subjects, n_scans, D, D)    
    :return:
    """
    print('\n\n')
    print(original_format_array.shape)

    # Full matrix flatten.
    # i2c2_format_array = original_format_array.reshape(
    #     original_format_array.shape[0], original_format_array.shape[1],
    #     original_format_array.shape[2] * original_format_array.shape[3]
    # )  # (n_subjects, n_scans, D*D)
    # Interactions only flatten.
    i2c2_format_array = _extract_interactions(original_format_array)  # (n_subjects, n_scans, D*(D-1)/2)
    print(i2c2_format_array.shape)

    # Order like [subject_1_scan_1, subject_1_scan_2, .., .., subject_2_scan_1, ..., subject_last_scan_last].
    i2c2_format_array = i2c2_format_array.reshape(
        i2c2_format_array.shape[0] * i2c2_format_array.shape[1],
        i2c2_format_array.shape[2]
    )
    print(i2c2_format_array)
    print('\n\n')
    return i2c2_format_array


def _extract_interactions(original_array: np.array):
    n_interactions = int(original_array.shape[2] * (original_array.shape[2] - 1) / 2)
    interactions_array = np.zeros((original_array.shape[0], original_array.shape[1], n_interactions))
    for i_subject in range(original_array.shape[0]):
        for i_scan in range(original_array.shape[1]):
            subject_scan_matrix = original_array[i_subject, i_scan, :, :]
            interactions_array[i_subject, i_scan, :] = subject_scan_matrix[np.tril_indices(subject_scan_matrix.shape[0], k=-1)]
    return interactions_array
