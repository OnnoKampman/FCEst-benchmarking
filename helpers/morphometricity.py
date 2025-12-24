import logging

import numpy as np
import pandas as pd
import scipy.linalg as la

from helpers.data import assert_normalized


def get_phenotype_array(
    phenotype_df: pd.DataFrame,
    subjects_subset_list: list,
    morphometricity_subject_measure: str,
) -> np.array:
    """
    Returns a normalized array with values for a single subject measure.

    Sources:
        https://github.com/ThomasYeoLab/CBIG/blob/703454dba7b6e0b1cfb97beb449e1593c5170547/stable_projects/preprocessing/Li2019_GSR/replication/scripts/HCP_lists/58behaviors_age_sex.txt
        https://github.com/ThomasYeoLab/CBIG/blob/703454dba7b6e0b1cfb97beb449e1593c5170547/stable_projects/preprocessing/Li2019_GSR/VarianceComponentModel/README.md
    
    Parameters
    ----------
    :param phenotype_df:
    :param subjects_subset_list:
    :param morphometricity_subject_measure:
    :return:
        denoted as `y`, the phenotype array of shape (n_subjects, 1)
    """
    phenotype_array = phenotype_df.loc[subjects_subset_list, morphometricity_subject_measure]  # floats array

    # Impute NaN entries as the mean of all subjects.
    n_nans = phenotype_array.isna().sum()
    print(f'{morphometricity_subject_measure:s}: found {n_nans:d} NaNs.')
    if n_nans > 0:
        phenotype_array[phenotype_array.isna()] = phenotype_array.mean()

    # Normalize array.
    # The variance component model assumes behavioral scores are Gaussian distributed.
    # TODO: should we do quantile normalization here?
    phenotype_array -= np.mean(phenotype_array)
    phenotype_array /= np.std(phenotype_array)
    assert_normalized(phenotype_array)

    phenotype_array = phenotype_array.values.reshape(-1, 1)

    return phenotype_array


def get_covariates_array(
    phenotype_df: pd.DataFrame,
    subjects_subset_list: list,
    nuisance_variables: list,
    morphometricity_subject_measure: str,
) -> np.array:
    """
    These covariates or nuisance variables will be regressed out.
    TODO: add motion (FD and DVARS)

    Parameters
    ----------
    :param phenotype_df:
    :param subjects_subset_list:
    :param nuisance_variables: list of strings.
    :param morphometricity_subject_measure:
    :return:
    """
    # Remove subject measure from nuisance variables (if it occurs).
    print('nuisance variables:')
    print(nuisance_variables)
    try:
        nuisance_variables.remove(morphometricity_subject_measure)
        print(f"Nuisance variables after removing subject measure '{morphometricity_subject_measure:s}':")
    except ValueError as ve:
        print(ve)
        print(f"Subject measure '{morphometricity_subject_measure:s}' is not one of the nuisance variables:")
    print(nuisance_variables)

    # Select nuisance variables from subject measures DataFrame.
    covariates_array = phenotype_df.loc[subjects_subset_list, nuisance_variables]  # floats array
    covariates_array = np.atleast_2d(np.float64(covariates_array.values))  # (num_subjects, num_covariates)

    # Normalize covariates array.
    covariates_array -= np.mean(covariates_array, axis=0)
    covariates_array /= np.std(covariates_array, axis=0)
    for i_covariate_array in range(covariates_array.shape[1]):
        assert_normalized(covariates_array[:, i_covariate_array])

    # If there are no nuisance variables, we feed an array of zeros.
    if len(nuisance_variables) == 0:
        covariates_array = np.zeros(shape=(len(subjects_subset_list), 1))
    return covariates_array


def variance_component_model(
    phenotype_array: np.array,
    X: np.array,
    K: np.array,
    tol: float = 1e-4,
    max_num_iterations: int = 100,
) -> dict:
    """
    Compute (using variance components model) the variation of y explained by the covariance matrix K, after regressing
    the effects of confounds in X.

    Implementing the codes from Sabuncu et al.
    Original source code is available in Matlab
    http://people.csail.mit.edu/msabuncu/morphometricity/Morphometricity.m

    Source:
        https://github.com/sina-mansour/neural-identity/blob/bf8855b3443737f5c660ea242c43282bf9d2071f/codes/vcm.py

    TODO: add a jackknife procedure to get estimate of uncertainty in morphometricity scores

    Parameters
    ----------
    :param phenotype_array: (y): 
        (num_subjects, 1) vector array of phenotypes (trait values)
    :param X: (num_subjects, num_covariates)
        The (design) matrix of confounding variables (sometimes called covariates or nuisance variables) such as age and sex.
    :param K: (n_subjects, n_subjects)
        Array of anatomical similarity matrix (ASM)
        K has to be a symmetric, positive semi-definite matrix with its diagonal elements averaging to 1.
        If K is not non-negative definite, its negative eigenvalues will be set to zero, and a warning will be printed.
    :param tol:
        The tolerance for the convergence of the ReML algorithm
    :param max_num_iterations:
        The maximum number of iterations for the ReML algorithm
    :return:
        dict: {
            'flag': flag indicates the convergence of the ReML algorithm (1 if converged, and 0 if not),
            'm2': the morphometricity estimate (variance explained value),
            'SE': the standard error of the morphometricity estimate,
            'Va': the total anatomical/morphological/TVFC variance,
            'Ve': the residual variance,
            'Lnew': the ReML likelihood when the algorithm has converged,
        }
    """
    # Check whether the GRM is non-negative definite.
    # NOTE: in Python, D is a vector but in Matlab it is a diagonal matrix
    # [U, D] = eig(K);
    D, U = la.eigh(K)
    if np.min(D) < 0:
        logging.warning('The GRM is not non-negative definite! Setting negative eigenvalues to zero...')
        D[D < 0] = 0   # set negative eigenvalues to zero
        K = _mrdivide(np.dot(U, np.diag(D)), U)   # reconstruct the GRM (K = U * D / U)

    n_subjects = len(phenotype_array)  # the total number of subjects

    print('\nInitializing values...\n')

    # Calculate the phenotypic variance (Vp) and initialize the anatomical variance (Va) and residual variance (Ve).
    variance_phenotype = np.var(phenotype_array)
    variance_anatomical = variance_phenotype / 2  # Va, anatomical variance \sigma_a^2, total variance captured by the ASM
    variance_residual = variance_phenotype / 2  # Ve, residual variance \sigma_e^2 of noise vector
    print(f'Variance (phenotypic):           {variance_phenotype:.3f}')

    # Initialize the covariance matrix.
    V = variance_anatomical * K + variance_residual * np.eye(n_subjects)  # (n_subjects, n_subjects)

    projection_matrix = _compute_projection_matrix(n_subjects=n_subjects, V=V, X=X)  # (n_subjects, n_subjects)

    # Use the expectation maximization (EM) algorithm as an initial update.

    # Update the initial anatomical variance (Va).
    # Va = ( Va^2*y'*P*K*P*y + trace(Va*eye(Nsubj) - Va^2*P*K) ) / Nsubj
    variance_anatomical = np.dot(np.dot(np.dot(np.dot(((variance_anatomical**2) * phenotype_array.T), projection_matrix), K), projection_matrix), phenotype_array) + np.trace(variance_anatomical * np.eye(n_subjects) - np.dot(((variance_anatomical**2) * projection_matrix), K))
    variance_anatomical = variance_anatomical[0][0] / n_subjects

    # Update the initial residual variance (Ve).
    # Ve = ( Ve^2*y'*P*P*y + trace(Ve*eye(Nsubj) - Ve^2*P) ) / Nsubj
    variance_residual = np.dot(np.dot(np.dot(((variance_residual**2) * phenotype_array.T), projection_matrix), projection_matrix), phenotype_array) + np.trace(np.dot(variance_residual, np.eye(n_subjects)) - ((variance_residual**2) * projection_matrix))
    variance_residual = variance_residual[0][0] / n_subjects

    variance_anatomical = _ensure_positive(
        value=variance_anatomical,
        var_phenotype=variance_phenotype
    )
    variance_residual = _ensure_positive(
        value=variance_residual,
        var_phenotype=variance_phenotype
    )
    print(f'Variance (anatomical) (Va) init: {variance_anatomical:.3f}')
    print(f'Variance (residual) (Ve) init:   {variance_residual:.3f}')

    # Update the covariance matrix.
    V = variance_anatomical * K + variance_residual * np.eye(n_subjects)  # (n_subjects, n_subjects)

    projection_matrix = _compute_projection_matrix(n_subjects=n_subjects, V=V, X=X)  # (n_subjects, n_subjects)

    # Calculate the log determinant of the covariance matrix.
    sign, logdet = np.linalg.slogdet(V)
    logdetV = sign * logdet  # np.float64

    # Initialize the ReML likelihood.
    Lold = np.inf  # float
    Lnew = _compute_likelihood(logdetV, X, V, phenotype_array, projection_matrix)

    iteration = 0
    while abs(Lnew - Lold) >= tol and iteration < max_num_iterations:  # criteria of termination
        iteration += 1
        Lold = Lnew
        logging.info(f"ReML Iteration {iteration:02d}")

        # update the first-order derivative of the ReML likelihood

        score_vector = _construct_score_vector(
            projection_matrix=projection_matrix,
            K=K,
            phenotype_array=phenotype_array
        )  # (2, 1)
        I = _compute_information_matrix(phenotype_array, projection_matrix, K)  # (2, 2)

        # update the variance component parameters
        T = np.array([[variance_anatomical], [variance_residual]]) + la.solve(I, score_vector)  # (2, 1)
        variance_anatomical = T[0, 0]
        variance_residual = T[1, 0]

        # Set negative estimates of the variance component parameters to Vp * 1e-6.
        variance_anatomical = _ensure_positive(
            value=variance_anatomical,
            var_phenotype=variance_phenotype
        )
        variance_residual = _ensure_positive(
            value=variance_residual,
            var_phenotype=variance_phenotype
        )
        print(f'Variance: (anatomical) (Va): {variance_anatomical:.3f} (residual) (Ve): {variance_residual:.3f}')

        # Update the covariance matrix.
        V = variance_anatomical * K + variance_residual * np.eye(n_subjects)  # (n_subjects, n_subjects)

        projection_matrix = _compute_projection_matrix(n_subjects=n_subjects, V=V, X=X)

        if np.isnan(V).any():
            # if (np.isnan(V).any() or np.isinf(Lnew)):
            flag = 0
            m2 = np.nan
            SE = np.nan
            variance_anatomical = np.nan
            variance_residual = np.nan
            Lnew = np.nan
            return {
                'flag': flag,
                'm2': m2,
                'SE': SE,
                'Va': variance_anatomical,
                'Ve': variance_residual,
                'Lnew': Lnew,
                'dist': np.nan
            }

        # Calculate the log determinant of the covariance matrix.
        sign, logdet = np.linalg.slogdet(V)
        logdetV = sign * logdet

        # Update the ReML likelihood.
        Lnew = _compute_likelihood(logdetV, X, V, phenotype_array, projection_matrix)

    m2 = _estimate_morphometricity(var_anatomical=variance_anatomical, var_residual=variance_residual)
    I = _compute_information_matrix(phenotype_array, projection_matrix, K)
    standard_error = _estimate_standard_error(
        m2=m2,
        var_anatomical=variance_anatomical,
        information_matrix=I
    )

    # Diagnose the convergence.
    if (iteration == max_num_iterations) and (abs(Lnew - Lold) >= tol):
        flag = 0
    else:
        flag = 1

    # dist = abs(Lnew - Lold)  # difference of the last two steps in ReML convergence

    return {
        'flag': flag,
        'm2': m2, 'SE': standard_error,
        'Va': variance_anatomical, 'Ve': variance_residual,
        'Lnew': Lnew
    }


def _compute_projection_matrix(n_subjects: int, V: np.array, X: np.array) -> np.array:
    """
    Compute the projection matrix.
    P = (eye(n_subjects) - (V\X) / (X'/V*X)*X') / V
    V\X is the Matlab translation of la.solve(V, X).
    :param n_subjects:
    :param V: array of shape (n_subjects, n_subjects)
    :param X: nuisance variables array of shape (num_subjects, num_covariates)
    :return:
    """
    return _mrdivide(
        (
            np.eye(n_subjects) -
            np.dot(_mrdivide(la.solve(V, X), np.dot(_mrdivide(X.T, V), X)), X.T)
        ),
        V
    )


def _compute_likelihood(
    logdetV: float,
    X: np.array,
    V: np.array,
    y: np.array,
    projection_matrix: np.array,
) -> float:
    """
    Lnew = -1/2 * logdetV - 1/2*log(det(X'/V*X)) - 1/2*y'*P*y
    """
    term_1 = -0.5 * logdetV
    term_2 = 0.5 * np.log(la.det(np.dot(_mrdivide(X.T, V), X)))
    term_3 = 0.5 * np.dot(np.dot(y.T, projection_matrix), y)
    likel = term_1 - term_2 - term_3
    return likel[0][0]


def _construct_score_vector(
    projection_matrix: np.array,
    K: np.array,
    phenotype_array: np.array,
) -> np.array:
    """
    Compute score vector (S).

    Parameters
    ----------
    :param projection_matrix:
    :param K:
    :param phenotype_array:
    :return:
    """
    # Score equation of the anatomical variance (Va).
    # Sg = -1/2 * trace(P*K) + 1/2*y'*P*K*P*y
    Sg = -0.5 * np.trace(np.dot(projection_matrix, K)) + 0.5 * np.dot(np.dot(np.dot(np.dot(phenotype_array.T, projection_matrix), K), projection_matrix), phenotype_array)

    # Score equation of the residual variance (Ve).
    # Se = -1/2 * trace(P) + 1/2*y'*P*P*y
    Se = -0.5 * np.trace(projection_matrix) + 0.5 * np.dot(np.dot(np.dot(phenotype_array.T, projection_matrix), projection_matrix), phenotype_array)

    S = np.array([
        Sg[0], Se[0]
    ])  # (2, 1)
    return S


def _compute_information_matrix(y: np.array, P: np.array, K: np.array):
    """
    Compute the information matrix based on average information.
    Igg = 1/2 * y'*P*K*P*K*P*y
    Ige = 1/2 * y'*P*K*P*P*y
    Iee = 1/2 * y'*P*P*P*y

    Parameters
    ----------
    :param y:
    :param P:
        Projection matrix.
    :param K:
    :return:
    """
    Igg = 0.5 * np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(y.T, P), K), P), K), P), y)
    Ige = 0.5 * np.dot(np.dot(np.dot(np.dot(np.dot(y.T, P), K), P), P), y)
    Iee = 0.5 * np.dot(np.dot(np.dot(np.dot(y.T, P), P), P), y)
    return np.array([
        [Igg[0][0], Ige[0][0]],
        [Ige[0][0], Iee[0][0]]
    ])


def _estimate_morphometricity(var_anatomical: np.float64, var_residual: np.float64):
    return var_anatomical / (var_anatomical + var_residual)


def _estimate_standard_error(m2, var_anatomical: np.float64, information_matrix: np.array) -> float:
    """
    SE = sqrt( (m2/Va)^2*((1-m2)^2*invI(1,1) - 2*(1-m2)*m2*invI(1,2) + m2^2*invI(2,2)) )
    :param m2: morphometricity score.
    :param var_anatomical:
    :param information_matrix: square matrix
    :return:
    """
    invI = la.inv(information_matrix)
    # try:
    #     invI = la.inv(I)
    # except np.linalg.LinAlgError as e:
    #     invI = la.pinv(I)
    return np.sqrt(
        ((m2 / var_anatomical)**2) *
        (
            (((1 - m2)**2) * invI[0, 0]) -
            (2 * (1 - m2) * m2) * invI[0, 1] +
            (m2**2) * invI[1, 1])
    )


def _mrdivide(B, A):  # return B/A
    """
    Matlab translation.
    :param B:
    :param A:
    :return:
    """
    return la.lstsq(A.T, B.T)[0].T


def _ensure_positive(value: np.float64, var_phenotype: np.float64) -> np.float64:
    """
    Sets any negative value to 1e-6 * Vp.
    Before we set all values below 1e-6 * Vp equal to that with np.maximum(value, 1e-6 * var_phenotype).
    :param value:
    :param var_phenotype:
    :return:
    """
    if value < 0:
        return np.float64(1e-6 * var_phenotype)
    else:
        return value
