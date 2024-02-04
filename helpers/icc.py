import numpy as np
import pandas as pd
import pingouin as pg


def compute_icc_scores_pingouin(observations: np.array, icc_type: str):
    """
    Computes ICC scores using third-party library.

    :param observations:
    :param icc_type: "ICC1", "ICC2", "ICC3"
    :return:
    """
    icc_df = _get_pingouin_icc_scores_df(observations=observations)
    icc_score = icc_df.loc[icc_type, 'ICC']
    return icc_score


def _get_pingouin_icc_scores_df(observations: np.array) -> pd.DataFrame:
    """
    TODO: we may want to interpret between-subject variance directly, so we should extract that from pingouin function

    :param observations: array of shape [n_subject, n_sessions]
    :return:
    """
    df = pd.DataFrame({
        'Subject': np.tile(np.arange(observations.shape[0]), observations.shape[1]),
        'Session': np.floor(np.linspace(0, observations.shape[1], observations.shape[0] * observations.shape[1] + 1)[:-1]).astype(int)
    })
    df['Measure'] = observations[df['Subject'], df['Session']]
    df['Subject'] += 1  # make one-index
    df['Session'] += 1  # make one-index

    icc_df = pg.intraclass_corr(
        data=df,
        targets='Subject', raters='Session', ratings='Measure',
        nan_policy='raise'
    )
    icc_df.set_index('Type', inplace=True)
    print(icc_df.round(4))
    return icc_df


def compute_ICC(observations: np.array, icc_type="icc2"):
    """
    Calculate intraclass correlation coefficient for data within
        Brain_Data class
    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

    From Shou et al. (2013):
        Estimation is simple; σW2 can be estimated as the variance of Wij,and σU2 can be estimated by the variance of (Wi2 − Wi1)/2.

    Code modifed from nipype algorithms.icc
        https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    Args:
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    """
    Y = observations
    [n_subjects, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n_subjects - 1) * dfc
    dfr = n_subjects - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(n_subjects), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    # residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n_subjects
    MSC = SSC / dfc / n_subjects

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc1":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == "icc2":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + dfc * MSE + k * (MSC - MSE) / n_subjects)

        # ICC = (MSR)

    elif icc_type == "icc3":
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + dfc * MSE)
    else:
        print('error: ICC type not recognized.')

    return ICC
