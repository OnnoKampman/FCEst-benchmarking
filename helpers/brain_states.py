import logging
import os

import math
import numpy as np
import pandas as pd
from scipy import cluster
from sklearn.cluster import KMeans

from helpers.array_operations import reconstruct_symmetric_matrix_from_tril


def compute_basis_state(
        config_dict: dict, all_subjects_tril_tvfc: np.array,
        model_name: str, n_basis_states: int,
        n_time_series: int, n_time_steps: int,
        scan_session_id: int = None, analysis_type: str = None, medication_status: str = None,
        connectivity_metric: str = 'correlation'
) -> (float, pd.DataFrame, pd.DataFrame):
    """
    Brain states (recurring whole-brain patterns) are a way to summarize estimated TVFC.
    Here we follow Allen2012 and use k-means clustering to find the brain states across subjects,
    independently for each of the two runs within each of the two session.
    TODO: add tSNE plot of all states into clusters to see how distinct they really are
    TODO: save how well data is explained, since WP brain states are much higher contrast, we may expect more explanation

    :param config_dict:
    :param all_subjects_tril_tvfc: (n_subjects * N, D*(D-1)/2)
    :param model_name:
    :param n_basis_states: the number of states/clusters we want to extract.
        Ideally this number would be determined automatically, or based on some k-means elbow analysis.
    :param n_time_series:
    :param n_time_steps:
    :param scan_session_id: 0, 1, 2, or 3.
    :param analysis_type:
    :param medication_status:
    :param connectivity_metric:
    :return:
    """
    logging.info("Running k-means clustering...")
    kmeans = KMeans(
        n_clusters=n_basis_states,
        algorithm='lloyd',
        n_init=10,
        verbose=0
    ).fit(all_subjects_tril_tvfc)
    logging.info("Finished k-means clustering.")

    cluster_centers = _get_cluster_centroids(
        kmeans=kmeans,
        n_time_series=n_time_series
    )

    cluster_centers, cluster_sort_order = _sort_cluster_centers(
        cluster_centers=cluster_centers
    )

    # Save clusters (i.e. brain states) to file.
    if scan_session_id is not None:
        brain_states_savedir = os.path.join(
            config_dict['git-results-basedir'], 'brain_states',
            f'k{n_basis_states:02d}', f'scan_{scan_session_id:d}'
        )
    else:
        assert analysis_type is not None
        assert medication_status is not None
        brain_states_savedir = os.path.join(
            config_dict['git-results-basedir'], 'brain_states',
            f"{analysis_type:s}_{medication_status:s}".removesuffix('_'),
            f'k{n_basis_states:02d}'
        )
    if not os.path.exists(brain_states_savedir):
        os.makedirs(brain_states_savedir)
    for i_cluster, cluster_centroid in enumerate(cluster_centers):
        cluster_df = pd.DataFrame(cluster_centroid)  # (D, D)
        cluster_df.to_csv(
            os.path.join(
                brain_states_savedir,
                f'{connectivity_metric:s}_brain_state_{i_cluster:d}_{model_name:s}.csv'
            ),
            float_format='%.2f'
        )
        logging.info(f"Brain state saved in '{brain_states_savedir:s}'.")

    # TODO: Extract and re-order brain state assignments.
    all_subjects_brain_state_assignments_df = _get_brain_state_assignments(
        labels=kmeans.labels_,
        n_time_steps=n_time_steps
    )  # (n_subjects, N)
    print("Brain state assignments:")
    print(all_subjects_brain_state_assignments_df.head())

    all_subjects_brain_state_assignments_df = all_subjects_brain_state_assignments_df.replace(
        to_replace=cluster_sort_order,
        value=np.arange(n_basis_states)
    )
    print(cluster_sort_order)
    print("Brain state assignments (sorted):")
    print(all_subjects_brain_state_assignments_df.head())

    all_subjects_dwell_times_df = _compute_dwell_time(
        config_dict=config_dict,
        n_brain_states=n_basis_states,
        brain_state_assignments=all_subjects_brain_state_assignments_df
    )  # (n_subjects, n_brain_states)

    return kmeans.inertia_, all_subjects_brain_state_assignments_df, all_subjects_dwell_times_df


def _get_cluster_centroids(kmeans: KMeans, n_time_series: int) -> np.array:
    # Get cluster centers - these are the characteristic basis brain states.
    cluster_centers = kmeans.cluster_centers_  # (n_clusters, n_features)

    # Reconstruct correlation matrix per cluster.
    cluster_centers = [
        reconstruct_symmetric_matrix_from_tril(cluster_vector, n_time_series) for cluster_vector in cluster_centers
    ]
    cluster_centers = np.array(cluster_centers)  # (n_clusters, D, D)

    return cluster_centers


def _sort_cluster_centers(cluster_centers: np.array) -> np.array:

    # Re-order clusters based on high to low (descending) 'contrast' - higher contrast states are more interesting.
    cluster_contrasts = np.var(cluster_centers, axis=(1, 2))  # (n_clusters, )
    cluster_sort_order = np.argsort(cluster_contrasts)[::-1]  # (n_clusters, )

    cluster_centers = cluster_centers[cluster_sort_order, :, :]  # (n_clusters, D, D)

    return cluster_centers, cluster_sort_order


def _get_brain_state_assignments(
        labels, n_time_steps: int
) -> pd.DataFrame:
    """
    Get labels per covariance matrix, which can be used to re-construct a states time series per subject.
    Each subject is now only characterized by an assignment to one of the clusters at each time step.

    :param labels: array of shape (n_subjects * N, )
    :param n_time_steps:
    :return:
    """
    assert len(labels) % n_time_steps == 0
    n_subjects = int(len(labels) / n_time_steps)

    labels = labels.reshape(n_subjects, n_time_steps)  # (n_subjects, N)

    labels_df = pd.DataFrame(labels)

    return labels_df


def _compute_dwell_time(
        config_dict: dict, n_brain_states: int, brain_state_assignments: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute dwell times: proportion of scan each subject spends in each state.

    :param n_brain_states:
    :param brain_state_assignments: (n_subjects, N)
    """
    n_time_steps = brain_state_assignments.shape[1]  # N
    assert n_time_steps == config_dict['n-time-steps']
    n_subjects = len(brain_state_assignments)

    all_subjects_dwell_times_df = pd.DataFrame(
        0,
        index=np.arange(n_subjects),
        columns=np.arange(n_brain_states)
    )

    for i_subject in range(n_subjects):
        subject_brain_state_assignments = brain_state_assignments.iloc[i_subject, :]  # pd.Series of shape (N, )
        subject_dwell_times = subject_brain_state_assignments.value_counts(sort=False)
        if i_subject == 0:
            print(subject_dwell_times)
        all_subjects_dwell_times_df.iloc[i_subject, subject_dwell_times.index] = subject_dwell_times.values / n_time_steps

        assert math.isclose(
            all_subjects_dwell_times_df.iloc[i_subject, :].sum(), 1.0,
            abs_tol=1e-2
        )

    return all_subjects_dwell_times_df


def _compute_occupancy_rates():
    """
    Compute occupancy rates.
    """
    raise NotImplementedError


def match_brain_states():
    """
    Brain states were matched across runs, sessions, and dynamic FC methods by maximizing their spatial correlation.
    """
    raise NotImplementedError


def extract_number_of_brain_state_switches(brain_state_assignment_df: pd.DataFrame) -> pd.Series:
    """
    Compute the number of switches in brain state.

    :param brain_state_assignment_df:
    :return:
    """
    change_point_counts_series = pd.Series(
        [len([i for i in range(1, len(x)) if x[i] != x[i-1]]) for x in brain_state_assignment_df.values],
        index=brain_state_assignment_df.index
    )  # (n_subjects, )
    return change_point_counts_series
