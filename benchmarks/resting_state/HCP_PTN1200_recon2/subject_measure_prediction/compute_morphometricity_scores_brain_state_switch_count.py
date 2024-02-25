import logging
import os
import socket
import sys

import pandas as pd

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects, get_human_connectome_project_subjects_phenotypes
from helpers.morphometricity import variance_component_model, get_phenotype_array, get_covariates_array
from helpers.subject_similarity_matrix import get_brain_state_switch_count_similarity_matrix


if __name__ == '__main__':

    experiment_dimensionality = 'multivariate'
    metric = 'correlation'
    subject_similarity_type = 'brain_state_switch_count'

    data_dimensionality = sys.argv[1]      # 'd15', 'd50'
    subject_measures_subset = sys.argv[2]  # 'cognitive', 'other', 'personality', 'psychiatric', or 'social-emotional'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_subjects = cfg['n-subjects']
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=n_subjects,
        as_ints=True
    )

    subject_phenotypes_df = get_human_connectome_project_subjects_phenotypes(config_dict=cfg)
    subject_measures_list = cfg['subject-measures-nuisance-variables'] + cfg[f"subject-measures-{subject_measures_subset:s}"]

    n_subject_measures = len(subject_measures_list)

    morphometricity_results_df = pd.DataFrame()
    morphometricity_results_standard_error_df = pd.DataFrame()

    for i_subject_measure, subject_measure in enumerate(subject_measures_list):

        print(f"> Subject measure {i_subject_measure+1:02d}/{n_subject_measures:d}: '{subject_measure:s}'")

        for morphometricity_model in cfg['morphometricity-models']:

            # Do not run the sFC (STATIC) model.
            if morphometricity_model == 'sFC':
                continue

            y = get_phenotype_array(
                phenotype_df=subject_phenotypes_df,
                subjects_subset_list=all_subjects_list,
                morphometricity_subject_measure=subject_measure
            )  # (n_subjects, 1)
            X = get_covariates_array(
                phenotype_df=subject_phenotypes_df,
                subjects_subset_list=all_subjects_list,
                nuisance_variables=cfg['subject-measures-nuisance-variables'].copy(),  # do not edit original list
                morphometricity_subject_measure=subject_measure
            )  # (n_subjects, n_covariates)
            K = get_brain_state_switch_count_similarity_matrix(
                config_dict=cfg,
                n_basis_states=3,
                tvfc_estimation_method=morphometricity_model
            )  # (n_subjects, n_subjects)

            # Run variance component model.
            try:
                result = variance_component_model(phenotype_array=y, X=X, K=K)
                morphometricity_score = result['m2']
                morphometricity_score_standard_error = result['SE']
                logging.info(f"Morphometricity score for '{subject_measure:s}': {morphometricity_score:.3f}")
                logging.info(f"Morphometricity score standard error for '{subject_measure:s}': {morphometricity_score_standard_error:.3f}")
            except Exception as e:
                print(e)
                logging.warning("Could not compute morphometricity, setting it to zero.")
                morphometricity_score = 0
                morphometricity_score_standard_error = 0

            morphometricity_results_df.loc[subject_measure, morphometricity_model] = morphometricity_score
            morphometricity_results_standard_error_df.loc[subject_measure, morphometricity_model] = morphometricity_score_standard_error

    print(morphometricity_results_df)

    scores_savedir = os.path.join(
        cfg['git-results-basedir'], 'subject_measure_prediction',
        'morphometricity', subject_measures_subset
    )
    if not os.path.exists(scores_savedir):
        os.makedirs(scores_savedir)

    morphometricity_results_df.to_csv(
        os.path.join(scores_savedir, f'{metric:s}_morphometricity_scores_{subject_similarity_type:s}.csv'),
        float_format="%.3f"
    )
    morphometricity_results_standard_error_df.to_csv(
        os.path.join(scores_savedir, f'{metric:s}_morphometricity_scores_{subject_similarity_type:s}_standard_error.csv'),
        float_format="%.3f"
    )
