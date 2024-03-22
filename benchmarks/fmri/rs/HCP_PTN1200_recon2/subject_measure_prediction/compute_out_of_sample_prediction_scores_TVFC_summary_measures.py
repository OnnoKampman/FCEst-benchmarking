import logging
import os
import socket
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects, get_human_connectome_project_subjects_phenotypes
from helpers.morphometricity import get_phenotype_array


if __name__ == '__main__':

    data_split = 'all'
    experiment_dimensionality = 'multivariate'
    metric = 'correlation'
    subject_features_type = 'TVFC_summary_measures'

    data_dimensionality = sys.argv[1]      # 'd15', 'd50'
    subject_measures_subset = sys.argv[2]  # 'cognitive', 'other', 'personality', 'psychiatric', or 'social-emotional'
    tvfc_estimation_method = sys.argv[3]  # 'SVWP_joint', 'DCC_joint', 'DCC_bivariate_loop', 'SW_cross_validated', 'SW_30', 'SW_60', or 'sFC'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    all_subjects_list = get_human_connectome_project_subjects(
        data_dir=cfg['data-dir'],
        first_n_subjects=cfg['n-subjects'],
        as_ints=True
    )
    n_permutations = 100  # the number of permutations you want to perform
    cv_loops = 5  # the number of cross-validation loops you want to perform
    k = 3  # the number of folds you want in the inner and outer folds of the nested cross-validation
    train_ratio = .8  # the proportion of data you want in your training set
    test_ratio = 1 - train_ratio

    subject_phenotypes_df = get_human_connectome_project_subjects_phenotypes(config_dict=cfg)
    subject_measures_list = cfg[f"subject-measures-{subject_measures_subset:s}"]
    if subject_measures_subset == 'cognitive':
        subject_measures_list = cfg['subject-measures-nuisance-variables'] + subject_measures_list

    n_tvfc_summary_measures = len(cfg['TVFC-summary-measures'])
    n_subject_measures = len(subject_measures_list)

    # Linear ridge model out-of-sample prediction task, following Dhamala et al. (2021).

    # We use the vectorized lower triangular values as predictive features, aiming to predict subject measures.
    # We run a linear ridge model with cross-validation to predict subject measures independently.
    # https://github.com/elvisha/CognitivePredictions

    for i_tvfc_summary_measure, tvfc_summary_measure in enumerate(cfg['TVFC-summary-measures']):

        # Only run the sFC (STATIC) model for the 'mean' TVFC summary measure.
        if tvfc_estimation_method == 'sFC' and tvfc_summary_measure != 'mean':
            logging.info(f"Not running sFC model on summary measure '{tvfc_summary_measure:s}'.")
            continue

        # These are the features to predict with.
        all_subjects_tvfc_summary_measure_array = []
        for scan_id in cfg['scan-ids']:
            tvfc_estimates_summaries_filepath = os.path.join(
                cfg['experiments-basedir'], 'TVFC_estimates_summary_measures', f'scan_{scan_id:d}',
                data_split, experiment_dimensionality, metric, tvfc_estimation_method, f'TVFC_{tvfc_summary_measure:s}.csv'
            )
            if not os.path.exists(tvfc_estimates_summaries_filepath):
                logging.error(f"TVFC estimates '{tvfc_estimates_summaries_filepath:s}' not found.")
                exit()
            tvfc_estimates_summaries_df = pd.read_csv(
                tvfc_estimates_summaries_filepath,
                index_col=0
            )  # (n_subjects, D*(D-1)/2)
            all_subjects_tvfc_summary_measure_array.append(tvfc_estimates_summaries_df.values)
        all_subjects_tvfc_summary_measure_array = np.array(all_subjects_tvfc_summary_measure_array)  # (n_scans, n_subjects, D*(D-1)/2)

        # Take the mean TVFC estimates over all scans.
        all_subjects_tvfc_summary_measure_array = np.mean(all_subjects_tvfc_summary_measure_array, axis=0)  # (n_subjects, D*(D-1)/2)

        #set x data to be the input variable you want to use
        X = all_subjects_tvfc_summary_measure_array  # (n_subjects, n_features)
        n_features = X.shape[1]

        #set hyperparameter grid space you want to search through for the model
        #alpha is the constant with which the regularization L2 term is multiplied with
        alphas = np.linspace(
            100, 1000, num=10,
            endpoint=True,
            dtype=None,
            axis=0
        )

        #set the param grid be to the hyperparamters you want to search through
        paramGrid ={'alpha': alphas}

        regr = Ridge(
            # normalize=True,
            max_iter=1000000,
            solver='auto',
        )

        #create arrays to store variables
        #r^2 - coefficient of determination
        r2 = np.zeros([n_permutations, n_subject_measures])
        #explained variance
        var = np.zeros([n_permutations, n_subject_measures])
        #correlation between true and predicted (aka prediction accuracy)
        corr = np.zeros([n_permutations, n_subject_measures])

        #optimised alpha (hyperparameter)
        opt_alpha = np.zeros([n_permutations, n_subject_measures])
        #predictions made by the model
        preds = np.zeros(
            [n_permutations, n_subject_measures, int(np.ceil(X.shape[0]*test_ratio))]
        )
        #true test values for cognition
        # cogtest = np.zeros(
        #     [n_permutations, n_subject_measures, int(np.ceil(X.shape[0]*test_ratio))]
        # )
        feature_importances = np.zeros([n_permutations, n_features, n_subject_measures])

        for i_subject_measure, subject_measure in enumerate(subject_measures_list):

            logging.info(f"> Subject measure {i_subject_measure+1:02d}/{n_subject_measures:d}: '{subject_measure:s}'")

            # This is the subject measure we want to predict.
            y = get_phenotype_array(
                phenotype_df=subject_phenotypes_df,
                subjects_subset_list=all_subjects_list,
                morphometricity_subject_measure=subject_measure
            )  # (n_subjects, 1)

            for i_permutation in range(n_permutations):

                print(f'Permutation {i_permutation + 1:03d} / {n_permutations:d}')

                #split data into train and test sets
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_ratio,
                    shuffle=True,
                    random_state=i_permutation
                )  # (n_subjects*train_ratio, n_features), (n_subjects*test_ratio, n_features), (n_subjects*train_ratio, 1), (n_subjects*test_ratio, 1)
                # print('x_train, y_train', x_train.shape, y_train.shape)
                # print('x_test , y_test ', x_test.shape, y_test.shape)

                #store all the y_test values in a separate variable that can be accessed later if needed
                # cogtest[i_permutation, i_subject_measure, :] = y_test

                #create variables to store nested CV scores, and best parameters from hyperparameter optimization
                nested_scores = []
                best_params = []
                
                #optimize regression model using nested CV
                #go through the loops of the cross validation
                # logging.info('Training models...')
                for i_cv_loop in range(cv_loops):

                    #set parameters for inner and outer loops for CV
                    inner_cv = KFold(
                        n_splits=k,
                        shuffle=True,
                        random_state=i_cv_loop
                    )
                    outer_cv = KFold(
                        n_splits=k,
                        shuffle=True,
                        random_state=i_cv_loop
                    )
                    
                    #define regressor with grid-search CV for inner loop
                    gridSearch = GridSearchCV(
                        estimator=regr,
                        param_grid=paramGrid,
                        n_jobs=-1,
                        verbose=0,
                        cv=inner_cv,
                        scoring='r2'
                    )

                    #fit regressor
                    gridSearch.fit(x_train, y_train)

                    #save parameters corresponding to the best score
                    best_params.append(
                        list(gridSearch.best_params_.values())
                    )

                    #call cross_val_score for outer loop
                    nested_score = cross_val_score(
                        gridSearch,
                        X=x_train,
                        y=y_train,
                        cv=outer_cv,
                        scoring='r2',
                        verbose=1
                    )

                    #record nested CV scores
                    nested_scores.append(np.median(nested_score))
                    
                #once all CV loops are complete, fit models based on optimized hyperparameters
                # logging.info('Testing models...')
                opt_alpha[i_permutation, i_subject_measure] = np.median(best_params)

                #fit model using optimized hyperparameter
                model = Ridge(
                    alpha = opt_alpha[i_permutation, i_subject_measure],
                    # normalize=True,
                    max_iter=1000000
                )
                model.fit(x_train, y_train)
                
                #compute r^2 (coefficient of determination)
                r2[i_permutation, i_subject_measure] = model.score(x_test, y_test)

                #generate predictions from model
                preds[i_permutation, i_subject_measure, :] = model.predict(x_test).ravel()
                
                #compute explained variance 
                var[i_permutation, i_subject_measure] = explained_variance_score(
                    y_test, preds[i_permutation, i_subject_measure, :]
                )

                #compute correlation between true and predicted
                corr[i_permutation, i_subject_measure] = np.corrcoef(
                    y_test.flatten(), preds[i_permutation, i_subject_measure, :]
                )[1,0]

                #extract feature importance
                feature_importances[i_permutation, :, i_subject_measure] = model.coef_

        linear_ridge_model_results_r2_df = pd.DataFrame(
            r2,
            index=[f'p_{i_permutation:d}' for i_permutation in np.arange(n_permutations)],
            columns=subject_measures_list
        )  # (n_permutations, n_subject_measures)
        print('\nR2 scores:\n')
        print(linear_ridge_model_results_r2_df)

        linear_ridge_model_results_explained_variance_df = pd.DataFrame(
            var,
            index=[f'p_{i_permutation:d}' for i_permutation in np.arange(n_permutations)],
            columns=subject_measures_list
        )  # (n_permutations, n_subject_measures)
        print('\nVariance explained scores:\n')
        print(linear_ridge_model_results_explained_variance_df)

        linear_ridge_model_results_prediction_accuracy_df = pd.DataFrame(
            corr,
            index=[f'p_{i_permutation:d}' for i_permutation in np.arange(n_permutations)],
            columns=subject_measures_list
        )  # (n_permutations, n_subject_measures)
        print('\nPrediction accuracy scores:\n')
        print(linear_ridge_model_results_prediction_accuracy_df)

        # np.savetxt('filepath/crystal_featimp.txt', featimp[:,:,0], delimiter=',')
        # np.savetxt('filepath/fluid_featimp.txt', featimp[:,:,1], delimiter=',')
        # np.savetxt('filepath/total_featimp.txt', featimp[:,:,2], delimiter=',')

        # np.savetxt('filepath/crystal_preds.txt', preds[:,0,:], delimiter=',')
        # np.savetxt('filepath/luid_preds.txt', preds[:,1,:], delimiter=',')
        # np.savetxt('filepath/total_preds.txt', preds[:,2,:], delimiter=',')

        # np.savetxt('filepath/crystal_cogtest.txt', cogtest[:,:,0], delimiter=',')
        # np.savetxt('filepath/fluid_cogtest.txt', cogtest[:,:,1], delimiter=',')
        # np.savetxt('filepath/total_cogtest.txt', cogtest[:,:,2], delimiter=',')

        scores_savedir = os.path.join(
            cfg['git-results-basedir'], 'subject_measure_prediction',
            'out_of_sample_prediction', subject_measures_subset
        )
        if not os.path.exists(scores_savedir):
            os.makedirs(scores_savedir)

        linear_ridge_model_results_r2_df.to_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_linear_ridge_model_r2_scores_TVFC_{tvfc_summary_measure:s}_{tvfc_estimation_method:s}.csv'
            ),
            float_format="%.7f"
        )
        linear_ridge_model_results_explained_variance_df.to_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_linear_ridge_model_variance_explained_TVFC_{tvfc_summary_measure:s}_{tvfc_estimation_method:s}.csv'
            ),
            float_format="%.7f"
        )
        linear_ridge_model_results_prediction_accuracy_df.to_csv(
            os.path.join(
                scores_savedir, f'{metric:s}_linear_ridge_model_prediction_accuracy_TVFC_{tvfc_summary_measure:s}_{tvfc_estimation_method:s}.csv'
            ),
            float_format="%.7f"
        )
