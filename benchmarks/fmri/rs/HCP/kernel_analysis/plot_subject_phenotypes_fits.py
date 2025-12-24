import logging
import os
import socket
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

from configs.configs import get_config_dict
from helpers.hcp import get_human_connectome_project_subjects_phenotypes

# TODO: find better way of doing this, e.g. variance explained models
# TODO: it seems like longer lengthscales are generally related to worse performance on tasks, but more work is needed

sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = 'serif'


if __name__ == "__main__":

    kernel_param = 'kernel_lengthscales'

    subject_measures_subset = sys.argv[1]  # 'cognitive', 'social-emotional', 'personality', 'psychiatric', or 'other'
    data_dimensionality = sys.argv[2]      # 'd15', 'd50'

    cfg = get_config_dict(
        data_set_name='HCP_PTN1200_recon2',
        subset_dimensionality=data_dimensionality,
        hostname=socket.gethostname()
    )
    n_time_series = int(data_dimensionality[1:])

    figures_savedir = os.path.join(cfg['figures-basedir'], 'kernel_analysis', 'cognitive_measures', subject_measures_subset)
    if not os.path.exists(figures_savedir):
        os.makedirs(figures_savedir)

    # The kernel parameters were saved with the experiments in the git repo.
    kernel_params_savedir = os.path.join(cfg['git-results-basedir'], 'kernel_analysis')
    kernel_params_df_filename = f'{kernel_param:s}_kernel_params.csv'
    kernel_params_df = pd.read_csv(
        os.path.join(kernel_params_savedir, kernel_params_df_filename),
        index_col=0
    )
    print(kernel_params_df.head())

    subjects_list = kernel_params_df.index.values  # np.array of ints of shape (n_subjects, )
    # subjects_list = [int(subject_filename[:-4]) for subject_filename in subjects_list]
    print(subjects_list)
    print(type(subjects_list))

    # Load subject phenotypes.
    subject_phenotypes_df = get_human_connectome_project_subjects_phenotypes(config_dict=cfg)

    for phenotype in cfg[f"subject-measures-{subject_measures_subset:s}"]:
        print(phenotype)
        phenotype_array = subject_phenotypes_df.loc[subjects_list, phenotype]  # floats array

        # Impute NaN entries as the mean of all subjects.
        n_nans = phenotype_array.isna().sum()
        print(f'Found {n_nans:d} NaNs.')
        if n_nans > 0:
            phenotype_array[phenotype_array.isna()] = phenotype_array.mean()

        # We have 4 scans but the phenotypes are the same.
        phenotype_array = np.tile(phenotype_array, 4)  # (n_subjects * n_scans, )

        kernel_params_array = kernel_params_df.values.reshape(-1, 1)  # (n_subjects * n_scans, 1)
        print('kernel params array:', kernel_params_array.shape)

        if phenotype_array.dtype != object:  # that is, a string
            plt.scatter(
                kernel_params_array,
                phenotype_array,
                marker='x',
                s=3.5
            )
            plt.xticks(fontsize=14)
            plt.xlabel('kernel lengthscale', fontsize=16)
            plt.yticks(fontsize=14)
            plt.ylabel(phenotype, fontsize=16)

            X = kernel_params_array
            y = phenotype_array.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X, y)
            print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

            n_prediction_locations = 40
            xx = np.linspace(np.min(X), np.max(X), n_prediction_locations).reshape(-1, 1)  # (n_prediction_locations, 1)
            predictions = reg.predict(xx)
            plt.plot(
                xx,
                predictions,
                c='blue',
                linewidth=2
            )
            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()
            print(est2.summary())
            p_value_slope = est2.pvalues[1]
            print(p_value_slope)
            print(est2.pvalues)

            # Add slope p-value to plot.
            slope_text_label = f'p={p_value_slope:.2f}'
            alpha = 0.05
            if p_value_slope < alpha:
                slope_text_label += '*'
            plt.text(xx[-1], predictions[-1], slope_text_label)
        else:
            # TODO: convert to violin plot
            df = pd.DataFrame(kernel_params_array, index=phenotype_array)
            df = df.groupby(df.index).mean()
            # df = df.groupby(df.index).append()
            print(df)
            # sns.violinplot(
            #     data=df,
            #     data=[df[df.index == 'M'], df[df.index == 'F']],
            #     palette="light:g",
            #     inner="points",  # 'points', 'stick'
            #     orient="v",
            #     cut=2,
            #     scale_hue=False,
            #     bw=.2
            # )
            df.plot.bar(legend=False, rot=0)
            plt.xticks(fontsize=14)
            plt.xlabel('category', fontsize=16)
            plt.yticks(fontsize=14)
            plt.ylabel('mean kernel lengthscale', fontsize=16)

        plt.tight_layout()

        phenotype_regression_filename = f'{phenotype}.png'
        plt.savefig(os.path.join(figures_savedir, phenotype_regression_filename), dpi=200)
        logging.info(f"Saved phenotype figures '{phenotype_regression_filename:s}' to '{figures_savedir:s}'.")
        plt.close()
