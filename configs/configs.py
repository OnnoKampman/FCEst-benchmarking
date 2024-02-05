import logging
import os
from pprint import pprint
import yaml

import numpy as np

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)


def get_config_dict(
        data_set_name: str,
        subset: str = None,
        subset_dimensionality: str = None,
        hostname: str = 'local'
) -> dict:
    """
    :param data_set_name:
        resting state data: 'HCP_PTN1200_recon2'
        task data: 'rockland'
    :param subset:
    :param subset_dimensionality:
    :param hostname:
    :return:
    """
    # Define general configs, shared across experiments.
    shared_config_dict = dict()
    shared_config_dict['data-set-name'] = data_set_name
    match hostname:
        case _:
            logging.warning(f"Unexpected hostname '{hostname:s}' found.")
    if 'login-' in hostname:  # TODO: this changes for each login
        git_basedir = ''
        project_basedir = ''
    shared_config_dict['git-basedir'] = git_basedir
    shared_config_dict['project-basedir'] = project_basedir

    # Global configs.
    shared_config_dict['kernel-params'] = [
        'kernel_variance',
        'kernel_lengthscales'
    ]
    shared_config_dict['n-brain-states-list'] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21
    ]
    shared_config_dict['n-inducing-points'] = 200
    shared_config_dict['significance-alpha'] = 0.05

    # General figures configs.
    shared_config_dict['figure-dpi-imputation'] = 250
    shared_config_dict['plot-likelihoods-figsize'] = (12, 6)
    shared_config_dict['plot-brain-state-switch-count-figsize'] = (12, 8)
    shared_config_dict['plot-time-series-dpi'] = 200
    shared_config_dict['plot-time-series-ylim'] = [-3.5, 3.5]
    shared_config_dict['plot-methods-palette'] = 'deep'  # deeper version of default Matplotlib
    shared_config_dict['wp-n-predict-samples'] = 800

    # Get experiment-specific configs.
    match data_set_name:
        case 'HCP_PTN1200_recon2':
            config_dict = _get_human_connectome_project_config_dict(
                shared_config_dict=shared_config_dict,
                data_dimensionality=subset_dimensionality
            )
        case 'rockland':
            config_dict = _get_rockland_config_dict(
                shared_config_dict=shared_config_dict,
                repetition_time=subset
            )
        case _:
            raise NotImplementedError(f"Data set name '{data_set_name:s}' not recognized.")

    # Merge global and experiment-specific dictionaries.
    config_dict = config_dict | shared_config_dict

    pprint(config_dict)
    return config_dict


def _get_human_connectome_project_config_dict(
        shared_config_dict: dict, data_dimensionality: str
) -> dict:
    """
    Get configs dictionary for Human Connectome Project (HCP) data.
    :param shared_config_dict:
    :param data_dimensionality: 'd15' or 'd50'
    :return:
    """
    subset = f'3T_HCP1200_MSMAll_{data_dimensionality:s}_ts2'
    return {
        'data-dir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets', 'resting_state',
            shared_config_dict['data-set-name'], 'node_timeseries', subset
        ),
        'data-dir-subject-measures': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets', 'resting_state', 'hcp-openaccess'
        ),
        'experiments-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'experiments', 'resting_state',
            shared_config_dict['data-set-name'], subset
        ),
        'figures-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'figures', 'resting_state',
            shared_config_dict['data-set-name'], subset
        ),
        'git-results-basedir': os.path.join(
            shared_config_dict['git-basedir'], 'results', 'resting_state',
            shared_config_dict['data-set-name'], subset
        ),
        # 'ica-id-to-rsn-id-manual-map': _get_ica_id_to_rsn_id_manual_map()[data_dimensionality],
        'ica-id-to-rsn-id-algo-map': _get_ica_id_to_rsn_id_algorithmic_map(shared_config_dict)[data_dimensionality],
        'rsn-id-to-functional-region-map': _get_rsn_id_to_functional_region_map(shared_config_dict),
        'log-interval-svwp': 200,
        'max-n-cpus': 10,  # the maximum number of CPUs used on the Hivemind
        'mgarch-models': [
            'DCC',
            # 'GO'
        ],
        'mgarch-training-types': [
            'bivariate_loop',
            'joint'
        ],
        'models-brain-state-analysis': [
            'SVWP_joint',
            'DCC_joint',
            'SW_cross_validated',
            # 'SW_30',
            # 'SW_60',
        ],
        'morphometricity-models': [
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_30',
            'SW_60',
            'SW_cross_validated',
            'sFC',
        ],
        # For nuisance variables we want to include things that are known to predict TVFC.
        'subject-measures-nuisance-variables': [
            'Age',
            'Gender',
            # 'Race',
            # 'SSAGA_Educ'  # years of education
            # 'BMI',
            # 'BPSystolic',
            # 'BPDiastolic',
            # '',  # FD, frame-wise displacement, (head) movement
            # '',  # root-mean-square of voxel-wise differentiated signal (DVARS)
        ],
        'subject-measures-cognitive': _get_subject_measures(shared_config_dict)['cognitive'],
        'subject-measures-social-emotional': _get_subject_measures(shared_config_dict)['social-emotional'],
        'subject-measures-personality': _get_subject_measures(shared_config_dict)['personality'],
        'subject-measures-psychiatric': _get_subject_measures(shared_config_dict)['psychiatric'],
        'subject-measures-other': _get_subject_measures(shared_config_dict)['other'],
        'n-inducing-points': 200,
        'n-iterations': 8000,
        'n-subjects': 812,
        'n-time-steps': 1200,
        'phenotypes-restricted-filename': 'RESTRICTED_opk20_12_3_2021_9_5_4.csv',
        'phenotypes-unrestricted-filename': 'unrestricted_opk20_5_5_2021_12_24_50.csv',
        'phenotypes-restricted': [
            'Age_in_Yrs'  # Age by year is in the restricted access data,
            # 'FamHist_Moth_Dep',
            # 'FamHist_Fath_Dep',
            # 'ASR_Anxd_Raw',
            # 'ASR_Anxd_Pct',
            # 'ASR_Intn_Raw',
            # 'ASR_Intn_T',
            # 'DSM_Depr_Raw',
            'DSM_Depr_Pct',
            # 'DSM_Anxi_Raw',
            # 'DSM_Anxi_Pct',
            # 'SSAGA_Depressive_Ep',
            # 'SSAGA_Depressive_Sx',
        ],
        'phenotypes-unrestricted': [
            # 'Age',  # Age in five year ranges is in unrestricted access data
            'Gender'
        ],
        'plot-models': [
            'SVWP_joint',
            'DCC_joint',
            # 'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_30',
            # 'SW_60',
            'sFC',
        ],
        'plot-model-estimates-figsize': (12, 8),
        'plot-model-estimates-methods': [
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_30',
            # 'SW_60',
            'sFC',
        ],
        'plot-model-estimates-summary-measures-methods': [
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            'SW_30',
            'SW_60',
            # 'sFC',
        ],
        'plot-morphometricity-scores-figsize': (12, 10),
        'plot-subject-measures-figsize': (7.2, 6),
        'plot-time-series-figsize': (12, 13),
        'plot-time-series-xlim': [-0.2, 14.6],  # in minutes
        'plot-TVFC-summary-measures': [
            'mean',
            'variance',
            'rate_of_change',
        ],
        'repetition-time': 0.72,  # TR, in seconds
        'scan-ids': [0, 1, 2, 3],
        'SW-window-lengths': [  # in seconds
            # 15,
            30,
            60,
            # 120,
        ],
        'subset': subset,
        'TVFC-summary-measures': [
            # 'ALFF',  # amplitude of low frequency fluctuations (Shen et al., 2014; Qin et al., 2015; Zhang et al., 2020)
            'ar1',
            'mean',
            'variance',
            'rate_of_change',
            # 'std',
        ],
        'test-retest-models': [
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            'SW_30',
            'SW_60',
            'sFC',
        ]
    }


def _get_rockland_config_dict(
        shared_config_dict: dict, repetition_time: str, roi_list_name: str = 'final'
) -> dict:
    """
    Rockland tb-fMRI specific configurations.
    :param shared_config_dict:
    :param repetition_time:
    :param roi_list_name: which set of ROIs we use; 'final', 'V1_V2_V3_V4_ACC'
    :return:
    """
    subset = f"CHECKERBOARD{repetition_time:s}"
    return {
        'cutoff-v1-stim-correlation': 0.4,
        'data-basedir': os.path.join(
            shared_config_dict['project-basedir'],
            'opk20_hivemind_paper_1', 'datasets', 'task', shared_config_dict['data-set-name'], subset
        ),
        'experiments-basedir': os.path.join(
            shared_config_dict['project-basedir'],
            'opk20_hivemind_paper_1', 'benchmarks', 'task', shared_config_dict['data-set-name'], subset
        ),
        'figures-basedir': os.path.join(
            shared_config_dict['project-basedir'],
            'opk20_hivemind_paper_1', 'figures', 'task', shared_config_dict['data-set-name'], subset
        ),
        'git-results-basedir': os.path.join(
            shared_config_dict['git-basedir'],
            'results', 'task', shared_config_dict['data-set-name'], subset
        ),
        'log-interval': 100,
        'max-n-cpus': 10,
        'mgarch-models': [
            'DCC'
        ],
        'mgarch-training-types': [
            'joint',
            'bivariate_loop'
        ],
        'n-iterations': 14000,
        'n-iterations-svwp': 14000,
        'n-time-steps': 240,
        'plot-likelihoods-models': [
            # 'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            # 'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_16',  # in seconds, based on Di2015
            # 'SW_30',
            # 'SW_60',
            'sFC'
        ],
        'plot-stimulus-prediction-models': [
            # 'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            # 'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_16',  # in seconds, based on Di2015
            # 'SW_30',
            # 'SW_60',
            # 'sFC'
        ],
        'plot-time-series-figsize': (12, 6),
        'repetition-time': 0.645,  # in seconds (either 1.4 or 0.645)
        'roi-list': np.array(['V1', 'V2', 'V3', 'V4', 'mPFC', 'M1']),
        'roi-list-name': roi_list_name,
        'roi-edges-list': np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5]
        ]),
        'stimulus-prediction-models': [
            'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            'SW_16',  # in seconds, based on Di2015
            'SW_30',
            'SW_60',
            # 'sFC'
        ],
        'subset': subset,
        'test-set-ratio': 0.2
    }


def _get_ica_id_to_rsn_id_manual_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'resting_state', 'HCP_PTN1200_recon2',
        'ICA_ID_to_RSN_ID_manual_map.yaml'
    )) as f:
        ica_id_to_rsn_id_manual_map = yaml.load(f, Loader=yaml.FullLoader)
    return ica_id_to_rsn_id_manual_map


def _get_ica_id_to_rsn_id_algorithmic_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'resting_state', 'HCP_PTN1200_recon2',
        'ICA_ID_to_RSN_ID_algorithmic_map.yaml'
    )) as f:
        ica_id_to_rsn_id_algorithmic_map = yaml.load(f, Loader=yaml.FullLoader)
    return ica_id_to_rsn_id_algorithmic_map


def _get_rsn_id_to_functional_region_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'resting_state', 'HCP_PTN1200_recon2',
        'RSN_ID_to_RSN_name_map.yaml'
    )) as f:
        rsn_id_to_functional_region_map = yaml.load(f, Loader=yaml.FullLoader)
    return rsn_id_to_functional_region_map


def _get_subject_measures(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'resting_state', 'HCP_PTN1200_recon2',
        'HCP_subject_measures.yaml'
    )) as f:
        subject_measures = yaml.load(f, Loader=yaml.FullLoader)
    return subject_measures
