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
        experiment_data: str = None,
        hostname: str = 'local'
) -> dict:
    """
    Load benchmark-specific configurations.

    Parameters
    ----------
    :param data_set_name:
        simulations: 'sim', 'd2', 'd3d', 'd3s', 'd4s', 'd6s', 'd9s', 'd15s'
        resting state data: 'HCP_PTN1200_recon2'
        task data: 'rockland'
    :param subset:
    :param subset_dimensionality:
    :param experiment_data:
        Expected in format 'Nxxx_Txxx'
    :param hostname:
    :return:
    """
    # Get filepaths.
    # filepaths_dict = _load_filepaths()
    # try:
        # filepaths_dict = filepaths_dict[hostname]
        # git_basedir = filepaths_dict['git-basedir']
        # project_basedir = filepaths_dict['project-basedir']
    # except KeyError:
        # logging.warning(f"Unexpected hostname '{hostname:s}' found.")
        # git_basedir = ''
        # project_basedir = ''

    # if 'login-' in hostname:  # TODO: this changes for each login
    #     git_basedir = ''
    #     project_basedir = ''

    # Define general configs, shared across experiments.
    shared_config_dict = dict()
    shared_config_dict['data-set-name'] = data_set_name
    # shared_config_dict['git-basedir'] = git_basedir
    # shared_config_dict['project-basedir'] = project_basedir
    match hostname:
        case 'hivemind':
            logging.warning(f"Running on '{hostname:s}'.")
            git_basedir = '/home/opk20/git_repos/FCEst-benchmarking'
            project_basedir = '/mnt/Data/neuro-dynamic-covariance'
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
        case 'sim':
            config_dict = _get_simulations_shared_config_dict(
                shared_config_dict=shared_config_dict,
                benchmark_dimensions=experiment_data
            )
        case 'd2':
            config_dict = _get_d2_config_dict(
                shared_config_dict=shared_config_dict,
                benchmark_dimensions=experiment_data
            )
        case 'd3d':
            config_dict = _get_d3d_config_dict(
                shared_config_dict=shared_config_dict,
                benchmark_dimensions=experiment_data
            )
        case 'd3s' | 'd4s' | 'd5s' | 'd6s' | 'd9s' | 'd15s':
            config_dict = _get_sparse_config_dict(
                shared_config_dict=shared_config_dict,
                benchmark_dimensions=experiment_data
            )
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

    Parameters
    ----------
    :param shared_config_dict:
    :param data_dimensionality:
        'd15' or 'd50'
    :return:
    """
    subset = f'3T_HCP1200_MSMAll_{data_dimensionality:s}_ts2'
    return {
        'data-dir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets',
            'resting_state', shared_config_dict['data-set-name'], 'node_timeseries', subset
        ),
        'data-dir-subject-measures': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets',
            'resting_state', 'hcp-openaccess'
        ),
        'experiments-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'experiments',
            'resting_state', shared_config_dict['data-set-name'], subset
        ),
        'figures-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'figures',
            'resting_state', shared_config_dict['data-set-name'], subset
        ),
        'git-results-basedir': os.path.join(
            shared_config_dict['git-basedir'], 'results', 'fmri', 'rs', 'HCP',
            shared_config_dict['data-set-name'], subset
        ),
        # 'ica-id-to-rsn-id-manual-map': _get_ica_id_to_rsn_id_manual_map()[data_dimensionality],
        'ica-id-to-rsn-id-algo-map': _get_ica_id_to_rsn_id_algorithmic_map(shared_config_dict)[data_dimensionality],
        'rsn-id-to-functional-region-map': _get_rsn_id_to_functional_region_map(shared_config_dict),
        'log-interval-svwp': 200,
        'max-n-cpus': 10,  # the maximum number of CPUs used on the Hivemind
        'mgarch-models': [
            'DCC',
            # 'GO',
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


def _get_simulations_shared_config_dict(shared_config_dict: dict, benchmark_dimensions: str) -> dict:
    """
    Define all global properties of simulated data sets.
    """
    simulated_data_dirpath = os.path.join(
        shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets',
        'simulations', shared_config_dict['data-set-name']
    )
    if os.path.exists(simulated_data_dirpath):
        logging.info("Existing data sets found:")
        print(sorted(os.listdir(simulated_data_dirpath)))
    return {
        'all-covs-types': [  # these are the ones we train on and may be different from the ones we report
            'null',
            'constant',
            'periodic_1',
            'periodic_2',
            'periodic_3',
            'periodic_4',
            'periodic_5',
            'boxcar',
            'stepwise',
            'state_transition',
            'change_point',
        ],
        'constant-covariance': 0.8,
        'data-dir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets',
            'simulations', shared_config_dict['data-set-name'], benchmark_dimensions
        ),
        'experiments-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'benchmarks',
            'simulations', shared_config_dict['data-set-name'], benchmark_dimensions
        ),
        'figures-basedir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'figures',
            'simulations', shared_config_dict['data-set-name'], benchmark_dimensions
        ),
        'git-results-basedir': os.path.join(
            shared_config_dict['git-basedir'], 'results', 'fmri', 'sim',
            shared_config_dict['data-set-name'], benchmark_dimensions
        ),
        'hcp-data-dir': os.path.join(
            shared_config_dict['project-basedir'], 'opk20_hivemind_paper_1', 'datasets',
            'resting_state'
        ),
        'figure-quantitative-results-dpi': 300,
        'figure-covariance-structures-dpi': 250,
        'figure-covariance-structures-figsize': (12, 4),
        'figure-model-predictions-dpi': 300,
        'figure-quantitative-results-figsize': (12, 3.5),
        'log-interval': 200,
        'max-n-cpus': 10,  # the maximum number of CPUs used on the Hivemind
        'n-inducing-points': 200,
        'noise-routines': [
            [None, None],
            # [None, 0.5],
            [None, 1],
            [None, 2],
            [None, 6],
            # [0.5, None],
            [1, None],
            [2, None],
            [6, None],
        ],
        'plot-covs-types': [  # these will be plotted in this order
            'null',
            'constant',
            'periodic_1',
            'periodic_3',
            'stepwise',
            'state_transition',
            'checkerboard',
        ],
        'plot-covs-types-palette': 'Set2',
        'plot-lengthscales-window-lengths': (12, 10),
        'plot-data-xlim': [-0.01, 1.01],
        'repetition-time': 1,  # synthetic TR is one second for simplicity
        'window-lengths': [
            15,
            30,
            60,
            120,
        ]
    }


def _get_d2_config_dict(
    shared_config_dict: dict, benchmark_dimensions: str
) -> dict:
    """
    Parameters
    ----------
    :param shared_config_dict:
    :param benchmark_dimensions:
    :return:
    """
    simulations_shared_config_dict = _get_simulations_shared_config_dict(shared_config_dict, benchmark_dimensions)
    config_dict = {
        'all-quantitative-results-models': [
            'VWP',
            'SVWP',
            'DCC_joint',
            'SW_cross_validated',
            'SW_15',
            'SW_30',
            'SW_60',
            'SW_120',
            'sFC',
        ],
        'figure-model-estimates-figsize': (12, 5),
        'mgarch-models': [
            'DCC',
            # 'GO',
        ],
        'mgarch-training-types': [
            'joint'
        ],
        'n-iterations-svwp': 14000,
        'n-iterations-vwp': 14000,
        'noise-types': [
            'no_noise',
            'HCP_noise_snr_2',
            'HCP_noise_snr_1',
            'HCP_noise_snr_6',
        ],
        'performance-metrics': [
            'covariance_RMSE',
            'correlation_RMSE',
            # 'covariance_correlation',
            'covariance_matrix_RMSE',
            'correlation_matrix_RMSE',
            # 'test_log_likelihood',
        ],
        'plot-models': [
            # 'VWP',
            'SVWP',
            'DCC_joint',
            'SW_cross_validated',
            # 'SW_15',
            # 'SW_30',
            # 'SW_60',
            # 'SW_120',
            'sFC',
        ],
        'plot-time-series-figsize': (12, 6),
    }
    # Merge general simulations and experiment-specific dictionaries.
    config_dict = config_dict | simulations_shared_config_dict
    return config_dict


def _get_d3d_config_dict(
    shared_config_dict: dict, benchmark_dimensions: str
) -> dict:
    """
    Parameters
    ----------
    :param shared_config_dict:
    :param benchmark_dimensions:
    :return:
    """
    simulations_shared_config_dict = _get_simulations_shared_config_dict(shared_config_dict, benchmark_dimensions)
    config_dict = {
        'all-quantitative-results-models': [
            'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            'SW_15',
            'SW_30',
            'SW_60',
            'SW_120',
            'sFC',
        ],
        'figure-model-estimates-figsize': (12, 13),
        'mgarch-models': [
            'DCC',
            # 'GO',
        ],
        'mgarch-training-types': [
            'joint',
            'bivariate_loop'
        ],
        'n-iterations-svwp': 14000,
        'n-iterations-vwp': 14000,
        'noise-types': [
            'no_noise',
            'HCP_noise_snr_2',
            'HCP_noise_snr_1',
            'HCP_noise_snr_6',
        ],
        'performance-metrics': [
            'covariance_matrix_RMSE',
            'correlation_matrix_RMSE',
            # 'test_log_likelihood',
        ],
        'plot-models': [
            # 'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_15',
            # 'SW_30',
            # 'SW_60',
            # 'SW_120',
            'sFC',
        ],
    }
    # Merge general simulations and experiment-specific dictionaries.
    config_dict = config_dict | simulations_shared_config_dict
    return config_dict


def _get_sparse_config_dict(
    shared_config_dict: dict, benchmark_dimensions: str
) -> dict:
    """
    Parameters
    ----------
    :param shared_config_dict:
    :param benchmark_dimensions:
    :return:
    """
    simulations_shared_config_dict = _get_simulations_shared_config_dict(shared_config_dict, benchmark_dimensions)
    config_dict = {
        'all-quantitative-results-models': [
            'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            'SW_15',
            'SW_30',
            'SW_60',
            'SW_120',
            'sFC'
        ],
        'figure-model-estimates-figsize': (12, 13),
        'mgarch-models': [
            'DCC',
            # 'GO',
        ],
        'mgarch-training-types': [
            'joint',
            'bivariate_loop'
        ],
        'n-inducing-points': 200,
        'n-iterations-svwp': 14000,
        'n-iterations-vwp': 14000,
        'noise-types': [
            'no_noise',
            'HCP_noise_snr_2',
            'HCP_noise_snr_1',
            'HCP_noise_snr_6',
        ],
        'performance-metrics': [
            'covariance_RMSE',
            'correlation_RMSE',
            # 'covariance_correlation',
            'covariance_matrix_RMSE',
            'correlation_matrix_RMSE',
            # 'test_log_likelihood',
        ],
        'plot-models': [
            # 'VWP_joint',
            'SVWP_joint',
            'DCC_joint',
            'DCC_bivariate_loop',
            'SW_cross_validated',
            # 'SW_15',
            # 'SW_30',
            # 'SW_60',
            # 'SW_120',
            'sFC',
        ],
    }
    # Merge general simulations and experiment-specific dictionaries.
    config_dict = config_dict | simulations_shared_config_dict
    return config_dict


def _get_rockland_config_dict(
        shared_config_dict: dict, repetition_time: str, roi_list_name: str = 'final'
) -> dict:
    """
    Rockland tb-fMRI specific configurations.

    Parameters
    ----------
    :param shared_config_dict:
    :param repetition_time:
    :param roi_list_name:
        Which set of ROIs we use; 'final', 'V1_V2_V3_V4_ACC'
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
            'results', 'fmri', 'tb', shared_config_dict['data-set-name'], subset
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
            # 'sFC',
        ],
        'subset': subset,
        'test-set-ratio': 0.2
    }


def _load_filepaths() -> dict:
    with open(os.path.join(
        'configs', 'filepaths.yaml'
    )) as f:
        filepaths = yaml.load(f, Loader=yaml.FullLoader)
    return filepaths


def _get_ica_id_to_rsn_id_manual_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'fmri', 'rs', 'HCP_PTN1200_recon2',
        'ICA_ID_to_RSN_ID_manual_map.yaml'
    )) as f:
        ica_id_to_rsn_id_manual_map = yaml.load(f, Loader=yaml.FullLoader)
    return ica_id_to_rsn_id_manual_map


def _get_ica_id_to_rsn_id_algorithmic_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'fmri', 'rs', 'HCP_PTN1200_recon2',
        'ICA_ID_to_RSN_ID_algorithmic_map.yaml'
    )) as f:
        ica_id_to_rsn_id_algorithmic_map = yaml.load(f, Loader=yaml.FullLoader)
    return ica_id_to_rsn_id_algorithmic_map


def _get_rsn_id_to_functional_region_map(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'fmri', 'rs', 'HCP_PTN1200_recon2',
        'RSN_ID_to_RSN_name_map.yaml'
    )) as f:
        rsn_id_to_functional_region_map = yaml.load(f, Loader=yaml.FullLoader)
    return rsn_id_to_functional_region_map


def _get_subject_measures(shared_config_dict: dict) -> dict:
    with open(os.path.join(
        shared_config_dict['git-basedir'], 'datasets', 'fmri', 'rs', 'HCP_PTN1200_recon2',
        'HCP_subject_measures.yaml'
    )) as f:
        subject_measures = yaml.load(f, Loader=yaml.FullLoader)
    return subject_measures
