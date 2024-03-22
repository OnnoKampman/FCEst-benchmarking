import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from helpers.data import normalize_array


def load_human_connectome_project_data(
        data_file: str, scan_id: int, scan_length: int = 1200, verbose: bool = True
) -> (np.array, np.array):
    """
    Load Human Connectome Project (HCP) data files.
    TODO: add option to only select (family) unrelated subjects

    :param data_file:
    :param scan_id:
    :param scan_length:
    :param verbose:
    :return:
    """
    df = pd.read_csv(data_file, header=None, delimiter=' ')
    scan_start = scan_id * scan_length
    df = df.iloc[scan_start:(scan_start+scan_length), :]  # (N, D)
    for i in range(df.shape[1]):
        df.iloc[:, i] = normalize_array(df.iloc[:, i], verbose=verbose)
    if verbose:
        logging.info(f"Loaded data from '{data_file:s}'.")
        print(df.head())
        print(df.shape)
    n_time_steps = len(df)
    xx = np.linspace(0, 1, n_time_steps).reshape(-1, 1).astype(np.float64)
    return xx, df.values


def get_human_connectome_project_subjects(
        data_dir: str,
        first_n_subjects: int = None,
        as_ints: bool = False,
) -> list[str]:
    """
    Returns the full list of HCP subjects.
    TODO: add option to only select (family) unrelated subjects

    Parameters
    ----------
    :param data_dir:
    :param first_n_subjects:
    :param as_ints:
        Whether or not to return a list of strings (e.g. '123456.txt') or integers (e.g. 123456).
    :return:
    """
    all_subjects_list = sorted(os.listdir(data_dir))  # list of all files in this directory, alphabetically sorted
    if first_n_subjects is not None:
        all_subjects_list = all_subjects_list[:first_n_subjects]
    if as_ints:
        all_subjects_list = [int(subj.removesuffix('.txt')) for subj in all_subjects_list]
    logging.info(f'Found {len(all_subjects_list):d} subjects in total.')
    return all_subjects_list


def get_human_connectome_project_subjects_phenotypes(config_dict: dict) -> pd.DataFrame:
    """
    Loads all relevant subject phenotypes.
    """
    # Load unrestricted and restricted DataFrames.
    subject_phenotypes_unrestricted_df = _get_subject_phenotypes_unrestricted_df(config_dict)
    subject_phenotypes_restricted_df = _get_subject_phenotypes_restricted_df(config_dict)

    # Combine DataFrames.
    subject_phenotypes_df = subject_phenotypes_unrestricted_df.copy()

    subject_phenotypes_df['Age'] = subject_phenotypes_restricted_df['Age_in_Yrs']
    subject_phenotypes_df['DSM_Depr_Pct'] = subject_phenotypes_restricted_df['DSM_Depr_T']

    print(subject_phenotypes_df.head())

    return subject_phenotypes_df


def _get_subject_phenotypes_unrestricted_df(config_dict: dict) -> pd.DataFrame:
    """
    Load unrestricted (publicly available) data.

    :param config_dict:
    :return:
    """
    subject_phenotypes_unrestricted_df = pd.read_csv(
        os.path.join(
            config_dict['data-dir-subject-measures'], config_dict['phenotypes-unrestricted-filename']
        ),
        index_col='Subject'
    )
    print(subject_phenotypes_unrestricted_df.head())

    # Select relevant columns (subject measures) only.
    relevant_columns = config_dict['phenotypes-unrestricted'] + config_dict['subject-measures-cognitive'] + config_dict['subject-measures-social-emotional'] + config_dict['subject-measures-other'] + config_dict['subject-measures-personality']
    subject_phenotypes_unrestricted_df = subject_phenotypes_unrestricted_df.loc[:, relevant_columns]

    # Turn 'Age' category strings into one of [0, 1, 2, 3].
    # The classes are already ordered from younger to older in ascending order.
    # le = LabelEncoder()
    # subject_phenotypes_unrestricted_df['Age'] = le.fit_transform(subject_phenotypes_unrestricted_df['Age'])
    # print(le.classes_)

    # Turn 'Gender' category strings into 0 and 1.
    le = LabelEncoder()
    subject_phenotypes_unrestricted_df['Gender'] = le.fit_transform(subject_phenotypes_unrestricted_df['Gender'])

    print(subject_phenotypes_unrestricted_df.head())

    return subject_phenotypes_unrestricted_df


def _get_subject_phenotypes_restricted_df(config_dict: dict) -> pd.DataFrame:
    """
    Load restricted (sensitive) data.
    Be careful not to expose this data publicly!

    :param config_dict:
    :return:
    """
    subject_phenotypes_restricted_df = pd.read_csv(
        os.path.join(
            config_dict['data-dir-subject-measures'], config_dict['phenotypes-restricted-filename']
        ),
        index_col='Subject'
    )
    print(subject_phenotypes_restricted_df.head())

    # Normalize Age_in_Yrs column.
    phenotype_array = subject_phenotypes_restricted_df['Age_in_Yrs']
    phenotype_array -= np.mean(phenotype_array)
    phenotype_array /= np.std(phenotype_array)
    subject_phenotypes_restricted_df['Age_in_Yrs'] = phenotype_array

    return subject_phenotypes_restricted_df


def rename_variables_for_plots(
        original_df: pd.DataFrame, axis: int = 1
) -> pd.DataFrame:

    # TODO: replace this maps from .yaml files

    renamed_df = original_df.copy()

    if axis == 0:
        renamed_df.columns = renamed_df.columns.str.replace('CardSort_Unadj', 'Cognitive Flexibility (DCCS)')
        renamed_df.columns = renamed_df.columns.str.replace('DDisc_AUC_40K', 'Delay Discounting')
        renamed_df.columns = renamed_df.columns.str.replace('Flanker_Unadj', 'Inhibition (Flanker Task)')
        renamed_df.columns = renamed_df.columns.str.replace('IWRD_TOT', 'Verbal Episodic Memory')
        renamed_df.columns = renamed_df.columns.str.replace('ListSort_Unadj', 'Working Memory (List Sorting)')
        renamed_df.columns = renamed_df.columns.str.replace('PicSeq_Unadj', 'Visual Episodic Memory')
        renamed_df.columns = renamed_df.columns.str.replace('PicVocab_Unadj', 'Vocab. (Picture Matching)')
        renamed_df.columns = renamed_df.columns.str.replace('PMAT24_A_CR', 'Fluid Intelligence (PMAT)')
        renamed_df.columns = renamed_df.columns.str.replace('ProcSpeed_Unadj', 'Processing Speed')
        renamed_df.columns = renamed_df.columns.str.replace('ReadEng_Unadj', 'Reading (Pronounciation)')
        renamed_df.columns = renamed_df.columns.str.replace('SCPT_SEN', 'Sustained Attention Sens.')
        renamed_df.columns = renamed_df.columns.str.replace('SCPT_SPEC', 'Sustained Attention Spec.')
        renamed_df.columns = renamed_df.columns.str.replace('VSPLOT_TC', 'Spatial Orientation')
    elif axis == 1:
        renamed_df.index = renamed_df.index.str.replace('CardSort_Unadj', 'Cognitive Flexibility (DCCS)')
        renamed_df.index = renamed_df.index.str.replace('DDisc_AUC_40K', 'Delay Discounting')
        renamed_df.index = renamed_df.index.str.replace('Flanker_Unadj', 'Inhibition (Flanker Task)')
        renamed_df.index = renamed_df.index.str.replace('IWRD_TOT', 'Verbal Episodic Memory')
        renamed_df.index = renamed_df.index.str.replace('ListSort_Unadj', 'Working Memory (List Sorting)')
        renamed_df.index = renamed_df.index.str.replace('PicSeq_Unadj', 'Visual Episodic Memory')
        renamed_df.index = renamed_df.index.str.replace('PicVocab_Unadj', 'Vocab. (Picture Matching)')
        renamed_df.index = renamed_df.index.str.replace('PMAT24_A_CR', 'Fluid Intelligence (PMAT)')
        renamed_df.index = renamed_df.index.str.replace('ProcSpeed_Unadj', 'Processing Speed')
        renamed_df.index = renamed_df.index.str.replace('ReadEng_Unadj', 'Reading (Pronounciation)')
        renamed_df.index = renamed_df.index.str.replace('SCPT_SEN', 'Sustained Attention Sens.')
        renamed_df.index = renamed_df.index.str.replace('SCPT_SPEC', 'Sustained Attention Spec.')
        renamed_df.index = renamed_df.index.str.replace('VSPLOT_TC', 'Spatial Orientation')
    return renamed_df


def scan_id_to_scan_name(scan_id: int) -> str:
    """
    Map scan index to scan ID as used in manuscript.
    """
    scan_id_map = {
        0: '1A',
        1: '1B',
        2: '2A',
        3: '2B'
    }
    return scan_id_map[scan_id]
