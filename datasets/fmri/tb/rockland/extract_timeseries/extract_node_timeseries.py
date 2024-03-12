import logging
import os
import socket

import pandas as pd

from configs.configs import get_config_dict
from helpers.data import normalize_array
from helpers.rockland import extract_time_series_rockland


def _extract_time_series_final_regions(config_dict: dict, roi_list_name: str) -> None:
    """
    Extract time series and save them to a convenient format.

    Parameters
    ----------
    :param config_dict:
    :param roi_list_name:
        'final',
        'V1_V2_V3_V4_ACC'
    """
    full_df = pd.read_csv(
        os.path.join(
            config_dict['data-basedir'], pp_pipeline, 'results', roi_list_name, 'subject_region_timeseries.csv'
        )
    )
    print(full_df.head())
    full_df = _rename_column_names(full_df)
    full_df = _rename_regions_of_interest(full_df)
    print(full_df.head())

    # Combine all time series observations into single array.
    # ts_columns = [col for col in full_df.columns if 'results3_' in col]
    # full_df["ts"] = full_df[ts_columns].apply(lambda x: list(x), axis=1)
    # full_df.drop(ts_columns, axis=1, inplace=True)
    # print(full_df)

    time_series_savedir = os.path.join(
        config_dict['data-basedir'], pp_pipeline, 'node_timeseries', roi_list_name
    )
    if not os.path.exists(time_series_savedir):
        os.makedirs(time_series_savedir)

    _save_time_series_per_subject(full_df, time_series_savedir)


def _extract_time_series_hand_drawn_regions():
    regions_of_interest = [
        'v1',
        'ips',
        # 'pfc',
        # 'uncor_reg'
    ]
    subset = '_'.join(regions_of_interest)

    full_df = pd.read_csv(
        os.path.join('datasets', 'rockland', 'V1_seed_to_hand_drawn_region_results_test.csv')
    )
    print(full_df)

    ts_columns = [col for col in full_df.columns if 'results3_' in col]
    full_df["ts"] = full_df[ts_columns].apply(lambda x: list(x), axis=1)
    full_df.drop(ts_columns, axis=1, inplace=True)
    print(full_df)

    subjects = full_df['subject'].unique()  # list of strings
    for subject in subjects:
        subject_df = full_df[full_df['subject'] == subject]
        print(subject_df)
        bivariate_df = extract_time_series_rockland(subject_df, regions_of_interest)
        print(bivariate_df)
        bivariate_df.to_csv(
            os.path.join('datasets', 'rockland', subset, f'{subject:s}.csv'),
            index=False, header=True
        )
        logging.info(f"Saved extracted data from subject '{subject:s}'.")


def _save_time_series_per_subject(all_subjects_df: pd.DataFrame, time_series_savedir: str) -> None:
    subjects = all_subjects_df['subject'].unique()  # list of strings
    for i_subject, subject in enumerate(subjects):
        print(f'\nSubject {i_subject+1:d} / {len(subjects):d}: {subject:s}\n')

        # Select subject rows.
        subject_df = all_subjects_df[all_subjects_df['subject'] == subject]

        # Drop subject name column.
        subject_df = subject_df.drop('subject', axis=1)

        # Set region names as index.
        subject_df = subject_df.set_index('ROI')
        print(subject_df.head())

        # Transpose with ROIs as columns.
        subject_df = subject_df.T.reset_index(drop=True)  # (N, D)
        print(subject_df.head())

        # Normalize time series.
        for i_ts in range(subject_df.shape[1]):
            subject_df.iloc[:, i_ts] = normalize_array(subject_df.iloc[:, i_ts], verbose=False)
        print(subject_df.head())

        subject_df.to_csv(
            os.path.join(time_series_savedir, f'{subject:s}.csv'),
            index=False,  # TODO: does this still work?
            header=True
        )
        logging.info(f"Saved extracted time series from subject '{subject:s}' in '{time_series_savedir:s}'.")


def _rename_column_names(original_df: pd.DataFrame) -> pd.DataFrame:
    renamed_df = original_df.copy()
    renamed_df.columns = renamed_df.columns.str.replace('results1', 'subject')
    renamed_df.columns = renamed_df.columns.str.replace('results2', 'ROI')
    return renamed_df


def _rename_regions_of_interest(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make regions of interest (ROIs) more concise and readable.
    """
    renamed_df = original_df.copy()

    renamed_df['ROI'] = renamed_df['ROI'].str.replace('v1_ts', 'V1')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('reg1_ts', 'V2')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('reg2_ts', 'V3')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('reg3_ts', 'V4')

    renamed_df['ROI'] = renamed_df['ROI'].str.replace('V1_sphere_mask', 'V1')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('mPFC_sphere_mask', 'mPFC')

    renamed_df['ROI'] = renamed_df['ROI'].str.replace('GM_V1_union', 'V1')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('GM_V2_union', 'V2')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('GM_V3_union', 'V3')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('GM_V4_union', 'V4')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('GM_ACC_R', 'ACC/mPFC')

    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_v1', 'V1')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_v2', 'V2')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_v3', 'V3')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_v4', 'V4')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_mpfc', 'mPFC')
    renamed_df['ROI'] = renamed_df['ROI'].str.replace('gm_union_m1', 'M1')

    return renamed_df


if __name__ == "__main__":

    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )
    _extract_time_series_final_regions(
        config_dict=cfg,
        roi_list_name='final'
    )
