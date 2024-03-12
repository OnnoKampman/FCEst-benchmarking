import gzip
import logging
import os
import shutil
import socket

from configs.configs import get_config_dict


if __name__ == "__main__":

    # Copy and MNI denoised functionals

    # Script to take the output from FSL preprocessing pipeline and save
    # MNI-space versions in a dedicated directory for further analysis.
    # Re-samples voxels etc. to fit MNI mould, standard is 2mm voxels.

    roi_list_name = 'final'
    pp_pipeline = 'custom_fsl_pipeline'

    cfg = get_config_dict(
        data_set_name='rockland',
        subset='645',
        hostname=socket.gethostname()
    )

    os.system('which applywarp')

    agressive_setting = 'nonaggr'
    path_ = f"task_{agressive_setting:s}"
    data_base_path = '/mnt/Data/fcest/datasets/task/rockland'
    mpp_data_basedir = os.path.join(data_base_path, 'CHECKERBOARD645', pp_pipeline, 'mpp')

    mni_data_basedir = os.path.join(data_base_path, 'CHECKERBOARD645', pp_pipeline, 'mpp_mni')
    if not os.path.exists(mni_data_basedir):
        os.makedirs(mni_data_basedir)
    mni_results_dir = os.path.join(mni_data_basedir, path_)  # directory where MNI-space files will be saved
    if not os.path.exists(mni_results_dir):
        os.makedirs(mni_results_dir)

    task_name = '_ses-BAS1_task-CHECKERBOARD_acq-645_bold_mpp'
    ref_filename = 'standard.nii.gz'
    warp_filename = 'example_func2standard_warp.nii.gz'
    nii_filename = f"denoised_func_data_{agressive_setting:s}.nii.gz"

    subjects_to_exclude = [
        'sub-A00008399',
        'sub-A00010893',
        'sub-A00031145', 'sub-A00031166', 'sub-A00031604',
        'sub-A00037848',
        'sub-A00043450', 'sub-A00044171',
        'sub-A00051676',
        'sub-A00079702'
    ]

    # Copy relevant files and move EPIs to MNI space.
    all_subject_filenames_list = sorted([s for s in os.listdir(os.path.join(data_base_path, 'CHECKERBOARD645', 'raw')) if 'sub-' in s])
    for i_run, subject_name in enumerate(all_subject_filenames_list):
        if subject_name in subjects_to_exclude:
            logging.warning(f'Skipping subject {i_run:d}: {subject_name:s}')
            continue
        logging.info(f"Converting {subject_name:s}...")

        # Copy relevant preprocessed fields to MNI preprocessed data directory.
        mpp_data_subject_basedir = os.path.join(
            mpp_data_basedir, subject_name, 'ses-BAS1', f"{subject_name:s}{task_name:s}"
        )
        os.mkdir(os.path.join(mni_results_dir, subject_name))
        shutil.copyfile(
            src=os.path.join(mpp_data_subject_basedir, 'reg', ref_filename),
            dst=os.path.join(mni_results_dir, subject_name, ref_filename)
        )
        shutil.copyfile(
            src=os.path.join(mpp_data_subject_basedir, 'reg', warp_filename),
            dst=os.path.join(mni_results_dir, subject_name, warp_filename)
        )
        shutil.copyfile(
            src=os.path.join(mpp_data_subject_basedir, 'ica-aroma_fwhm5', nii_filename),
            dst=os.path.join(mni_results_dir, subject_name, nii_filename)
        )
        logging.info("Relevant files copied.")

        # Use FSL applywarp to go from native MPP space to MNI space.
        in_filepath = os.path.join(mni_results_dir, subject_name, nii_filename)
        out_filepath = os.path.join(mni_results_dir, subject_name, f'MNI_{nii_filename:s}')
        ref_filepath = os.path.join(mni_results_dir, subject_name, ref_filename)
        warp_filepath = os.path.join(mni_results_dir, subject_name, warp_filename)
        os.system(
            f"applywarp --in={in_filepath:s} --out={out_filepath:s} --ref={ref_filepath:s} --warp={warp_filepath:s} --interp=spline"
        )

        # Convert .nii.gz file into .nii file.
        with gzip.open(out_filepath, 'rb') as f_in:
            with open(out_filepath.removesuffix('.gz'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(out_filepath)

    print('> Finished converting MPP images to MNI space.')
