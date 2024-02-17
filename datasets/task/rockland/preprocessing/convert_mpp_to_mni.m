%% Copy and MNI denoised functionals
%
% Script to take the output from FSL preprocessing pipeline and save
% MNI-space versions in a dedicated directory for further analysis.
% Re-samples voxels etc. to fit MNI mould, standard is 2mm voxels.
%
% -------------------------------------------------------------------------

run set_rockland_configs
system('which applywarp')

path_ = ['task_', config.agressive_setting];
mni_results_dir = fullfile(config.mni_data_basedir, path_);
task_name = ['_ses-BAS1_task-CHECKERBOARD_acq-', config.TR, '_bold_mpp'];

% Handle multi-core
if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate'))
end
parpool(config.num_cores);

% Copy relevant files and move EPIs to MNI space.
parfor irun = 1:length(config.subjects)
    
    subject_name = config.subjects(irun).name;
    subject_name
    if ismember(subject_name, config.subjects_to_exclude)
        fprintf(['Skipping subject ', num2str(irun), ': ', subject_name, '\n'])
        continue
    end
    
    mpp_data_subject_basedir = fullfile(config.mpp_data_basedir, ... 
                                        subject_name, config.session,  ...
                                        strcat(subject_name, task_name));
    
    % Copy relevant preprocessed fields to MNI preprocessed data directory.
    mkdir(fullfile(mni_results_dir, subject_name));
    copyfile(fullfile(mpp_data_subject_basedir, 'reg', config.ref_filename),  ...
             fullfile(mni_results_dir, subject_name, config.ref_filename));
    copyfile(fullfile(mpp_data_subject_basedir, 'reg', config.warp_filename),  ...
             fullfile(mni_results_dir, subject_name, config.warp_filename));
    copyfile(fullfile(mpp_data_subject_basedir, 'ica-aroma_fwhm5', config.nii_filename),  ...
             fullfile(mni_results_dir, subject_name, config.nii_filename));
    
    % Use FSL applywarp to go from native MPP space to MNI space.
    system(strcat('applywarp',  ...
                  [' --in=', fullfile(mni_results_dir, subject_name, config.nii_filename)],  ...
                  [' --out=', fullfile(mni_results_dir, subject_name, strcat('MNI_', config.nii_filename))],  ...
                  [' --ref=', fullfile(mni_results_dir, subject_name, config.ref_filename)],  ...
                  [' --warp=', fullfile(mni_results_dir, subject_name, config.warp_filename)],  ...
                  ' --interp=spline'));
    
    % Convert .nii.gz file into .nii file.
    gunzip(fullfile(mni_results_dir, subject_name, strcat('MNI_', config.nii_filename))); 
    delete(fullfile(mni_results_dir, subject_name, strcat('MNI_', config.nii_filename)));            
end

disp('> Finished converting MPP images to MNI space.')
