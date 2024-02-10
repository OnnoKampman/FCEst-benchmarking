%% Check alignment of MNI space images
%
% Sometimes the alignment to MNI space does not work well.
% This script checks whether it did.
%
% Also the option to check alignment across all MNI scans and output a .mat 
% file containing the alignments to the output directory. Note alignment is 
% calculated simply as the voxelwise correlation.
%  
% Note that check alignment requires fMRIMethods from JZiminski's glass-patterns-three-session-mri project.
% TODO: move this to a shared tools repo?
%
% -------------------------------------------------------------------------

run set_rockland_configs
fmri = fMRIMethods;

path_ = ['task_', config.agressive_setting];
mni_results_dir = fullfile(config.mni_data_basedir, path_);
task_name = ['_ses-BAS1_task-CHECKERBOARD_acq-', config.TR, '_bold_mpp'];

% Handle multi-core
%if ~isempty(gcp('nocreate'))
%    delete(gcp('nocreate'))
%end
%parpool(config.num_cores);

niis_to_align = dir(fullfile(mni_results_dir, '**/MNI*.nii'));
num_niis = length(niis_to_align);

% TODO: where is this mask?
mask = load_nii(fullfile(config.mni_data_basedir, 'standard_mask.nii'));

coreg_check_results = {};

for i = 2:num_niis

    fprintf('running nii %s / %d\n', num2str(i), num_niis);
    nii = load_nii(fullfile(niis_to_align(i).folder, niis_to_align(i).name));
    nii_m1 = load_nii(fullfile(niis_to_align(i-1).folder, niis_to_align(i-1).name));

    coreg_check_results{i, 1} = fmri.check_align(nii.img, nii_m1.img, mask.img);
    coreg_check_results{i, 2} = strcat(niis_to_align(i).folder(end-12:end), '__', niis_to_align(i-1).folder(end-12:end));

end

% TODO: we should save this in the preprocessing folder
save(fullfile(mni_results_dir, 'coreg_check.mat'), 'coreg_check_results');
