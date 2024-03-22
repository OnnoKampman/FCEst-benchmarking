% This script takes the output files of `extract_checkerboard_task_rois.m`
% and plots the result into a `.ps` and `.pdf` file.
%
% TODO: finish this script
% 
% -------------------------------------------------------------------------------------------------------------------------

fmri = fMRIMethods;

roi_list_name = 'V1_mPFC';
roi_list = [
    "V1_sphere_mask.nii"; 
    "mPFC_sphere_mask.nii";
];
num_regions = length(roi_list);

% Configs.
results_dir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'results', roi_list_name);
mkdir(results_dir);

% TODO: this was task_aggr before, are we sure we want to use the nonaggr?
data_dir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'mpp_mni', 'task_nonaggr');

subjects = dir(fullfile(data_dir, 'sub*'))
TR = 1.4;  % in seconds
HRF = spm_hrf(TR);

% TODO: this is only for TR=1400, need to compute automatically
stim_on_idxes = [15:30, 44:58, 71:86];          % round(timepoint / TR) + 1. The +1 depends on if you call TR=1 starting at time 0 or 1.4
num_vols = 98;                                  % number of volumes in file TODO: can we not read this from data itself?

% Generate stimulus time series.
stim = zeros(1, num_vols);
stim(stim_on_idxes) = 1;
conv_stim = conv(stim, HRF');
aligned_conv_stim = conv_stim(1:num_vols);

time = linspace(0, (num_vols .* TR) - TR, num_vols);

% Load all data and variables.
load(fullfile(results_dir, 'roi_timeseries.mat'));

for i_subject = 1:length(subjects)

    subject_name = subjects(i_subject).name
    nii_path = dir(fullfile(data_dir, subject_name, 'MNI*.nii'));

    % There should only be one Nifti file here that starts with 'MNI...'.
    if length(nii_path) ~= 1
        continue
    end

    figure('Position', [500, 500, 800, 1200]); hold on;
    
    % load nii and mask timeseries
    nii = load_nii(fullfile(nii_path.folder, nii_path.name));

    for i_region = 1:num_regions

        reg = roi_list{i_region};
        region_name = extractBefore(reg, '.nii');

        mask = load_nii(fullfile(config.analysis_masks_path, reg));

        [~, data] = fmri.mask_nii_timeseries(nii, mask);

        subplot(num_regions + 2, 1, i_region);
        to_plot = data;
        plot(time, (to_plot(1:num_vols)), 'r',  'LineWidth', 2);
        title(sprintf('Region %s', region_name), 'Interpreter', 'None');
        
        hold on;
        plot(time, rescale(aligned_conv_stim, min(data), max(data)), 'k', 'LineWidth', 2);

    end

    sgtitle(subject_name, 'Interpreter', 'None');
    
    % TODO: it now says the figure is too big? not sure how to fix that
    print(gcf,'-dpsc2','-append', fullfile(results_dir, 'all_subjects_V1_seed.ps'));

end
