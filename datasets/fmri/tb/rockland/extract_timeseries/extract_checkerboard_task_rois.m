% This script saves time series per subject per region of interest (ROI) 
% into a file called `subject_region_timeseries.csv`.
%
% The stimulation protocol can be found in the `.tsv` files, in the `func` 
% folder of the raw data.
% 
% The stimulation parameters are:
%
% TODO: check this because the length is either shorter or longer for 2 TRs
% onset	duration	trial_type
% 0.0	20.0	FIXATION
% 20.0	20.0	CHECKER
% 40.0	20.0	FIXATION
% 60.0	20.0	CHECKER
% 80.0	20.0	FIXATION
% 100.0	20.0	CHECKER
% 120.0	20.0	FIXATION
%
% NOTES: How to handle the convolution length is not necessarily clear. It is correct to simple take the first n
%        samples of the convolution (see https://groups.google.com/g/mvpa-toolbox/c/2mjCKg9OoOQ and spm_Volterra)
%
% TODO: check with a script that the stimulation parameters are the same for all subjects (have only checked a few by eye).
%       It would be good to have a second pair of eyes check the time alignment.
%       I am 99% sure it is correct but was getting confused by a fencepost error
%       why is the data not zscored or normalised? This should happen during the ICA? Maybe not as it is soft denoise.
%       The data does not seem to be normalised in the FSL pipeline 
% -------------------------------------------------------------------------------------------------------------------------

fmri = fMRIMethods;
run set_rockland_configs

% Define which regions of interest (ROI) to extract.
% TODO: move this to config
roi_list_name = 'final';
roi_list = [
    "gm_union_v1.nii";
    "gm_union_v2.nii";
    "gm_union_v3.nii";
    "gm_union_v4.nii";
    "gm_union_mpfc.nii";
    "gm_union_m1.nii";
];
% roi_list_name = 'V1_V2_V3_V4_ACC';
% roi_list = [
%     "GM_V1_union.nii";
%     "GM_V2_union.nii";
%     "GM_V3_union.nii";
%     "GM_V4_union.nii";
%     "GM_ACC_R.nii";
% ];
num_regions = length(roi_list);

% Configs.
results_dir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'custom_fsl_pipeline', 'results', roi_list_name)
mkdir(results_dir);
data_dir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'custom_fsl_pipeline', 'mpp_mni', ['task_' config.agressive_setting])

subjects = dir(fullfile(data_dir, 'sub*'))
TR_seconds = str2double(config.TR) / 1000;
HRF = spm_hrf(TR_seconds);

% Generate stimulus time series.
stim = zeros(1, config.num_vols);
stim(config.stim_on_idxes) = 1;
conv_stim = conv(stim, HRF');
aligned_conv_stim = conv_stim(1:config.num_vols);

time = linspace(0, (config.num_vols .* TR_seconds) - TR_seconds, config.num_vols);

results = [];
rowcnt = 1;

for i_subject = 1:length(subjects)

    subject_name = subjects(i_subject).name
    nii_path = dir(fullfile(data_dir, subject_name, 'MNI*.nii'));

    % There should only be one Nifti file here that starts with 'MNI...'.
    if length(nii_path) ~= 1
        continue
    end
    
    % load nii and mask timeseries
    nii = load_nii(fullfile(nii_path.folder, nii_path.name));

    for i_region = 1:num_regions

        roi_mask_filename = roi_list{i_region};
        mask = load_nii(fullfile(config.analysis_masks_path, roi_mask_filename));

        [~, data] = fmri.mask_nii_timeseries(nii, mask);

        results{rowcnt, 1} = subject_name;
        results{rowcnt, 2} = extractBefore(roi_mask_filename, '.nii');
        results{rowcnt, 3} = data;
        rowcnt = rowcnt + 1;

    end

end

results_table = cell2table(results)

save(fullfile(results_dir, 'roi_timeseries.mat'));
writetable(results_table, fullfile(results_dir, 'subject_region_timeseries.csv'));

% Save stimuli TODO: is this not the same for all subjects?
writematrix(stim, fullfile(results_dir, 'stim.csv'));
writematrix(aligned_conv_stim, fullfile(results_dir, 'convolved_stim.csv'));
writematrix(time, fullfile(results_dir, 'time.csv'));

disp(['> Results saved to ' results_dir]);
