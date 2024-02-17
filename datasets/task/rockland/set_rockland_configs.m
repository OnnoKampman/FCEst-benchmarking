% Rockland dataset 
% http://fcon_1000.projects.nitrc.org/indi/enhanced/
%
% Data contains TR 645 and TR 1400 acquisitions using a ON/OFF checkerboard protocol. This is a subset (50) 
% of participants, the full dataset has hundreds of participants and lots of different scan modalities / experimental tasks.

%% Set Paths

config.TR = '645';  % '1400' or '645'
% TODO: can we read this from data itself?
config.num_vols = 240;  % number of volumes in file (98 for TR=1400, 240 for TR=645)
% TODO: need to compute the below automatically
% round(timepoint / TR) + 1. The +1 depends on if you call TR=1 starting at time 0 or 1.4
% config.stim_on_idxes = [15:30, 44:58, 71:86];  % for TR=1400
config.stim_on_idxes = [32:63, 94:125, 156:187];  % for TR=645

% this refers to hard or soft denoising
% TODO: this was task_aggr before, are we sure we want to use the nonaggr?
config.agressive_setting = 'nonaggr';  % 'aggr' or 'nonaggr'

% base paths
if ispc
   config.git_base = fullfile('D:', 'fMRIData');
   config.git_base_glass_patterns = fullfile(config.git_base);
   config.toolbox_base = fullfile('D:', 'Toolbox');
   config.data_base_path = fullfile('H:', 'neuro-dynamic-covariance', 'datasets', 'task', 'rockland');
   config.spm_path = fullfile('D:', 'spm12.4');
   config.num_cores = 1;
elseif isunix
    config.git_base = fullfile('/home', getenv('USERNAME'));
    config.git_base_glass_patterns = fullfile('/home', 'jjz33');
    config.data_base_path = fullfile('/mnt', 'Data', 'neuro-dynamic-covariance', 'datasets', 'task', 'rockland');
    config.spm_path = fullfile('/mnt', 'Data', 'Toolboxes', 'spm12.4');
    config.num_cores = 12;
end

% git paths
config.git_base_path =  fullfile(config.git_base_glass_patterns, 'git_repos', 'glass-patterns-three-session-mri');
addpath(genpath(fullfile(config.git_base, 'git_repos', 'neuro-dynamic-covariance')));
addpath(genpath(config.git_base_glass_patterns));

% preprocessing paths
config.raw_data_basedir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'raw');
config.mpp_data_basedir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'mpp');
config.mni_data_basedir = fullfile(config.data_base_path, ['CHECKERBOARD' num2str(config.TR)], 'mpp_mni');

% analysis path
config.seed_to_target_path = fullfile(config.data_base_path, 'analysis', 'seed_to_target');
% config.analysis_masks_path = fullfile(config.data_base_path, 'analysis', 'masks');
config.analysis_masks_path = fullfile(config.data_base_path, 'analysis', 'masks', 'final', 'gm_union_masks');

% file names
config.ref_filename = 'standard.nii.gz';
config.nii_filename = ['denoised_func_data_', config.agressive_setting, '.nii.gz'];
config.warp_filename = 'example_func2standard_warp.nii.gz';

%% Set analysis configs

% session
config.session = 'ses-BAS1';

% subjects
config.subjects = dir(fullfile(config.raw_data_basedir, 'sub*'));

% exclude certain subjects, due to missing data or bad MNI transformation
% TODO: how do we determine which subjects to exclude?
% 'sub-A00008399'; excluded for bad MNI transformation
% TODO: this should probably be replaced by some "try except" statement, or
% these can be automatically found and skipped if files are missing
if strcmp(config.TR, '1400')
    config.subjects_to_exclude = {
        'sub-A00008326'; 'sub-A00008399'; 
        'sub-A00010893';
        'sub-A00028287'; 'sub-A00028352';
        'sub-A00031145'; 'sub-A00031166'; 'sub-A00031604';
        'sub-A00037848'; 
        'sub-A00062248';
        'sub-A00075896'; 'sub-A00079702';
    };
elseif strcmp(config.TR, '645')
    config.subjects_to_exclude = {
        'sub-A00008399';
        'sub-A00010893';
        'sub-A00031145'; 'sub-A00031166'; 'sub-A00031604';
        'sub-A00037848';
        'sub-A00043450'; 'sub-A00044171';
        'sub-A00051676';
        'sub-A00079702';
    };
else
    disp('ERROR: invalid TR.');
end