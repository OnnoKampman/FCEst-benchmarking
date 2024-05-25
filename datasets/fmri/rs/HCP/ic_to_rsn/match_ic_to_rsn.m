% Mapping of ICs to RSNs for neuro-dynamic-covariance project. First thrsehold the raw probability maps from the IC and RSN. Then calculate
% dices coefficient of the overlap between each IC and all RSNs, as well as the percentage of the IC mask that overlaps with each RSN (these are almost perfectly
% correlated but percent is a bit more interpretable).
%
% ICS downloaded from HPC
% RSNs downloaded from https://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/
% See Smith et al., 2011. PNAS. Correspondence of the brain's functional architecture during activation and rest.
%
% Anatomical template from D:\spm12\toolbox\cat12\templates_1.50mm
% Dices Coefficient calculation from https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
%
% USEAGE:
% set:
%     base_dir - add the base_directory containing niftis to process
%     save_results_filename_prefix - prefix of the filename to save the results in
%     binary_IC_filename - filename to save the binarized IC map under
%     ic_prob_maps - set the filename for the 4D (x, y, z, IC) NII containig probability maps for each IC
%     
%
% TODO:
% Double check thr Z = 3, what units are the RSN TPM in?
% Find a ref for the IC cutoff
% Double check the overlap for the template and RSN / ICS
%
% --------------------------------------------------------------------------------------------------------------------------------------------------------------

base_dir = 'D:\neuro-covariance-maps';
save_results_filename_prefix = 'ic_to_rsn_mapping';
binary_IC_filename = 'binary_melodic_IC_sum.nii';
ic_prob_maps = load_nii(fullfile(base_dir, 'melodic_IC_sum.nii'));

rsn_prob_maps = load_nii(fullfile(base_dir, 'PNAS_Smith09_rsn10.nii'));
ic_thr = 30;  % based on previously seen in the field, any refs?
rsn_thr = 3;  % taken from smith et al., 2009

num_rsn = size(rsn_prob_maps.img, 4);
num_ic = size(ic_prob_maps.img, 4);

rsn_labels = {'medial_visual', 'occipital_pole', 'lateral_visual', 'default_mode', 'cerebellum', 'sensorimotor', ...  from Smith et al., 2011
               'auditory', 'executive_control', 'frontoparietal_right', 'frontoparietal_left'};
ic_labels = {};
for i = 1:num_ic; ic_labels{end + 1} = ['IC_', num2str(i)]; end

%% Threshold Tissue Probability Maps and Save

ic_bin_masks = ic_prob_maps;
rsn_bin_masks = rsn_prob_maps;

for i = 1:num_ic
    ic_bin_masks.img(:, :, :, i) = ic_prob_maps.img(:, :, :, i) > ic_thr;
end
save_nii(ic_bin_masks, fullfile(base_dir, binary_IC_filename));

for i = 1:num_rsn
    rsn_bin_masks.img(:, :, :, i) = rsn_prob_maps.img(:, :, :, i) > rsn_thr;
end
save_nii(rsn_bin_masks, fullfile(base_dir, 'binary_PNAS_Smith09_rsn10.nii'));

%% Calculate Dice Coefficient and percent overlap between every IC and RSN

ic_by_rsn_dice_coefficients = NaN(num_ic, num_rsn);
ic_by_rsn_percent = NaN(num_ic, num_rsn);

for i_ic = 1:num_ic
    
    ic_mask = ic_bin_masks.img(:, :, :, i_ic);
    cardinal_ic = nnz(ic_mask);
    
    for i_rsn = 1:num_rsn

        rsn_mask = rsn_bin_masks.img(:, :, :, i_rsn);
        cardinal_rsn = nnz(rsn_mask); 
        
        cardinal_intersection = nnz(ic_mask & rsn_mask);
        
        dices_coef = (2 * cardinal_intersection) / (cardinal_ic + cardinal_rsn);
        ic_by_rsn_dice_coefficients(i_ic, i_rsn) = dices_coef;
        
        percent_overlap = (cardinal_intersection / cardinal_ic) * 100;
        ic_by_rsn_percent(i_ic, i_rsn) = percent_overlap;
            
    end 
end

ic_by_rsn_dice_coefficients_table = array2table(ic_by_rsn_dice_coefficients, 'RowNames', ic_labels, 'VariableNames', rsn_labels);
ic_by_rsn_percent_table = array2table(ic_by_rsn_percent, 'RowNames', ic_labels, 'VariableNames',  rsn_labels);

writetable(ic_by_rsn_dice_coefficients_table,  ...
           fullfile(base_dir, [save_results_filename_prefix, '_dice_coefficients.csv']));
writetable(ic_by_rsn_percent_table,  ...
           fullfile(base_dir, [save_results_filename_prefix, '_percent.csv'])); 
