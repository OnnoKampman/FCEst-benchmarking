base_path = 'H:\neuro-dynamic-covariance\datasets\task\rockland\analysis\masks\final';
raw_prob_map_path = fullfile(base_path, 'raw_probability_maps');
resized_prob_map_path = fullfile(base_path, 'resized_probability_maps');
segmentation_path = fullfile(base_path, 'MNI_segmentation');
binarized_map_path = fullfile(base_path, 'binarized_masks');
union_mask_path = fullfile(base_path, 'union_masks');
gm_intersection_union_masks_path = fullfile(base_path, 'gm_union_masks');

fmri = fMRIMethods;
thr = 40;  % both brainnetome and wang atlas use range 0 - 100 for probabilities

all_region_names = ["v1", "v2", "v3", "v4", "mpfc", "m1"];
[standard_boundary_box, standard_voxel_size] = spm_get_bbox(fullfile(segmentation_path, 'standard.nii'));


% Unzip all raw probability maps (or will cause problems later)
zipped_files = dir(fullfile(raw_prob_map_path, '**', '*.nii.gz'));
for i = 1:length(zipped_files)
    
    gunzip(fullfile(zipped_files(i).folder, zipped_files(i).name));
    delete(fullfile(zipped_files(i).folder, zipped_files(i).name));
    
end


% Resize All probability Maps 
for region_name = all_region_names
    
   mask_paths = dir(fullfile(raw_prob_map_path, region_name, '*.ni*'));  % zipped and unziupped depending on atlas
   
   mkdir(fullfile(resized_prob_map_path, region_name));  % TODO: DRY on path
          
   for i = 1:length(mask_paths)      
           
       moved_file_path = fullfile(resized_prob_map_path, region_name, mask_paths(i).name);
       
       copyfile(fullfile(mask_paths(i).folder, mask_paths(i).name),  ...
                moved_file_path);

       resize_img(char(moved_file_path), abs(standard_voxel_size), standard_boundary_box, false);
       
       delete(moved_file_path);
       
   end
   
end
            
    
% Binarize All Probability Maps
for region_name = all_region_names
    
    region_name = char(region_name);
    
    mask_paths = dir(fullfile(resized_prob_map_path, region_name, '*.ni*'));
    
    mkdir(fullfile(binarized_map_path, region_name));
    
    for i = 1:length(mask_paths)
        
       moved_file_path = fullfile(binarized_map_path, region_name, mask_paths(i).name);
       
       copyfile(fullfile(mask_paths(i).folder, mask_paths(i).name),  ...
                moved_file_path);
                    
       resized_prob_map_nii = load_nii(char(moved_file_path));
       binarized_map = fmri.threshold_probability_map(resized_prob_map_nii, thr);
       
       save_nii(binarized_map, fullfile(binarized_map_path, region_name, ['binarized_', mask_paths(i).name]));
       
       delete(moved_file_path);

    end
end


% Make Union for All Binarized Masks
for region_name = all_region_names
    
    region_name = char(region_name);
    
    mask_paths = dir(fullfile(binarized_map_path, region_name, '*.ni*'));
    
    mask_reshaped_paths = fullfile({mask_paths.folder}, {mask_paths.name})';
    union_mask = fmri.make_union_mask(mask_reshaped_paths);

    mkdir(fullfile(union_mask_path, region_name));
    save_nii(union_mask,  ...
             fullfile(union_mask_path, region_name, [region_name, '_union_mask.nii']));

end


% Intersection Union with Grey Matter for ALl Binarized Masks
gm_nii = load_nii(fullfile(segmentation_path, 'binary_gm_mask.nii'));
for region_name = all_region_names
    
    region_name = char(region_name);
    
    region_union_mask_path = fullfile(union_mask_path, region_name, [region_name, '_union_mask.nii']);

    intersection_mask = fmri.make_intersection_mask(fullfile(segmentation_path, 'binary_gm_mask.nii'),  ...
                                                    region_union_mask_path, [], []);

    save_nii(intersection_mask, fullfile(gm_intersection_union_masks_path, ['gm_union_', region_name, '.nii']));
    
end
