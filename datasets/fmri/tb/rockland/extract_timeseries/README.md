# Analysis

## Create masks

Before running these scripts, we need to create brain voxel masks using `make_gm_intersection_union_mask_from_raw_probability_map.m`.

## Extract regions of interest

Brain regions of interests (ROIs) are extracted with the `extract_checkerboard_task_rois.m` script.
This script is written in Matlab because it uses SPM's HRF function.

Afterwards, we save time series in a convenient format using the `extract_node_timeseries.py` script.
