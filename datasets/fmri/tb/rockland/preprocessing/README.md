# Preprocessing data

After downloading the raw fMRI data, we need to do some preprocessing.

## Prerequisites

The scripts assume that everything is BIDS data format compliant.
They require Python 3.8+ and `fsl` to run.

More info on installing `fsl`: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

## Overview

We run a minimal preprocessing (MPP) pipeline on the raw data.
The script `nki_bids_mpp.sh` runs the preprocessing from raw data to preprocessed and de-noised fMRI images (see the respective script header for all preprocessing steps).
The script `reg_fun2mni.sh` is used by `nki_bids_mpp.sh` and outputs the required warp files to take the native space functional image to MNI.
However, this script does not actually convert the functional images to MNI space.
This is done by the Matlab script `convert_mpp_to_mni.m`.
After this step, time series can be extracted from these preprocessed images.

The `submit_checkerboard_bids_mpp_slurm_jobs.sh` script uses SLURM to run preprocessing on all subjects in parallel.
This script is essentially a wrapper around `nki_bids_mpp.sh`.

## Running preprocessing

To run the preprocessing in parallel for all subjects, edit the bash script `submit_checkerboard_bids_mpp_slurm_jobs.sh` with the relevant paths and settings for the preprocessing you want to do.
Run the bash script with `sbatch` to run it with SLURM (this has to be done in a virtual environment with Python 3.8 installed).

```
sbatch datasets/task/rockland/preprocessing/submit_checkerboard_bids_mpp_slurm_jobs.sh
```

The script `nki_bids_mpp.sh` can be run separately for an individual scan as well.

```
bash datasets/task/rockland/preprocessing/custom_fsl_pipeline/nki_bids_mpp.sh sub-A00023510 BAS1 1400 CHECKERBOARD
```

Once these preprocessing steps have finished, edit the `set_rockland_configs.m` file with the relevant paths and parameters.
Then, edit and run the `convert_mpp_to_mni.m` script with options to run as well.
This will copy out the final preprocessed functional images to a dedicated output folder for later analysis and puts them in MNI space.
This script should be run on the server as well, allowing for parallelization again.

Alternatively, you can run the Python-based script `convert_mpp_to_mni.py`.

After this, there is also the option to check the alignment (see script header).
These steps need to be in Matlab because they require SPM.

## Issues

* Some raw image files seem to be missing.
* Also, some raw data file names have `RR` in their filename after the respective TR.

## References
[1] [ABL Gitlab pipeline repository](https://gitlab.developers.cam.ac.uk/psychol/abg/pipeline_connectivity)
