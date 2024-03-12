# Rockland dataset

Data contains TR 645 and TR 1400 acquisitions using a ON/OFF checkerboard protocol.
This is a subset (25) of participants, the full dataset has hundreds of participants and lots of different scan modalities / experimental tasks.

Again data was installed on virtual box, using the python script as described in:
http://fcon_1000.projects.nitrc.org/indi/enhanced/neurodata.html (heading: Using the Downloader Script for S3)

The general pipeline consists of the following steps.

1) Download the raw data.
   The `download_rockland_raw_bids.py` script will pull all raw data into `/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/CHECKERBOARD645/raw/`.

2) Preprocess the data.
   This can be done with either one of two pipelines.

3) Extract time series.

4) Plot time series and assess data quality.

## raw

Raw data downloaded from rockland server.
For each session this contains a T1 structural imagine in 'anat' and functional runs in 'func'.
The functional runs include a checkboard task and resting state (both 1.4 TR).
Note the file ...events.tsv contains the timings of the visual stimulation ON / OFF.

## mpp

Preprocessing for the data.
See scripts in /scripts/mpp for details:
- /scripts/mpp/src/nki_bids_mpp.sh : main preprocessing scripts
- /scripts/mpp/src/reg_fun2mni.sh : normalisation scripts
- /scripts/mpp/submit...sh scripts : batch scripts for running preprocessing (with SLURM)

## mpp_mni

Normalised preprocessed data in MNI.
This contains dirs for:

- rsfmri data
- task data aggressive aroma cleanup (noise regressed) TODO: check the noise methods with Soroosh
- task data non-agressive aroma cleanup (noise subtracted)

the scripts for normalisation can be round at /scripts/matlab-spm/mpp_to_mni.m

## results

This folder contains the output results files from the Matlab script.

## node_timeseries

This folder contains the final time series data, one `.csv` file per subject.
These will be loaded by any subsequent analysis in Python.

## Source
http://fcon_1000.projects.nitrc.org/indi/enhanced/
